"""final.py - Bước 4: chạy TEST đúng một lần mỗi seed cho chung kết (F01) và mốc (T00 + I00), rồi eval.py.

Thứ tự bắt buộc: chỉ gọi sau khi decisions.json đã có "recipe" (Bước 2) và "inference" (Bước 3).
Mỗi lần đọc ảnh test ghi vào <runs>/test_access.log; gọi lại cho cùng exp_id/seed sẽ bị từ chối.

File ghi ra (định dạng eval.save_predictions):
    predictions/T00_seed<k>_test.csv       mốc: công thức nền + 1 view, không TS
    predictions/T00_seed<k>_val.csv
    predictions/F01uncal_seed<k>_test.csv  chung kết trước temperature scaling (I4a)
    predictions/F01_seed<k>_test.csv       chung kết sau TS (T khớp trên val của chính seed đó)
    predictions/F01_seed<k>_val.csv        dự đoán val của chung kết (I4b)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import dataset as D
from experiments import FINAL_SEEDS, T00_SEEDS, Paths, load_decisions, save_decision
from step3_inference import load_run_model, predict_spec, spec_latency, spec_name
from train import REPO_DIR, Config, _log_test_access


def _test_once(paths: Paths, exp_id: str, seed: int):
    _log_test_access(Config(exp_id=exp_id, seed=seed, out_dir=paths.out_dir))


def predict_run(paths: Paths, run_exp: str, seed: int, spec: dict, out_exp: str, ts: bool, uncal_exp: str | None):
    """Dự đoán val (+ khớp T) rồi test MỘT lần cho một seed; ghi các file predictions."""
    import torch
    from eval import save_predictions
    from inference import apply_temperature, fit_temperature, softmax_np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_df, test_df = D.load_split(paths.labels_dir, 0)
    model, cfg, dc = load_run_model(paths, run_exp, seed, device)
    pd_ = Path(paths.pred_dir)
    nv, yv, lv = predict_spec(model, spec, paths, val_df, dc, cfg.img_size, device)
    T = fit_temperature(lv, yv) if ts else 1.0
    save_predictions(pd_ / f"{out_exp}_seed{seed}_val.csv", nv, yv, apply_temperature(lv, T))
    _test_once(paths, out_exp, seed)                          # ---- truy cập test (một lần) ----
    nt, yt, lt = predict_spec(model, spec, paths, test_df, dc, cfg.img_size, device)
    np.save(Path(paths.out_dir) / run_exp / f"seed{seed}" / f"test_logits_{out_exp}.npy", lt)
    save_predictions(pd_ / f"{out_exp}_seed{seed}_test.csv", nt, yt, apply_temperature(lt, T))
    if uncal_exp:
        save_predictions(pd_ / f"{uncal_exp}_seed{seed}_test.csv", nt, yt, softmax_np(lt))
    return {"seed": seed, "T": T, "run": f"{run_exp}/seed{seed}", "spec": spec_name(spec)}


def run_eval_cli(paths: Paths, args: list[str], out_name: str) -> str:
    """Gọi eval.py gốc (không sửa) và lưu stdout vào work/eval_out/<out_name>.md."""
    cmd = [sys.executable, str(REPO_DIR / "eval.py"), *args]
    import os
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    text = res.stdout + (("\n[stderr]\n" + res.stderr) if res.returncode else "")
    (paths.sub("eval_out") / f"{out_name}.md").write_text(text, encoding="utf-8")
    if res.returncode:
        raise RuntimeError(f"eval.py lỗi ({res.returncode}):\n{text}")
    return text


def run_final(paths: Paths) -> dict:
    """Bước 4 đầy đủ. Trả về dict tóm tắt và ghi work/final.json."""
    dec = load_decisions(paths)
    if "recipe" not in dec or "inference" not in dec:
        raise RuntimeError("chưa chốt công thức (Bước 2) và phương pháp suy luận (Bước 3) trên val")
    spec = dec["inference"]["spec"]
    single = {"views": "single", "img_size": None, "crop_pct": 0.875}
    t00_single = {"views": "single", "img_size": None, "crop_pct": 0.875}
    info = {"final": [], "baseline": []}
    for s in FINAL_SEEDS:
        if not (Path(paths.pred_dir) / f"F01_seed{s}_test.csv").exists():
            info["final"].append(predict_run(paths, "F01", s, spec, "F01", ts=True, uncal_exp="F01uncal"))
    for s in T00_SEEDS:
        if not (Path(paths.pred_dir) / f"T00_seed{s}_test.csv").exists():
            info["baseline"].append(predict_run(paths, "T00", s, t00_single, "T00", ts=False, uncal_exp=None))

    # Độ trễ của cấu hình thời gian thực (= cấu hình chung kết nếu p95 <= 100 ms, ngược lại 1 view)
    import torch
    model, cfg, _ = load_run_model(paths, "F01", FINAL_SEEDS[0])
    lat = spec_latency(model, spec, cfg.img_size, 1, iters=200, label="F01 final b1")
    rt_spec, rt_lat = spec, lat
    if lat["p95_ms"] > 100:
        rt_spec = single
        rt_lat = spec_latency(model, single, cfg.img_size, 1, iters=200, label="F01 1-view b1")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    lab = paths.labels_dir
    common = ["--test-csv", f"{lab}/test_subset0.csv", "--labels", f"{lab}/labels.csv", "--out",
              str(paths.sub("eval_out"))]
    pdir = paths.pred_dir
    for tag in ("F01", "F01uncal", "T00"):
        run_eval_cli(paths, ["score", "--pred", f"{pdir}/{tag}_seed*_test.csv", "--tag", tag, *common], f"score_{tag}")
    grade = run_eval_cli(paths, [
        "grade", "--final", f"{pdir}/F01_seed*_test.csv", "--baseline", f"{pdir}/T00_seed*_test.csv",
        "--uncal", f"{pdir}/F01uncal_seed*_test.csv", "--final-val", f"{pdir}/F01_seed*_val.csv",
        "--val-csv", f"{lab}/val_subset0.csv", "--latency-p95-ms", f"{rt_lat['p95_ms']:.3f}",
        "--latency-method", "proper", *common], "grade")
    print(grade)
    result = {"spec": spec, "realtime_spec": rt_spec, "realtime_latency": rt_lat, "final_latency": lat,
              "runs": info, "grade_md": grade}
    (Path(paths.work_dir) / "final.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    save_decision(paths, "test_done", {"seeds": list(FINAL_SEEDS), "note": "test đã mở; không chỉnh cấu hình nữa"})
    return result

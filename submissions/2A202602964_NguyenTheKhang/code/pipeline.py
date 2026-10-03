"""pipeline.py - chạy toàn bộ bài lab theo đúng thứ tự GUIDE, có thể tiếp tục khi Colab bị ngắt.

    python pipeline.py --step all  --work /content/drive/MyDrive/deepweeds_work
    python pipeline.py --step 1    (hoặc 0, 2, 3, 4, 5)

Mỗi bước đọc quyết định của bước trước từ work/decisions.json, mỗi lần chạy đã xong được bỏ qua
(train.run đọc lại summary.json), nên chạy lại cùng lệnh sau khi mất kết nối là an toàn.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import experiments as X
from experiments import Paths


def step0(paths: Paths) -> None:
    from step0_checks import run_step0
    if not (Path(paths.work_dir) / "step0.json").exists():
        run_step0(paths)


def step1(paths: Paths, with_dinov2: bool = True) -> dict:
    X.run_many(X.backbone_configs(paths))
    if with_dinov2:
        try:
            from bonus_dinov2 import run_dinov2_probe
            run_dinov2_probe(paths)
        except Exception as e:  # noqa: BLE001  điểm thưởng: không làm hỏng bài chính
            print("DINOv2 probe thất bại:", e)
    summaries = [X.load_summary(paths, e) for e, *_ in X.BACKBONES]
    pick, reason = X.select_backbone(summaries)
    X.save_decision(paths, "backbone", {"backbone": pick["backbone"], "exp_id": pick["exp_id"], "reason": reason})
    print("Chọn backbone:", reason)
    return pick


def _reuse_as_t00(paths: Paths, b_exp: str) -> None:
    """T00 seed 0 có cấu hình GIỐNG HỆT lần chạy Bước 1 của backbone được chọn (cùng seed 0):
    dùng lại thay vì huấn luyện lần nữa (ghi rõ trong báo cáo)."""
    src, dst = Path(paths.out_dir) / b_exp / "seed0", Path(paths.out_dir) / "T00" / "seed0"
    if (dst / "summary.json").exists() or not (src / "summary.json").exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)
    for name in ("summary.json", "config.json"):
        d = json.loads((dst / name).read_text(encoding="utf-8"))
        cfg = d["config"]
        cfg["exp_id"], cfg["desc"] = "T00", "baseline"
        if name == "summary.json":
            d["exp_id"], d["desc"], d["reused_from"] = "T00", "baseline", b_exp
        (dst / name).write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    import numpy as np
    import pandas as pd
    from train import plot_curves
    h = pd.read_csv(dst / "history.csv").to_dict("records")
    s = json.loads((dst / "summary.json").read_text(encoding="utf-8"))
    plot_curves(h, dst / "curve.png", f"T00 | {s['backbone']} | seed 0 | baseline (= {b_exp})",
                np.load(dst / "lr_steps.npy").tolist())


def step2(paths: Paths) -> None:
    dec = X.load_decisions(paths)["backbone"]
    bb = dec["backbone"]
    _reuse_as_t00(paths, dec["exp_id"])
    X.run_many([X.t00_config(paths, bb, s) for s in X.T00_SEEDS])
    X.run_many(X.ablation_configs(paths, bb))
    mean, std, f1 = X.noise_std(paths)
    combos = X.combo_specs(paths)
    X.save_decision(paths, "noise", {"T00_val_f1": f1, "mean": mean, "std": std})
    X.save_decision(paths, "combos", combos)
    X.run_many(X.combo_configs(paths, bb))
    recipe, reason = X.select_recipe(paths, bb)
    base = X.t00_config(paths, bb)
    diff = {k: v for k, v in dataclasses.asdict(recipe).items()
            if k not in ("exp_id", "desc") and getattr(base, k) != v}
    X.save_decision(paths, "recipe", {"exp_id": recipe.exp_id, "config": dataclasses.asdict(recipe), "diff": diff,
                                      "diff_str": ", ".join(f"{k}={v}" for k, v in diff.items()) or "như T00",
                                      "summary": f"{bb} + {', '.join(f'{k}={v}' for k, v in diff.items()) or 'T00'}",
                                      "reason": reason})
    print("Công thức chung kết:", reason)


def step3(paths: Paths) -> None:
    from step3_inference import run_step3
    dec = X.load_decisions(paths)
    if "inference" in dec:
        return
    res = run_step3(paths, dec["recipe"]["exp_id"], 0)
    X.save_decision(paths, "inference", res["decision"])
    print("Phương pháp suy luận:", res["decision"]["reason"])


def step4(paths: Paths) -> None:
    from final import run_final
    dec = X.load_decisions(paths)
    recipe = X.Config(**{k: v for k, v in dec["recipe"]["config"].items() if k in X.Config.__dataclass_fields__})
    X.run_many(X.final_configs(paths, recipe))
    run_final(paths)


def step5(paths: Paths, submission: Path) -> None:
    from make_report import write_report
    from make_results import build_all
    build_all(paths, submission)
    write_report(paths, submission)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["all", "0", "1", "2", "3", "4", "5"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--submission", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--no-dinov2", action="store_true")
    a = ap.parse_args()
    paths = Paths(a.images, a.labels, a.cache or None, a.work, a.workers)
    steps = ["0", "1", "2", "3", "4", "5"] if a.step == "all" else [a.step]
    for s in steps:
        print(f"===== Bước {s} =====")
        {"0": lambda: step0(paths), "1": lambda: step1(paths, not a.no_dinov2), "2": lambda: step2(paths),
         "3": lambda: step3(paths), "4": lambda: step4(paths), "5": lambda: step5(paths, Path(a.submission))}[s]()


if __name__ == "__main__":
    main()

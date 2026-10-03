"""step3_inference.py - Bước 3: so sánh phương pháp suy luận trên VAL (không huấn luyện lại).

Một "spec" suy luận mô tả đầy đủ cách chạy model, để áp dụng y hệt cho val (chọn) và test (Bước 4):
    {"views": "single" | "hflip" | "crop10" | "crop5" | "multiscale",
     "space": "prob" | "logit", "img_size": 224, "crop_pct": 0.875, "scales": [224, 256, 288]}
predict_spec(...) trả về "logit" của dự đoán gộp: với gộp xác suất là log(p̄) (softmax(log p̄) = p̄),
nên temperature scaling áp dụng được thống nhất cho mọi spec.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import dataset as D
from experiments import BACKBONES, Paths, load_summary, noise_std, smoke_limit
from train import Config, metrics_from_logits

INFER_TIE = 0.002      # chênh macro-F1 val nhỏ hơn mức này giữa hai cách suy luận CÙNG model: coi như ngang
REALTIME_P95_MS = 100.0


# --------------------------------------------------------------------------- #
# Nạp model đã huấn luyện
# --------------------------------------------------------------------------- #
def load_run_model(paths: Paths, exp_id: str, seed: int = 0, device=None):
    """(model eval trên device, Config, data_config) từ <runs>/<exp_id>/seed<k>/best.pt."""
    import torch
    from model import build_model, data_config
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd = Path(paths.out_dir) / exp_id / f"seed{seed}"
    meta = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    c = meta["config"]
    cfg = Config(**{k: v for k, v in c.items() if k in Config.__dataclass_fields__})
    model = build_model(cfg.backbone, pretrained=False, num_classes=D.NUM_CLASSES, drop_rate=cfg.drop_rate,
                        init="finetune", drop_path_rate=cfg.drop_path_rate, img_size=cfg.img_size)
    model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    dc = meta.get("data_config") or data_config(model)
    return model, cfg, dc


def eval_loader(paths: Paths, df, dc: dict, img_size: int, crop_pct: float, batch_size: int = 128):
    cache = None
    if paths.cache_dir:
        tr, va, te = D.load_split(paths.labels_dir, 0)
        cache = D.ImageCache.build(list(tr.Filename) + list(va.Filename) + list(te.Filename),
                                   paths.images_dir, paths.cache_dir)
    t = D.eval_transform(img_size, crop_pct, tuple(dc["mean"]), tuple(dc["std"]))
    return D.make_loader(df, paths.images_dir, t, batch_size, False, None, paths.num_workers, cache, 0)


# --------------------------------------------------------------------------- #
# Spec -> dự đoán
# --------------------------------------------------------------------------- #
def spec_name(spec: dict) -> str:
    v = spec["views"]
    s = f"{v}@{spec.get('img_size') or 'train'}/c{spec.get('crop_pct', 0.875)}"
    if v == "multiscale":
        s = f"multiscale{tuple(spec['scales'])}"
    return s + (f" [{spec['space']}]" if v != "single" else "")


def k_views(spec: dict) -> int:
    return {"single": 1, "hflip": 2, "crop5": 5, "crop10": 10}.get(spec["views"], len(spec.get("scales", [])))


def _views_fn(spec: dict, train_size: int):
    from inference import view_identity, views_hflip, views_multicrop, views_multiscale
    v = spec["views"]
    if v == "single":
        return lambda x: [view_identity(x)]
    if v == "hflip":
        return views_hflip
    if v in ("crop5", "crop10"):
        return lambda x: views_multicrop(x, train_size, flip=(v == "crop10"))
    if v == "multiscale":
        return lambda x: views_multiscale(x, spec["scales"])
    raise ValueError(v)


def _loader_geometry(spec: dict, train_size: int) -> tuple[int, float]:
    if spec["views"] in ("crop5", "crop10"):
        return 256, 1.0       # ảnh gốc 256 đầy đủ, rồi cắt 5 crop kích thước train_size
    return spec.get("img_size") or train_size, spec.get("crop_pct", 0.875)


def predict_spec(model, spec: dict, paths: Paths, df, dc: dict, train_size: int, device):
    """(filenames, y_true, logits_gộp) theo spec."""
    from inference import aggregate_views, predict_views
    size, crop = _loader_geometry(spec, train_size)
    loader = eval_loader(paths, df, dc, size, crop)
    names, y, outs = predict_views(model, loader, device, _views_fn(spec, train_size))
    if len(outs) == 1:
        return names, y, outs[0].astype(np.float64)
    if spec.get("space", "prob") == "logit":
        return names, y, np.mean(np.stack(outs), 0).astype(np.float64)
    return names, y, np.log(np.clip(aggregate_views(outs, "prob"), 1e-12, None))


def spec_latency(model, spec: dict, train_size: int, batch: int = 1, dtype: str = "fp32",
                 iters: int = 100, label: str = "") -> dict:
    """Độ trễ của cả phương pháp (forward K view + gộp), đo đúng cách bằng benchmark.bench."""
    import torch
    from benchmark import _device_info, bench
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu, sync = _device_info(device)
    size, _ = _loader_geometry(spec, train_size)
    x = torch.randn(batch, 3, size, size, device=device)
    vf = _views_fn(spec, train_size)
    use_amp = dtype == "amp" and device == "cuda"

    @torch.inference_mode()
    def fn():
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            vs = vf(x)
            if len({tuple(v.shape) for v in vs}) == 1:     # cùng kích thước -> một batch K*B
                out = model(torch.cat(vs, 0)).float().softmax(-1).view(len(vs), batch, -1).mean(0)
            else:
                out = torch.stack([model(v).float().softmax(-1) for v in vs]).mean(0)
        return out
    r = bench(fn, warmup=10, iters=iters, sync=sync)
    return {"config": label or spec_name(spec), "gpu": gpu, "dtype": dtype, "batch": batch, "img_size": size,
            "fused_bn": False, "k_views": k_views(spec), "p50_ms": r["p50"], "p95_ms": r["p95"],
            "p99_ms": r["p99"], "mean_ms": r["mean"], "images_per_s": batch / (r["p50"] / 1000.0),
            "n_iters": r["n"], "warmup": r["warmup"], "torch": torch.__version__,
            "preprocessing_included": False}


def models_latency(models: list, img_sizes: list[int], batch: int = 1, iters: int = 100, label: str = "") -> dict:
    """Độ trễ ensemble: chạy lần lượt từng model rồi trung bình softmax."""
    import torch
    from benchmark import _device_info, bench
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu, sync = _device_info(device)
    xs = [torch.randn(batch, 3, s, s, device=device) for s in img_sizes]

    @torch.inference_mode()
    def fn():
        return torch.stack([m(x).float().softmax(-1) for m, x in zip(models, xs)]).mean(0)
    r = bench(fn, warmup=10, iters=iters, sync=sync)
    return {"config": label, "gpu": gpu, "dtype": "fp32", "batch": batch, "img_size": max(img_sizes),
            "fused_bn": False, "k_views": len(models), "p50_ms": r["p50"], "p95_ms": r["p95"], "p99_ms": r["p99"],
            "mean_ms": r["mean"], "images_per_s": batch / (r["p50"] / 1000.0), "n_iters": r["n"],
            "warmup": r["warmup"], "torch": torch.__version__, "preprocessing_included": False}


# --------------------------------------------------------------------------- #
# Bước 3
# --------------------------------------------------------------------------- #
def _row(exp_id, method, model_desc, k, y, logits, lat_b1, lat_b32, base_p50, extra=None):
    from eval import ece_score
    from inference import softmax_np
    m = metrics_from_logits(y, logits)
    r = {"exp_id": exp_id, "method": method, "model": model_desc, "K": k,
         "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": ece_score(softmax_np(logits), y),
         "val_f1_chinee": float(m["f1"][0]), "val_f1_snake": float(m["f1"][7]),
         "p50_ms": lat_b1.get("p50_ms") if lat_b1 else None, "p95_ms": lat_b1.get("p95_ms") if lat_b1 else None,
         "p99_ms": lat_b1.get("p99_ms") if lat_b1 else None,
         "throughput_b32": lat_b32.get("images_per_s") if lat_b32 else None}
    r["rel_cost_vs_I00"] = (r["p50_ms"] / base_p50) if (r["p50_ms"] and base_p50) else None
    r.update(extra or {})
    return r


def run_step3(paths: Paths, ref_exp: str, ref_seed: int = 0, chosen_backbone_exp: str | None = None) -> dict:
    """Chạy toàn bộ Bước 3 trên model tham chiếu (công thức đã chọn ở Bước 2, seed 0). Ghi work/step3.json."""
    import torch
    from eval import ece_score
    from inference import (apply_temperature, crossfit_temperature_ece, ensemble_probs, fit_temperature,
                           fuse_conv_bn, recalibrate_bn, softmax_np, uniform_soup)

    out_path = Path(paths.work_dir) / "step3.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_df, _ = D.load_split(paths.labels_dir, 0)
    val_df = smoke_limit(val_df)
    model, cfg, dc = load_run_model(paths, ref_exp, ref_seed, device)
    s0 = cfg.img_size
    desc = f"{ref_exp}/seed{ref_seed} ({cfg.backbone})"
    rows, latency, specs = [], [], {}

    def do(exp_id, method, spec, extra=None):
        try:
            names, y, lg = predict_spec(model, spec, paths, val_df, dc, s0, device)
            l1 = spec_latency(model, spec, s0, 1, label=f"{exp_id} {spec_name(spec)}")
            l32 = spec_latency(model, spec, s0, 32, iters=50, label=f"{exp_id} {spec_name(spec)}")
            latency.extend([l1, l32])
            base = rows[0]["p50_ms"] if rows else l1["p50_ms"]
            rows.append(_row(exp_id, method, desc, k_views(spec), y, lg, l1, l32, base,
                             {"spec": spec, "spec_name": spec_name(spec), **(extra or {})}))
            specs[exp_id] = spec
            print(f"{exp_id:14s} {method:40s} F1 {rows[-1]['val_macro_f1']:.4f}  p95 {l1['p95_ms']:.2f} ms")
            return y, lg
        except Exception as e:  # noqa: BLE001  ví dụ Swin không chạy được ở độ phân giải khác
            rows.append({"exp_id": exp_id, "method": method, "model": desc, "K": k_views(spec),
                         "spec": spec, "spec_name": spec_name(spec), "note": f"không áp dụng: {e}"[:200]})
            print(f"{exp_id} lỗi: {e}")
            return None, None

    single = {"views": "single", "img_size": s0, "crop_pct": 0.875}
    y_val, l_i00 = do("I00", "1 view (mốc)", single)
    do("I01", "TTA lật ngang K=2, gộp xác suất", {"views": "hflip", "space": "prob", "img_size": s0, "crop_pct": 0.875})
    do("I01L", "TTA lật ngang K=2, gộp logit (I03)", {"views": "hflip", "space": "logit", "img_size": s0, "crop_pct": 0.875})
    do("I02", "TTA 10-crop (5 crop + lật), gộp xác suất", {"views": "crop10", "space": "prob"})
    do("I02L", "TTA 10-crop, gộp logit (I03)", {"views": "crop10", "space": "logit"})
    do("I02S", "TTA đa tỉ lệ K=3, gộp xác suất",
       {"views": "multiscale", "space": "prob", "img_size": s0, "crop_pct": 0.875, "scales": [s0, s0 + 32, s0 + 64]})
    for size in (224, 256, 288, 320):
        for crop in (0.875, 1.0):
            if size == s0 and crop == 0.875:
                continue
            do(f"I04_{size}_c{crop}", f"Độ phân giải kiểm tra {size}, crop_pct {crop}",
               {"views": "single", "img_size": size, "crop_pct": crop})

    # I05 ensemble: top backbone của Bước 1 (logit val đã lưu, cùng tiền xử lý 1 view)
    ens_info = {}
    bsum = [load_summary(paths, e) for e, *_ in BACKBONES]
    bsum = sorted([s for s in bsum if s], key=lambda s: -s["val_macro_f1"])
    for k in (2, 3):
        if len(bsum) >= k:
            members = bsum[:k]
            probs = ensemble_probs([softmax_np(np.load(Path(paths.out_dir) / s["exp_id"] / "seed0" / "val_logits.npy"))
                                    for s in members])
            ms = [load_run_model(paths, s["exp_id"], 0, device)[0] for s in members]
            l1 = models_latency(ms, [s["img_size"] for s in members], 1, label=f"I05 ensemble top{k}")
            l32 = models_latency(ms, [s["img_size"] for s in members], 32, iters=50, label=f"I05 ensemble top{k}")
            latency.extend([l1, l32])
            del ms
            rows.append(_row(f"I05_top{k}", f"Ensemble {k} backbone (Bước 1, gộp xác suất)",
                             " + ".join(s["backbone"] for s in members), k, y_val, np.log(np.clip(probs, 1e-12, None)),
                             l1, l32, rows[0]["p50_ms"]))
            ens_info[k] = [s["exp_id"] for s in members]
    # Ensemble 3 seed của T00 (công thức nền)
    try:
        t00 = [softmax_np(np.load(Path(paths.out_dir) / "T00" / f"seed{s}" / "val_logits.npy")) for s in (0, 1, 2)]
        tm = [load_run_model(paths, "T00", s, device)[0] for s in (0, 1, 2)]
        l1 = models_latency(tm, [s0] * 3, 1, label="I05 ensemble T00 x3 seed")
        latency.append(l1)
        rows.append(_row("I05_seeds", "Ensemble 3 seed T00 (gộp xác suất)", "T00 seed0-2", 3, y_val,
                         np.log(np.clip(ensemble_probs(t00), 1e-12, None)), l1, None, rows[0]["p50_ms"]))
        # I06 model soup: trung bình trọng số 3 seed T00 + hiệu chỉnh lại BN trên train
        soup_model, tcfg, tdc = load_run_model(paths, "T00", 0, device)
        soup_model.load_state_dict(uniform_soup([m.state_dict() for m in tm]))
        tr_df, _, _ = D.load_split(paths.labels_dir, 0)
        tr_df = smoke_limit(tr_df)
        recalibrate_bn(soup_model, eval_loader(paths, tr_df.sample(frac=1.0, random_state=0), tdc, s0, 0.875, 64),
                       device, max_batches=60)
        _, ys, ls = predict_spec(soup_model, single, paths, val_df, tdc, s0, device)
        rows.append(_row("I06_soup", "Model soup đồng đều 3 seed T00 (+ hiệu chỉnh BN)", "soup(T00 seed0-2)", 1,
                         ys, ls, rows[0] and {"p50_ms": rows[0]["p50_ms"], "p95_ms": rows[0]["p95_ms"],
                                              "p99_ms": rows[0]["p99_ms"]}, None, rows[0]["p50_ms"],
                         {"T00_seed_val_f1": [load_summary(paths, "T00", s)["val_macro_f1"] for s in (0, 1, 2)]}))
        del tm, soup_model
    except FileNotFoundError as e:
        print("bỏ qua ensemble/soup T00:", e)

    # I06 EMA: so sánh trọng số EMA và trọng số thường của cùng lần chạy T12 (cùng epoch tốt nhất)
    ema_info = None
    hist_p = Path(paths.out_dir) / "T12" / "seed0" / "history.csv"
    if hist_p.exists():
        import pandas as pd
        h = pd.read_csv(hist_p)
        b = h.loc[h.val_macro_f1.idxmax()]
        ema_info = {"best_epoch": int(b.epoch), "ema_val_f1": float(b.val_macro_f1),
                    "raw_val_f1_same_epoch": float(b.raw_val_macro_f1),
                    "raw_val_f1_best_any_epoch": float(h.raw_val_macro_f1.max())}
        rows.append({"exp_id": "I06_ema", "method": "Trọng số EMA (T12) so với trọng số thường cùng lần chạy",
                     "model": "T12/seed0", "K": 1, "val_macro_f1": ema_info["ema_val_f1"],
                     "note": f"raw cùng epoch {ema_info['raw_val_f1_same_epoch']:.4f}, raw tốt nhất "
                             f"{ema_info['raw_val_f1_best_any_epoch']:.4f}; chi phí suy luận như 1 view",
                     "rel_cost_vs_I00": 1.0})

    # I07 temperature scaling (T khớp trên val; báo cả ECE val cross-fit cho khách quan)
    T = fit_temperature(l_i00, y_val)
    p_before, p_after = softmax_np(l_i00), apply_temperature(l_i00, T)
    cf = crossfit_temperature_ece(l_i00, y_val, ece_score)
    ts = {"T": T, "ece_before": ece_score(p_before, y_val), "ece_after_in_sample": ece_score(p_after, y_val),
          **cf, "nll_before": float(-np.log(p_before[np.arange(len(y_val)), y_val]).mean()),
          "nll_after": float(-np.log(p_after[np.arange(len(y_val)), y_val]).mean())}
    rows.append(_row("I07", f"Temperature scaling (T = {T:.3f}, khớp trên val)", desc, 1, y_val,
                     l_i00 / T, {k: rows[0][k] for k in ("p50_ms", "p95_ms", "p99_ms")}, None, rows[0]["p50_ms"],
                     {"note": f"ECE {ts['ece_before']:.4f} -> {ts['ece_after_in_sample']:.4f} (in-sample), "
                              f"cross-fit {ts['ece_crossfit']:.4f}; accuracy không đổi"}))

    # I08 gộp BN + FP16 / channels_last
    from benchmark import latency_report
    from inference import predict_logits
    loader = eval_loader(paths, val_df, dc, s0, 0.875)
    bn_target = (model, cfg, dc, desc)
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    if not has_bn:   # ConvNeXt/ViT/Swin dùng LayerNorm: minh hoạ gộp BN trên ResNet-50 của Bước 1
        m2, c2, d2 = load_run_model(paths, "B01", 0, device)
        bn_target = (m2, c2, d2, "B01/seed0 (resnet50)")
    bm, bcfg, bdc, bdesc = bn_target
    fused = fuse_conv_bn(bm, img_size=bcfg.img_size)
    bl = eval_loader(paths, val_df, bdc, bcfg.img_size, 0.875)
    for tag, mm, fb in (("unfused", bm, False), ("fused", fused, True)):
        _, yb, lb = predict_logits(mm, bl, device, amp=False)
        l1 = latency_report(mm, 1, bcfg.img_size, "fp32", str(device), fused_bn=fb, label=f"I08 {bdesc} {tag}")
        latency.append(l1)
        rows.append(_row(f"I08_bn_{tag}", f"{'Gộp' if fb else 'Không gộp'} BN, FP32", bdesc, 1, yb, lb, l1, None,
                         None, {"n_fused": getattr(mm, "n_fused", 0), "max_abs_diff": getattr(mm, "max_abs_diff", 0.0)}))
    for dtype, cl in (("fp32", True), ("amp", False), ("amp", True), ("fp16", True)):
        try:
            l1 = latency_report(model, 1, s0, dtype, str(device), channels_last=cl, label=f"I08 {desc} {dtype} cl={cl}")
            l32 = latency_report(model, 32, s0, dtype, str(device), channels_last=cl, iters=50,
                                 label=f"I08 {desc} {dtype} cl={cl}")
            latency.extend([l1, l32])
            if dtype == "fp16" and device.type == "cuda":
                import copy
                mh = copy.deepcopy(model).half()
                ys_, lh = [], []
                with torch.inference_mode():
                    for x, y, _ in loader:
                        lh.append(mh(x.to(device).half()).float().cpu())
                        ys_.append(y)
                yh, lh = torch.cat(ys_).numpy(), torch.cat(lh).numpy()
                del mh
            else:
                _, yh, lh = predict_logits(model, loader, device, amp=(dtype == "amp"))
            rows.append(_row(f"I08_{dtype}{'_cl' if cl else ''}", f"{dtype.upper()}{' + channels_last' if cl else ''}",
                             desc, 1, yh, lh, l1, l32, rows[0]["p50_ms"]))
        except Exception as e:  # noqa: BLE001
            print("I08", dtype, "lỗi:", e)
    # Độ trễ TTA tuần tự vs batched (kiểm tra "chi phí ~ K lần", slide trang 63)
    from benchmark import tta_latency
    tta = tta_latency(model, 2, batch_size=1, img_size=s0, device=str(device))
    latency.extend([tta["sequential"] | {"config": "TTA hflip K=2 sequential"},
                    tta["batched"] | {"config": "TTA hflip K=2 batched"}])

    # Chọn phương pháp suy luận cho chung kết (chỉ phương pháp 1 model, theo val)
    cands = [r for r in rows if r.get("spec") and r.get("val_macro_f1") is not None and r.get("p95_ms")
             and r["p95_ms"] <= REALTIME_P95_MS]
    best = max(cands, key=lambda r: r["val_macro_f1"])
    near = [r for r in cands if best["val_macro_f1"] - r["val_macro_f1"] <= INFER_TIE]
    pick = min(near, key=lambda r: r["p50_ms"])
    decision = {"spec": pick["spec"], "exp_id": pick["exp_id"], "temperature_scaling": True,
                "reason": f"macro-F1 val cao nhất: {best['exp_id']} ({best['val_macro_f1']:.4f}); trong nhóm cách "
                          f"<= {INFER_TIE} chọn {pick['exp_id']} ({pick['spec_name']}, F1 {pick['val_macro_f1']:.4f}, "
                          f"p50 {pick['p50_ms']:.2f} ms) vì rẻ nhất. Luôn áp dụng temperature scaling (T khớp trên val "
                          f"của từng seed) vì không đổi argmax."}
    result = {"ref_exp": ref_exp, "ref_seed": ref_seed, "backbone": cfg.backbone, "rows": rows, "latency": latency,
              "temperature": ts, "ema": ema_info, "ensembles": ens_info, "tta_latency_check": {
                  "k_times_p50": tta["k_times_p50"], "sequential_p50": tta["sequential"]["p50_ms"],
                  "batched_p50": tta["batched"]["p50_ms"]}, "decision": decision,
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return result

"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Hoàn thiện từ starter/benchmark.py. Quy tắc đo được cài đặt:
  - warmup: bỏ `warmup` (mặc định 10, tối thiểu 10) lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU mỗi lần đo
  - `iters` (mặc định 100, tối thiểu 50) lần đo, báo cáo p50 / p95 / p99 / mean
  - ghi GPU, dtype, batch, độ phân giải, gộp BN, channels_last, phiên bản torch
  - KHÔNG tính tiền xử lý (đầu vào là tensor ngẫu nhiên đã nằm sẵn trên GPU); chỉ đo forward
    (+ softmax/gộp view với TTA). Lựa chọn này ghi trong báo cáo.
"""
from __future__ import annotations

import copy
import time

import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian `fn()` (mili-giây) với warmup và đồng bộ trước/sau mỗi lần đo."""
    if warmup < 10 or iters < 50:
        raise ValueError("GUIDE 4.1: warmup >= 10 và iters >= 50")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = np.empty(iters)
    for i in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)), "mean": float(times.mean()),
            "std": float(times.std(ddof=1)), "n": int(iters), "warmup": int(warmup)}


def _device_info(device: str) -> tuple[str, object]:
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0), torch.cuda.synchronize
    import platform
    return f"CPU ({platform.processor() or platform.machine()})", None


def prepare_model(model, dtype: str = "fp32", device: str = "cuda", channels_last: bool = False):
    """Bản sao của model ở chế độ eval, đúng dtype/memory format. Không sửa model gốc."""
    import torch
    m = copy.deepcopy(model).eval().to(device)
    if dtype == "fp16":
        m = m.half()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    return m


def make_forward(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                 channels_last: bool = False, k_views: int = 1, tta_mode: str = "batched"):
    """Trả về fn() chạy một lượt suy luận (forward + softmax, gộp K view nếu TTA).

    tta_mode="batched": K view ghép thành một batch K*B (một lần forward).
    tta_mode="sequential": K lần forward liên tiếp (đúng định nghĩa "chi phí ~ K lần" của slide).
    """
    import torch
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        x = x.half()
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    use_amp = dtype == "amp" and device.startswith("cuda")

    def views():
        if k_views == 1:
            return [x]
        return [x if i % 2 == 0 else torch.flip(x, dims=[3]) for i in range(k_views)]

    @torch.inference_mode()
    def fn():
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            vs = views()
            if k_views > 1 and tta_mode == "batched":
                out = model(torch.cat(vs, 0)).float().softmax(-1)
                return out.view(k_views, batch_size, -1).mean(0)
            return torch.stack([model(v).float().softmax(-1) for v in vs]).mean(0)
    return fn


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, channels_last: bool = False, fused_bn: bool = False,
                   k_views: int = 1, tta_mode: str = "batched", label: str = "") -> dict:
    """Đo độ trễ forward với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict ghi thẳng vào sheet `Latency`. dtype: "fp32" | "amp" (autocast) | "fp16" (model.half()).
    Lưu ý slide trang 73: ở batch 1, AMP có thể CHẬM hơn FP32, nên luôn đo thật.
    """
    import torch
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"
        if dtype in ("amp", "fp16"):
            dtype = "fp32"   # CPU: không đo fp16
    gpu, sync = _device_info(device)
    m = prepare_model(model, dtype, device, channels_last)
    fn = make_forward(m, batch_size, img_size, dtype, device, channels_last, k_views, tta_mode)
    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    del m
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"config": label, "gpu": gpu, "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "fused_bn": fused_bn, "channels_last": channels_last, "k_views": k_views,
            "tta_mode": tta_mode if k_views > 1 else "-",
            "p50_ms": r["p50"], "p95_ms": r["p95"], "p99_ms": r["p99"], "mean_ms": r["mean"],
            "images_per_s": batch_size / (r["p50"] / 1000.0), "n_iters": r["n"], "warmup": r["warmup"],
            "torch": torch.__version__, "preprocessing_included": False}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ TTA K view (chạy tuần tự K lần forward), so với K * p50 của 1 view (slide trang 63)."""
    one = latency_report(model, k_views=1, **kw)
    seq = latency_report(model, k_views=k_views, tta_mode="sequential", **kw)
    bat = latency_report(model, k_views=k_views, tta_mode="batched", **kw)
    return {"one_view": one, "sequential": seq, "batched": bat,
            "k_times_p50": k_views * one["p50_ms"],
            "sequential_over_k_p50": seq["p50_ms"] / (k_views * one["p50_ms"])}

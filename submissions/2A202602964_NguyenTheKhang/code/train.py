"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Hoàn thiện từ starter/train.py. MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm
chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0

Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.

Các lựa chọn (ghi trong báo cáo):
  - Lịch LR cập nhật THEO BƯỚC: warmup tuyến tính (từ 1% LR) trong `warmup_epochs`, rồi cosine về
    min_lr_ratio * LR (mặc định 1e-3, tức "gần 0").
  - EMA (nếu bật) áp dụng cho cả tham số và buffer số thực (running mean/var của BN), có warmup
    decay_eff = min(decay, (1 + n) / (10 + n)). Khi có EMA, checkpoint được chọn và đánh giá bằng
    trọng số EMA; macro-F1 val của trọng số thường vẫn được log (cột raw_val_macro_f1) để so sánh I06.
  - Tái lập: seed cố định cho random/numpy/torch/CUDA và DataLoader; cudnn.benchmark = True để nhanh,
    nên kết quả lặp lại được tới mức sai khác nhỏ do phép toán không tất định của GPU.
  - Test: chỉ đánh giá khi cfg.save_test_predictions=True (Bước 4). Mỗi lần truy cập test ghi vào
    <out_dir>/test_access.log; lần thứ hai cho cùng exp_id/seed bị từ chối.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def find_repo_dir(start: Path = HERE) -> Path:
    """Thư mục chứa eval.py gốc (repo bài lab): đi ngược lên từ code/."""
    for p in [start, *start.parents]:
        if (p / "eval.py").exists() and (p / "RUBRIC.md").exists():
            return p
    env = os.environ.get("LAB_REPO_DIR")
    if env and (Path(env) / "eval.py").exists():
        return Path(env)
    raise FileNotFoundError("không tìm thấy eval.py của repo gốc; đặt biến môi trường LAB_REPO_DIR")


REPO_DIR = find_repo_dir()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn, dùng trong tên ảnh curves/<exp_id>_<desc>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    mix_prob: float = 1.0             # xác suất áp dụng mix cho mỗi batch
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    optimizer: str = "adamw"          # adamw | sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 1e-3
    clip_grad: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    cache_dir: str | None = None      # nếu có: cache ảnh đã giải mã (ImageCache)
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    # --- tiện ích ---
    resume: bool = True               # bỏ qua nếu đã xong; tiếp tục từ last.pt nếu bị ngắt
    keep_ckpt: bool = True            # giữ best.pt (cần cho Bước 3/4)
    bench_latency: bool = True        # đo độ trễ sơ bộ batch 1 (Bước 1)
    debug_limit: int | None = None    # CHỈ cho smoke test: giới hạn số ảnh mỗi tập. None ở mọi lần chạy thật.


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Cố định random, numpy, torch (CPU và CUDA). deterministic=True: cudnn tất định (chậm hơn)."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def env_info() -> dict:
    import timm
    import torch
    import torchvision
    return {
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "timm": timm.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD momentum 0.9 nesterov) với các nhóm tham số của model.param_groups."""
    import torch
    from model import param_groups
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    for g in groups:
        g["initial_lr"] = g["lr"]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer không hợp lệ: {cfg.optimizer}")


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 1e-3) -> float:
    """Hệ số LR tại bước `step`: warmup tuyến tính 0.01 -> 1, rồi cosine 1 -> min_ratio."""
    if warmup_steps > 0 and step < warmup_steps:
        return 0.01 + 0.99 * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0, cập nhật theo BƯỚC (LambdaLR, gọi step() sau mỗi batch)."""
    import torch
    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total, warm, cfg.min_lr_ratio))


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Giữ một bản sao riêng `self.module` để đánh giá. Tham số và buffer số thực (kể cả running
    mean/var của BN) đều được làm trơn; buffer số nguyên (num_batches_tracked) được chép thẳng.
    """

    def __init__(self, model, decay: float):
        import copy
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = 0

    def update(self, model) -> None:
        import torch
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        with torch.no_grad():
            src = model.state_dict()
            for k, v in self.module.state_dict().items():
                s = src[k].detach()
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(s.to(v.dtype), alpha=1 - d)
                else:
                    v.copy_(s)

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd["module"])
        self.updates = sd["updates"]


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, mix_rng: np.random.Generator | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc", "lr", "lr_steps"}.

    train_acc trên batch đã Mixup/CutMix so với y_a nên KHÔNG mang nghĩa bình thường (ghi chú trong
    biểu đồ); đánh giá bằng val.
    """
    import torch
    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)          # giữ BN của backbone đóng băng ở eval
    use_amp = cfg.amp and device.type == "cuda"
    tot_loss, tot_correct, n, lrs = 0.0, 0, 0, []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        mixed = cfg.mix and (mix_rng.random() < cfg.mix_prob)
        if mixed:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix, rng=mix_rng)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = mixed_loss(criterion, logits, targets) if mixed else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()} (không hữu hạn)")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        bs = y.size(0)
        tot_loss += loss.item() * bs
        tot_correct += (logits.argmax(1) == y).sum().item()
        n += bs
        lrs.append(optimizer.param_groups[0]["lr"])
    return {"train_loss": tot_loss / max(n, 1), "train_acc": tot_correct / max(n, 1),
            "lr": lrs[-1] if lrs else float("nan"), "lr_steps": lrs}


def evaluate(model, loader, criterion, device, amp: bool = True, channels_last: bool = False):
    """Chạy model ở eval(), KHÔNG gradient. Trả về (filenames, y_true[N], logits[N, 9], loss)."""
    import torch
    model.eval()
    names, ys, outs = [], [], []
    use_amp = amp and device.type == "cuda"
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            outs.append(logits.float().cpu())
            ys.append(torch.as_tensor(y))
            names.extend(f)
    logits = torch.cat(outs)
    y_true = torch.cat(ys)
    loss = float(criterion(logits, y_true).item()) if criterion is not None else float("nan")
    return names, y_true.numpy(), logits.numpy(), loss


def metrics_from_logits(y_true, logits) -> dict:
    """Chỉ số theo đúng định nghĩa của eval.compute_metrics (repo gốc)."""
    from eval import compute_metrics
    from inference import softmax_np
    probs = softmax_np(logits)
    return compute_metrics(np.asarray(y_true), probs.argmax(1), probs)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps: list[float] | None = None,
                note: str = "") -> None:
    """Vẽ loss train/val, macro-F1/top-1 val (và train acc), LR theo bước -> một ảnh png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    ncol = 3 if lr_steps else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train loss")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val loss (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1")
    ax[1].plot(ep, [h["val_top1"] for h in history], "s-", label="val top-1")
    ax[1].plot(ep, [h["train_acc"] for h in history], "^--", alpha=.6,
               label="train acc" + (" (mix: không chuẩn)" if "mix" in note else ""))
    if any("raw_val_macro_f1" in h for h in history):
        ax[1].plot(ep, [h.get("raw_val_macro_f1") for h in history], "x:", label="val macro-F1 (raw, không EMA)")
    best = max(history, key=lambda h: (h["val_macro_f1"], -h["epoch"]))
    ax[1].axvline(best["epoch"], color="gray", ls=":", lw=1)
    ax[1].annotate(f"best ep {best['epoch']}\nF1={best['val_macro_f1']:.4f}", (best["epoch"], best["val_macro_f1"]),
                   textcoords="offset points", xytext=(-60, -30), fontsize=8)
    ax[1].set(xlabel="epoch", ylabel="metric", title="Val metrics")
    ax[1].legend(fontsize=8)
    if lr_steps:
        ax[2].plot(np.arange(len(lr_steps)), lr_steps)
        ax[2].set(xlabel="step", ylabel="LR (nhóm backbone)", title="LR schedule")
        ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(title + (f"  [{note}]" if note else ""))
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _log_test_access(cfg: Config) -> None:
    log = Path(cfg.out_dir) / "test_access.log"
    key = f"{cfg.exp_id}_seed{cfg.seed}"
    if log.exists() and any(line.split("\t")[1] == key for line in log.read_text(encoding="utf-8").splitlines() if "\t" in line):
        raise RuntimeError(f"test của {key} đã được chạy trước đó (quy tắc: một lần mỗi seed). Xem {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{key}\n")


def build_data(cfg: Config, model, splits=None, need_test: bool = False):
    """Dựng train/val(/test) loader đúng tiền xử lý của trọng số đang dùng."""
    import dataset as D
    from model import data_config
    train_df, val_df, test_df = splits or D.load_split(cfg.labels_dir, cfg.fold)
    if cfg.debug_limit:
        train_df, val_df, test_df = (d.groupby("Label", group_keys=False).head(max(1, cfg.debug_limit // 9))
                                     for d in (train_df, val_df, test_df))
    dc = data_config(model)
    cache = None
    if cfg.cache_dir:
        all_names = list(train_df.Filename) + list(val_df.Filename) + list(test_df.Filename)
        cache = D.ImageCache.build(all_names, cfg.images_dir, cfg.cache_dir)
    t_train = D.build_transforms(True, cfg.img_size, cfg.aug, dc["mean"], dc["std"])
    t_eval = D.build_transforms(False, cfg.img_size, "basic", dc["mean"], dc["std"], crop_pct=0.875)
    mk = lambda df, t, tr: D.make_loader(df, cfg.images_dir, t, cfg.batch_size if tr else cfg.batch_size * 2,
                                         tr, cfg.sampler if tr else None, cfg.num_workers, cache, cfg.seed)
    loaders = {"train": mk(train_df, t_train, True), "val": mk(val_df, t_eval, False)}
    if need_test:
        loaders["test"] = mk(test_df, t_eval, False)
    return loaders, (train_df, val_df, test_df), dc


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict tóm tắt.

    1. set_seed; run_dir; config.json        2. load_split + check_split
    3. loader (test chỉ khi save_test_predictions)   4. model, criterion, optimizer, scheduler, scaler, EMA
    5. mỗi epoch: train -> evaluate(val) -> history; best theo MACRO-F1 VAL (hòa: epoch sớm hơn)
    6. nạp best, lưu val logits + val_pred.csv     7. (Bước 4) test đúng MỘT lần
    8. history.csv, curve.png, summary.json
    """
    import torch
    import dataset as D
    from eval import save_predictions
    from inference import softmax_np
    from losses import build_criterion, class_weights
    from model import build_model, count_gmacs, count_params, weight_tag

    rd = run_dir(cfg)
    if cfg.resume and (rd / "summary.json").exists():
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã xong, đọc lại summary.json")
        return json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    rd.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = D.load_split(cfg.labels_dir, cfg.fold)
    split_report = D.check_split(*splits, cfg.images_dir, verbose=False) if not cfg.debug_limit else {}

    model = build_model(cfg.backbone, True, D.NUM_CLASSES, cfg.drop_rate, cfg.init, cfg.drop_path_rate,
                        img_size=cfg.img_size)
    loaders, (train_df, val_df, test_df), dc = build_data(cfg, model, splits, cfg.save_test_predictions)
    model.to(device)
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)

    counts = np.bincount(train_df.Label.to_numpy(), minlength=D.NUM_CLASSES)
    kw = {"smoothing": cfg.label_smoothing or 0.1, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted":
        kw["weight"] = class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
    criterion = build_criterion(cfg.loss, **kw).to(device)
    val_criterion = torch.nn.CrossEntropyLoss()   # val loss luôn là CE thường để so sánh được giữa các loss
    optimizer = build_optimizer(model, cfg)
    steps = len(loaders["train"])
    scheduler = build_scheduler(optimizer, cfg, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    mix_rng = np.random.default_rng(cfg.seed + 12345)

    meta = {"config": dataclasses.asdict(cfg), "env": env_info(), "weight_tag": weight_tag(model),
            "data_config": dc, "train_counts": counts.tolist(), "steps_per_epoch": steps,
            "split_check": {k: split_report.get(k) for k in ("n", "overlap", "union", "missing_files")}}
    (rd / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    history, lr_steps, best_f1, best_epoch, start_epoch = [], [], -1.0, -1, 1
    last = rd / "last.pt"
    if cfg.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        history, lr_steps = ck["history"], ck["lr_steps"]
        best_f1, best_epoch, start_epoch = ck["best_f1"], ck["best_epoch"], ck["epoch"] + 1
        mix_rng = ck.get("mix_rng", mix_rng)
        print(f"[{cfg.exp_id} seed{cfg.seed}] tiếp tục từ epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler, cfg,
                             device, ema, mix_rng)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0
        eval_model = ema.module if ema else model
        _, yv, lv, vloss = evaluate(eval_model, loaders["val"], val_criterion, device, cfg.amp, cfg.channels_last)
        m = metrics_from_logits(yv, lv)
        row = {"epoch": epoch, "train_loss": tr["train_loss"], "train_acc": tr["train_acc"], "val_loss": vloss,
               "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_bal_acc": m["balanced_acc"],
               "val_ece": m["ece"], "lr": tr["lr"], "train_time_s": train_time, "epoch_time_s": time.time() - t0}
        if ema:
            _, yr, lr_, _ = evaluate(model, loaders["val"], None, device, cfg.amp, cfg.channels_last)
            row["raw_val_macro_f1"] = metrics_from_logits(yr, lr_)["macro_f1"]
        history.append(row)
        lr_steps += tr["lr_steps"]
        if m["macro_f1"] > best_f1:          # hòa -> giữ epoch sớm hơn
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save(eval_model.state_dict(), rd / "best.pt")
        print(f"[{cfg.exp_id} seed{cfg.seed}] ep {epoch:2d}/{cfg.epochs} loss {tr['train_loss']:.4f} "
              f"val_loss {vloss:.4f} val_F1 {m['macro_f1']:.4f} top1 {m['top1']:.4f} ({row['epoch_time_s']:.0f}s)"
              + (f" rawF1 {row['raw_val_macro_f1']:.4f}" if ema else ""))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema else None, "history": history, "lr_steps": lr_steps,
                    "best_f1": best_f1, "best_epoch": best_epoch, "epoch": epoch, "mix_rng": mix_rng}, last)

    # 6. nạp checkpoint tốt nhất, lưu dự đoán val
    best_model = ema.module if ema else model
    best_model.load_state_dict(torch.load(rd / "best.pt", map_location=device, weights_only=True))
    names_v, yv, lv, _ = evaluate(best_model, loaders["val"], val_criterion, device, cfg.amp, cfg.channels_last)
    np.save(rd / "val_logits.npy", lv)
    save_predictions(rd / "val_pred.csv", names_v, yv, softmax_np(lv))
    mv = metrics_from_logits(yv, lv)
    if cfg.save_test_predictions:
        save_predictions(pred_path(cfg, "val"), names_v, yv, softmax_np(lv))

    # 7. test: đúng một lần, chỉ ở Bước 4
    if cfg.save_test_predictions:
        _log_test_access(cfg)
        names_t, yt, lt, _ = evaluate(best_model, loaders["test"], None, device, cfg.amp, cfg.channels_last)
        np.save(rd / "test_logits.npy", lt)
        save_predictions(pred_path(cfg, "test"), names_t, yt, softmax_np(lt))

    # 8. tóm tắt
    import pandas as pd
    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    np.save(rd / "lr_steps.npy", np.asarray(lr_steps))
    title = f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}" + (f" | {cfg.desc}" if cfg.desc else "")
    note = ",".join(x for x in [cfg.mix and f"mix={cfg.mix}", cfg.ema_decay and "EMA"] if x)
    plot_curves(history, rd / "curve.png", title, lr_steps, note)

    lat = None
    if cfg.bench_latency:
        from benchmark import latency_report
        try:
            lat = latency_report(best_model, 1, cfg.img_size, "fp32", str(device), warmup=10, iters=50)
        except Exception as e:  # noqa: BLE001
            print("bỏ qua đo độ trễ:", e)
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "desc": cfg.desc,
        "weight_tag": meta["weight_tag"], "best_epoch": best_epoch, "epochs": cfg.epochs, "img_size": cfg.img_size,
        "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"], "val_bal_acc": mv["balanced_acc"],
        "val_ece": mv["ece"], "val_nll": mv["nll"],
        "val_f1_per_class": mv["f1"].tolist(), "val_recall_per_class": mv["recall"].tolist(),
        "train_time_per_epoch_s": float(np.mean([h["train_time_s"] for h in history])),
        "params_M": count_params(model), "gmacs": count_gmacs(model, cfg.img_size),
        "latency_b1_fp32": lat, "env": meta["env"], "config": meta["config"],
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if last.exists():
        last.unlink()
    if not cfg.keep_ckpt:
        (rd / "best.pt").unlink(missing_ok=True)
    del model, ema, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _coerce(value: str, hint):
    """Ép chuỗi về kiểu của field (hỗ trợ X | None)."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if args and type(None) in args:
        if value.lower() in ("none", "null", ""):
            return None
        hint = next(a for a in args if a is not type(None))
    if hint is bool:
        if value.lower() in ("1", "true", "yes", "y"):
            return True
        if value.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"giá trị bool không hợp lệ: {value!r}")
    if hint in (int, float, str):
        return hint(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict đã ép kiểu theo field của Config."""
    hints = typing.get_type_hints(Config)
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"cần dạng KEY=VALUE, nhận {pair!r}")
        k, v = pair.split("=", 1)
        if k not in hints:
            raise KeyError(f"Config không có field {k!r}. Có: {sorted(hints)}")
        out[k] = _coerce(v, hints[k])
    return out


def main(argv: list[str] | None = None) -> None:
    """`python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một thí nghiệm DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = Config(**parse_overrides(args.set))
    s = run(cfg)
    print(json.dumps({k: s[k] for k in ("exp_id", "seed", "best_epoch", "val_macro_f1", "val_top1")}, indent=2))


if __name__ == "__main__":
    main()

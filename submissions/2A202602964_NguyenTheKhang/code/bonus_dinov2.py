"""bonus_dinov2.py - Điểm thưởng: linear probe trên DINOv2 ViT-S/14 đóng băng (exp_id B08).

Đặc trưng CLS (+ trung bình patch token, như khuyến nghị của DINOv2) trích MỘT lần với tiền xử lý
đánh giá (không augmentation), rồi huấn luyện một lớp Linear 9 lớp. Chọn epoch theo macro-F1 val,
giống mọi thí nghiệm khác. Không đụng tới test.
Ghi runs/B08/seed0/{summary.json, history.csv, curve.png, val_logits.npy} cùng định dạng train.run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import dataset as D
from experiments import Paths
from train import env_info, metrics_from_logits, plot_curves, set_seed

DINO_NAME = "vit_small_patch14_dinov2.lvd142m"


def run_dinov2_probe(paths: Paths, epochs: int = 30, lr: float = 1e-3, seed: int = 0) -> dict:
    import timm
    import torch
    from benchmark import latency_report
    from eval import save_predictions
    from inference import softmax_np
    from model import count_gmacs, count_params

    rd = Path(paths.out_dir) / "B08" / f"seed{seed}"
    if (rd / "summary.json").exists():
        return json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    rd.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = timm.create_model(DINO_NAME, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    cfg = backbone.pretrained_cfg
    mean, std = tuple(cfg.get("mean")), tuple(cfg.get("std"))
    tr, va, _ = D.load_split(paths.labels_dir, 0)
    cache = D.ImageCache.build(list(tr.Filename) + list(va.Filename), paths.images_dir, paths.cache_dir) \
        if paths.cache_dir else None

    def feats(df):
        ld = D.make_loader(df, paths.images_dir, D.eval_transform(224, 0.875, mean, std), 128, False,
                           num_workers=paths.num_workers, cache=cache)
        out, ys = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            for x, y, _ in ld:
                tok = backbone.forward_features(x.to(device))          # (B, 1 + N, C)
                out.append(torch.cat([tok[:, 0], tok[:, 1:].mean(1)], 1).float().cpu())
                ys.append(y)
        return torch.cat(out), torch.cat(ys)

    t0 = time.time()
    xtr, ytr = feats(tr)
    xva, yva = feats(va)
    feat_time = time.time() - t0
    head = torch.nn.Linear(xtr.shape[1], D.NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(xtr) / 256))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch)
    g = torch.Generator().manual_seed(seed)
    history, lr_steps, best = [], [], (-1.0, None, -1)
    xtr_d, ytr_d, xva_d = xtr.to(device), ytr.to(device), xva.to(device)
    for ep in range(1, epochs + 1):
        t1 = time.time()
        head.train()
        perm = torch.randperm(len(xtr), generator=g).to(device)
        tot, correct = 0.0, 0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            lg = head(xtr_d[idx])
            loss = torch.nn.functional.cross_entropy(lg, ytr_d[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            lr_steps.append(opt.param_groups[0]["lr"])
            tot += loss.item() * len(idx)
            correct += (lg.argmax(1) == ytr_d[idx]).sum().item()
        head.eval()
        with torch.inference_mode():
            lv = head(xva_d).float().cpu()
        vloss = torch.nn.functional.cross_entropy(lv, yva).item()
        m = metrics_from_logits(yva.numpy(), lv.numpy())
        history.append({"epoch": ep, "train_loss": tot / len(xtr), "train_acc": correct / len(xtr), "val_loss": vloss,
                        "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_bal_acc": m["balanced_acc"],
                        "val_ece": m["ece"], "lr": lr_steps[-1], "train_time_s": time.time() - t1,
                        "epoch_time_s": time.time() - t1})
        if m["macro_f1"] > best[0]:
            best = (m["macro_f1"], {k: v.clone() for k, v in head.state_dict().items()}, ep)
    head.load_state_dict(best[1])
    with torch.inference_mode():
        lv = head(xva_d).float().cpu().numpy()
    np.save(rd / "val_logits.npy", lv)
    save_predictions(rd / "val_pred.csv", va.Filename.tolist(), yva.numpy(), softmax_np(lv))
    torch.save(head.state_dict(), rd / "head.pt")
    import pandas as pd
    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    plot_curves(history, rd / "curve.png", f"B08 | DINOv2 ViT-S/14 đóng băng + linear probe | seed {seed}", lr_steps)

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, x):
            tok = self.backbone.forward_features(x)
            return self.head(torch.cat([tok[:, 0], tok[:, 1:].mean(1)], 1))
    full = Probe().eval()
    lat = latency_report(full, 1, 224, "fp32", str(device), iters=50)
    m = metrics_from_logits(yva.numpy(), lv)
    summary = {"exp_id": "B08", "seed": seed, "backbone": DINO_NAME, "desc": "dinov2_probe",
               "weight_tag": f"{cfg.get('architecture')}.{cfg.get('tag')}", "best_epoch": best[2], "epochs": epochs,
               "img_size": 224, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_bal_acc": m["balanced_acc"],
               "val_ece": m["ece"], "val_f1_per_class": m["f1"].tolist(), "val_recall_per_class": m["recall"].tolist(),
               "train_time_per_epoch_s": float(np.mean([h["train_time_s"] for h in history])),
               "feature_extraction_s": feat_time, "params_M": count_params(full),
               "gmacs": count_gmacs(full, 224), "latency_b1_fp32": lat, "env": env_info(),
               "config": {"probe": "linear on [CLS, mean(patch)]", "epochs": epochs, "lr": lr, "wd": 1e-4,
                          "batch": 256, "frozen": True}}
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def error_gallery(paths: Paths, pred_csv: str, out_png: str, pairs=((0, 7), (7, 0)), n: int = 8) -> list[dict]:
    """Ảnh TEST bị đoán sai cho các cặp (thật, đoán) — chỉ để PHÂN TÍCH LỖI sau Bước 4, không để chọn gì."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image
    df = pd.read_csv(pred_csv)
    pcols = [f"p{i}" for i in range(D.NUM_CLASSES)]
    df["conf"] = df[pcols].max(1)
    rows = []
    fig, axes = plt.subplots(len(pairs), n, figsize=(2.1 * n, 2.6 * len(pairs)), squeeze=False)
    for r, (t, p) in enumerate(pairs):
        wrong = df[(df.y_true == t) & (df.y_pred == p)].sort_values("conf", ascending=False).head(n)
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(wrong):
                w = wrong.iloc[c]
                ax.imshow(Image.open(Path(paths.images_dir) / w.Filename).convert("RGB"))
                ax.set_title(f"thật {D.CLASS_NAMES[t][:9]}\nđoán {D.CLASS_NAMES[p][:9]} ({w.conf:.2f})", fontsize=7)
                rows.append({"Filename": w.Filename, "true": D.CLASS_NAMES[t], "pred": D.CLASS_NAMES[p],
                             "conf": float(w.conf)})
    fig.suptitle("Ảnh test bị đoán sai (độ tin cậy cao nhất trước)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return rows

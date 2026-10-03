"""step0_checks.py - Bước 0: EDA, kiểm tra chia dữ liệu, kiểm tra pipeline (GUIDE 1.2-1.3, RUBRIC A).

Kết quả: work/figures/*.png và work/step0.json (đưa vào báo cáo và sheet Sanity).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import dataset as D
from experiments import BACKBONES, Paths


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def run_eda(paths: Paths) -> dict:
    plt = _plt()
    fig_dir = paths.sub("figures")
    tr, va, te = D.load_split(paths.labels_dir, 0)
    rep = D.check_split(tr, va, te, paths.images_dir)

    # 1. Phân bố lớp theo tập + đối chiếu Table 1
    counts = pd.DataFrame(rep["per_class"])
    counts["total"] = counts.sum(1)
    counts["paper_table1"] = [D.PAPER_TABLE1[c] for c in counts.index]
    counts["diff_vs_paper"] = counts["total"] - counts["paper_table1"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(counts))
    for i, s in enumerate(("train", "val", "test")):
        ax.bar(x + (i - 1) * 0.27, counts[s], 0.27, label=f"{s} (n={rep['n'][s]})")
    ax.set_xticks(x, counts.index, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set(ylabel="số ảnh (thang log)", title="DeepWeeds fold 0: số ảnh mỗi lớp theo tập")
    for i, v in enumerate(counts["total"]):
        ax.text(i, counts.loc[counts.index[i], "train"] * 1.15, str(v), ha="center", fontsize=7)
    ax.legend()
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_class_distribution.png", dpi=130)
    plt.close(fig)

    # 2. Ảnh mẫu: 4 ảnh mỗi lớp (từ train)
    from PIL import Image
    fig, axes = plt.subplots(D.NUM_CLASSES, 4, figsize=(8, 18))
    for c in range(D.NUM_CLASSES):
        files = tr[tr.Label == c].sample(4, random_state=0).Filename.tolist()
        for j, f in enumerate(files):
            axes[c, j].imshow(Image.open(Path(paths.images_dir) / f).convert("RGB"))
            axes[c, j].axis("off")
        axes[c, 0].set_title(D.CLASS_NAMES[c], loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_samples.png", dpi=110)
    plt.close(fig)

    # 3. Thống kê ảnh (kích thước, kênh, mean/std trên 500 ảnh train)
    sizes, modes, px = {}, {}, []
    for f in tr.sample(500, random_state=0).Filename:
        with Image.open(Path(paths.images_dir) / f) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
            px.append(np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3) / 255.0)
    px = np.concatenate(px)
    stats = {"sizes": {f"{k[0]}x{k[1]}": v for k, v in sizes.items()}, "modes": modes,
             "rgb_mean": px.mean(0).round(4).tolist(), "rgb_std": px.std(0).round(4).tolist(),
             "imagenet_mean": D.IMAGENET_MEAN, "imagenet_std": D.IMAGENET_STD}
    imbalance = counts["total"].max() / counts["total"].min()
    out = {"split_check": {k: rep[k] for k in ("n", "ratio", "overlap", "union", "expected_total", "missing_files")},
           "counts": counts.reset_index(names="class").to_dict(orient="records"),
           "imbalance_ratio_max_min": float(imbalance), "negatives_share": float(counts.loc["Negatives", "total"] / rep["union"]),
           "image_stats": stats}
    print(f"Tỉ lệ lớp lớn nhất / nhỏ nhất = {imbalance:.2f}; Negatives chiếm {out['negatives_share']:.1%}")
    return out


def run_sanity(paths: Paths, backbone: str = "resnet50") -> dict:
    """Kiểm tra pipeline (slide trang 59): loss ban đầu ≈ ln 9, overfit 1 batch, ảnh sau augmentation."""
    import torch
    from losses import FocalLoss, mix_batch
    from model import build_model, data_config, set_train_mode
    from train import set_seed

    plt = _plt()
    fig_dir = paths.sub("figures")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    tr, va, _ = D.load_split(paths.labels_dir, 0)
    out = {"ln9": math.log(9)}

    # (2) loss ban đầu của head mới, trên 256 ảnh val, cho mọi backbone của Bước 1
    sub = va.groupby("Label").sample(n=29, random_state=0)
    init_losses = {}
    for _, name, _, _ in BACKBONES:
        set_seed(0)
        m = build_model(name, True).to(device).eval()
        dc = data_config(m)
        ld = D.make_loader(sub, paths.images_dir, D.eval_transform(224, 0.875, dc["mean"], dc["std"]), 64, False,
                           num_workers=paths.num_workers)
        tot, n = 0.0, 0
        with torch.inference_mode():
            for x, y, _ in ld:
                lg = m(x.to(device)).float()
                tot += torch.nn.functional.cross_entropy(lg, y.to(device), reduction="sum").item()
                n += len(y)
        init_losses[name] = tot / n
        print(f"loss ban đầu {name:32s} {tot / n:.4f} (ln 9 = {math.log(9):.4f})")
        del m
    out["initial_loss"] = init_losses

    # (3) overfit một batch nhỏ (18 ảnh, 2 ảnh/lớp), không augmentation
    set_seed(0)
    m = build_model(backbone, True).to(device)
    dc = data_config(m)
    tiny = tr.groupby("Label", group_keys=False).head(2)
    ds = D.DeepWeedsDataset(tiny, paths.images_dir, D.eval_transform(224, 0.875, dc["mean"], dc["std"]))
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    y = torch.tensor([ds[i][1] for i in range(len(ds))], device=device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    losses = []
    for step in range(80):
        set_train_mode(m)
        loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    m.eval()
    with torch.inference_mode():
        acc = (m(x).argmax(1) == y).float().mean().item()
    out["overfit"] = {"n_images": len(ds), "steps": 80, "loss_first": losses[0], "loss_last": losses[-1],
                      "train_acc_eval_mode": acc, "passed": losses[-1] < 0.05}
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(losses)
    ax.set(yscale="log", xlabel="step", ylabel="CE loss", title=f"Overfit 1 batch ({len(ds)} ảnh, {backbone})")
    ax.axhline(math.log(9), ls=":", c="gray", label="ln 9")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "sanity_overfit_one_batch.png", dpi=130)
    plt.close(fig)
    del m

    # (4) ảnh sau augmentation (đã giải chuẩn hoá) kèm nhãn, cho mỗi mức aug + CutMix/Mixup
    fig, axes = plt.subplots(4, 6, figsize=(13, 9.5))
    sample = tr.sample(6, random_state=1)
    for r, aug in enumerate(D.AUG_LEVELS):
        ds = D.DeepWeedsDataset(sample, paths.images_dir, D.build_transforms(True, 224, aug))
        for c in range(6):
            img, lab, f = ds[c]
            axes[r, c].imshow(D.denormalize(img).permute(1, 2, 0).numpy())
            axes[r, c].set_title(f"{aug}: {D.CLASS_NAMES[lab]}", fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "sanity_augmentations.png", dpi=110)
    plt.close(fig)

    ds = D.DeepWeedsDataset(sample, paths.images_dir, D.build_transforms(False, 224))
    xb = torch.stack([ds[i][0] for i in range(6)])
    yb = torch.tensor([ds[i][1] for i in range(6)])
    fig, axes = plt.subplots(2, 6, figsize=(13, 5))
    rng = np.random.default_rng(0)
    for r, mode in enumerate(("cutmix", "mixup")):
        xm, (ya, yb2, lam) = mix_batch(xb, yb, 1.0, mode, rng=rng)
        for c in range(6):
            axes[r, c].imshow(D.denormalize(xm[c]).permute(1, 2, 0).numpy())
            axes[r, c].set_title(f"{mode} λ={lam:.2f}\n{D.CLASS_NAMES[ya[c]]} / {D.CLASS_NAMES[yb2[c]]}", fontsize=7)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "sanity_mix.png", dpi=110)
    plt.close(fig)

    # (5) chế độ train/eval và BN khi đóng băng
    m = build_model(backbone, True, init="frozen")
    set_train_mode(m)
    bn = [mod for mod in m.modules() if isinstance(mod, torch.nn.BatchNorm2d)]
    out["frozen_bn_eval"] = {"n_bn": len(bn), "n_bn_training": sum(b.training for b in bn),
                             "n_trainable_params": sum(p.numel() for p in m.parameters() if p.requires_grad)}
    # focal gamma=0 == CE
    lg, yy = torch.randn(64, 9), torch.randint(0, 9, (64,))
    out["focal_gamma0_minus_ce"] = abs(FocalLoss(0.0)(lg, yy).item() - torch.nn.functional.cross_entropy(lg, yy).item())
    return out


def run_step0(paths: Paths) -> dict:
    res = {"eda": run_eda(paths), "sanity": run_sanity(paths, BACKBONES[0][1])}
    (Path(paths.work_dir) / "step0.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return res

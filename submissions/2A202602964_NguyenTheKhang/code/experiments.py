"""experiments.py - danh sách thí nghiệm (B, T, F) và các QUY TẮC CHỌN viết trước khi có kết quả.

Mọi quyết định (chọn backbone, chọn yếu tố đưa vào kết hợp, chọn phương pháp suy luận) đều là hàm
thuần của số liệu VAL, được cố định trong file này trước khi chạy, để không thể "nhìn rồi chọn" và
không bao giờ đụng tới test (README S2, S4).

Thiết kế Bước 2: "một yếu tố mỗi lần" so với T00 (nguyên tắc N1), sau đó kết hợp các yếu tố thắng.
Đây KHÔNG phải tham lam tuần tự: mọi ablation đều so với cùng một nền T00, nên thứ tự các trục không
ảnh hưởng kết quả từng ablation; chỉ bước kết hợp (T15/T16) mới gộp các yếu tố.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from train import Config, run, run_dir

# CHỈ cho kiểm tra end-to-end trên CPU (LAB_SMOKE=1): model tí hon của timm, 1 epoch, tập con nhỏ.
# Không bao giờ bật khi chạy thật; mọi số trong báo cáo đến từ lần chạy KHÔNG có biến này.
SMOKE = os.environ.get("LAB_SMOKE") == "1"
SMOKE_MAP = {"resnet50": "test_resnet", "resnext50_32x4d": "test_byobnet", "convnext_tiny": "test_convnext",
             "deit_small_patch16_224": "test_vit", "swin_tiny_patch4_window7_224": "test_vit2",
             "efficientnet_b0": "test_efficientnet", "mobilenetv3_large_100": "test_efficientnet_evos"}
SMOKE_OVERRIDES = dict(epochs=1, img_size=64, batch_size=16, debug_limit=180, num_workers=0)

# --------------------------------------------------------------------------- #
# Đường dẫn dùng chung
# --------------------------------------------------------------------------- #


@dataclass
class Paths:
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    cache_dir: str | None = "data/cache"
    work_dir: str = "work"            # bền (Google Drive): runs/, predictions/, curves/, figures/, ...
    num_workers: int = 2

    @property
    def out_dir(self) -> str:
        return str(Path(self.work_dir) / "runs")

    @property
    def pred_dir(self) -> str:
        return str(Path(self.work_dir) / "predictions")

    def sub(self, name: str) -> Path:
        p = Path(self.work_dir) / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def base(self) -> dict:
        d = dict(images_dir=self.images_dir, labels_dir=self.labels_dir, cache_dir=self.cache_dir,
                 out_dir=self.out_dir, pred_dir=self.pred_dir, num_workers=self.num_workers)
        if SMOKE:
            d.update(SMOKE_OVERRIDES)
        return d


def smoke_limit(df):
    """Chỉ khi LAB_SMOKE=1: cắt tập con giống train.build_data (debug_limit). Lần chạy thật: trả nguyên df."""
    if not SMOKE:
        return df
    return df.groupby("Label", group_keys=False).head(max(1, SMOKE_OVERRIDES["debug_limit"] // 9))



# --------------------------------------------------------------------------- #
# Bước 1: backbone (cùng công thức nền T00, cùng seed 0)
# --------------------------------------------------------------------------- #
BACKBONES = [
    # exp_id, tên timm, họ, mô tả ngắn
    ("B01", "resnet50", "ResNet", "resnet50"),
    ("B02", "resnext50_32x4d", "ResNeXt", "resnext50"),
    ("B03", "convnext_tiny", "ConvNeXt", "convnext_tiny"),
    ("B04", "deit_small_patch16_224", "Transformer (DeiT)", "deit_small"),
    ("B05", "swin_tiny_patch4_window7_224", "Transformer (Swin)", "swin_tiny"),
    ("B06", "efficientnet_b0", "Nhẹ", "efficientnet_b0"),
    ("B07", "mobilenetv3_large_100", "Nhẹ", "mobilenetv3_large"),
]
if SMOKE:
    BACKBONES = [(e, SMOKE_MAP[n], f, d) for e, n, f, d in BACKBONES]
FAMILY = {name: fam for _, name, fam, _ in BACKBONES}
# Ngưỡng nhiễu đặt TRƯỚC khi có kết quả cho Bước 1 (chưa có std đo được): 0,005 macro-F1.
# Lý do: val có 3.5k ảnh nhưng mỗi lớp cỏ chỉ ~200 ảnh, sai số lấy mẫu của F1 một lớp ~2 điểm %,
# trung bình 9 lớp còn ~0,5-0,7 điểm %.
BACKBONE_TIE = 0.005


def backbone_configs(paths: Paths, seed: int = 0) -> list[Config]:
    return [Config(exp_id=e, backbone=n, desc=d, seed=seed, **paths.base()) for e, n, _, d in BACKBONES]


def load_summary(paths: Paths, exp_id: str, seed: int = 0) -> dict | None:
    p = Path(paths.out_dir) / exp_id / f"seed{seed}" / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def select_backbone(summaries: list[dict], tie: float = BACKBONE_TIE) -> tuple[dict, str]:
    """Quy tắc: lấy macro-F1 val cao nhất; mọi backbone cách nó <= `tie` coi là ngang nhau
    (không phân biệt được với 1 seed), trong nhóm ngang nhau chọn độ trễ batch-1 p50 thấp nhất."""
    ok = [s for s in summaries if s]
    best = max(ok, key=lambda s: s["val_macro_f1"])
    ties = [s for s in ok if best["val_macro_f1"] - s["val_macro_f1"] <= tie]
    lat = lambda s: (s.get("latency_b1_fp32") or {}).get("p50_ms", float("inf"))
    pick = min(ties, key=lat)
    reason = (f"macro-F1 val cao nhất là {best['backbone']} ({best['val_macro_f1']:.4f}). "
              f"Nhóm không phân biệt được (cách <= {tie}): "
              + ", ".join(f"{s['backbone']} ({s['val_macro_f1']:.4f}, p50 {lat(s):.1f} ms)" for s in ties)
              + f". Chọn {pick['backbone']} vì độ trễ thấp nhất trong nhóm.")
    return pick, reason


# --------------------------------------------------------------------------- #
# Bước 2: công thức huấn luyện (mỗi dòng khác T00 đúng MỘT yếu tố)
# --------------------------------------------------------------------------- #
ABLATIONS = [
    # exp_id, trục, mô tả, khác T00 ở điểm nào
    ("T01", "A", "scratch", {"init": "scratch"}),
    ("T02", "A", "frozen", {"init": "frozen"}),
    ("T03", "B", "color", {"aug": "color"}),
    ("T04", "B", "trivialaug", {"aug": "trivial"}),
    ("T05", "B", "cutmix", {"mix": "cutmix", "mix_alpha": 1.0}),
    ("T06", "B", "mixup", {"mix": "mixup", "mix_alpha": 0.2}),
    ("T07", "C", "labelsmooth", {"loss": "ls", "label_smoothing": 0.1}),
    ("T08", "C", "focal", {"loss": "focal", "focal_gamma": 2.0}),
    ("T09", "C", "cbweighted", {"loss": "ce_weighted", "class_weight_beta": 0.999}),
    ("T10", "D", "balancedsampler", {"sampler": "balanced"}),
    ("T11", "E", "samelr", {"lr_head": 1e-4}),
    ("T12", "F", "ema", {"ema_decay": 0.998}),
    ("T13", "G", "res256", {"img_size": 256}),
    ("T14", "G", "20epochs", {"epochs": 20}),
]
AXIS_NAME = {"A": "Khởi tạo", "B": "Augmentation", "C": "Loss", "D": "Cân bằng mẫu",
             "E": "LR/optimizer", "F": "Chính quy hoá", "G": "Độ phân giải / thời gian"}
# Nhóm loại trừ nhau khi kết hợp: mỗi nhóm lấy tối đa một yếu tố.
#   aug_level: color | trivial;   mix: cutmix | mixup;   imbalance: focal | cb-weighted | sampler
#   (không chồng hai cách sửa mất cân bằng, sẽ hiệu chỉnh hai lần);  ls đứng riêng nhưng chung
#   trường `loss` với focal/cb nên cũng loại trừ với chúng.
EXCLUSIVE = {
    "init": ["T01", "T02"], "aug_level": ["T03", "T04"], "mix": ["T05", "T06"],
    "loss_or_sampler": ["T07", "T08", "T09", "T10"], "lr": ["T11"], "ema": ["T12"],
    "res": ["T13"], "epochs": ["T14"],
}
T00_SEEDS = (0, 1, 2)
# Chung kết dùng 3 seed MỚI, chưa từng tham gia bất kỳ lựa chọn nào: macro-F1 val của F01 vì vậy không
# bị thiên lệch "người thắng cuộc" (seed 0 của kết hợp đã được dùng để chọn công thức).
FINAL_SEEDS = (10, 11, 12)


def t00_config(paths: Paths, backbone: str, seed: int = 0) -> Config:
    return Config(exp_id="T00", backbone=backbone, desc="baseline", seed=seed, **paths.base())


def ablation_configs(paths: Paths, backbone: str, seed: int = 0) -> list[Config]:
    out = []
    for e, _, d, diff in ABLATIONS:
        out.append(dataclasses.replace(t00_config(paths, backbone, seed), exp_id=e, desc=d, **diff))
    return out


def noise_std(paths: Paths) -> tuple[float, float, list[float]]:
    """mean và std mẫu (ddof=1) macro-F1 val của T00 qua 3 seed = thước đo nhiễu."""
    f1 = [load_summary(paths, "T00", s)["val_macro_f1"] for s in T00_SEEDS]
    return float(np.mean(f1)), float(np.std(f1, ddof=1)), f1


def ablation_table(paths: Paths) -> list[dict]:
    """Δ macro-F1 val của từng ablation so với T00 (seed 0, cùng seed) và so với mean T00."""
    t00 = load_summary(paths, "T00", 0)
    mean, std, _ = noise_std(paths)
    rows = []
    for e, ax, d, diff in ABLATIONS:
        s = load_summary(paths, e, 0)
        if s is None:
            continue
        delta = s["val_macro_f1"] - t00["val_macro_f1"]
        rows.append({"exp_id": e, "axis": ax, "desc": d, "diff": diff, "val_macro_f1": s["val_macro_f1"],
                     "val_top1": s["val_top1"], "delta_vs_T00_seed0": delta,
                     "delta_vs_T00_mean": s["val_macro_f1"] - mean, "noise_std": std,
                     "verdict": verdict(s["val_macro_f1"] - mean, std)})
    return rows


def verdict(delta: float, std: float) -> str:
    """Ngôn ngữ thận trọng (GUIDE N4): chỉ nói tốt hơn/kém hơn khi |Δ| > std (và > 2·std là 'rõ')."""
    if abs(delta) <= std:
        return "không phân biệt được (|Δ| ≤ std)"
    word = "tốt hơn" if delta > 0 else "kém hơn"
    return f"{word} rõ (|Δ| > 2·std)" if abs(delta) > 2 * std else f"{word} (std < |Δ| ≤ 2·std, 1 seed)"


def _pick_winners(rows: list[dict], threshold: float) -> dict:
    by_id = {r["exp_id"]: r for r in rows}
    diff = {}
    chosen = []
    for group, ids in EXCLUSIVE.items():
        cands = [by_id[i] for i in ids if i in by_id and by_id[i]["delta_vs_T00_mean"] > threshold]
        if cands:
            w = max(cands, key=lambda r: r["delta_vs_T00_mean"])
            diff.update(w["diff"])
            chosen.append(w["exp_id"])
    return {"diff": diff, "from": chosen}


def combo_specs(paths: Paths) -> dict:
    """Hai kết hợp định trước:
    T15 = các yếu tố có Δ (so với mean T00) > std nhiễu  (thận trọng)
    T16 = các yếu tố có Δ > 0                             (lạc quan, kiểm tra cộng dồn/triệt tiêu)
    """
    rows = ablation_table(paths)
    _, std, _ = noise_std(paths)
    return {"T15": {**_pick_winners(rows, std), "desc": "combo_strict", "rule": f"Δ > std ({std:.4f})"},
            "T16": {**_pick_winners(rows, 0.0), "desc": "combo_all_pos", "rule": "Δ > 0"}}


def combo_configs(paths: Paths, backbone: str, seed: int = 0) -> list[Config]:
    out, seen = [], []
    for e, spec in combo_specs(paths).items():
        if not spec["diff"] or spec["diff"] in seen:   # trống hoặc trùng kết hợp trước: bỏ
            continue
        seen.append(spec["diff"])
        out.append(dataclasses.replace(t00_config(paths, backbone, seed), exp_id=e, desc=spec["desc"],
                                       **spec["diff"]))
    return out


def select_recipe(paths: Paths, backbone: str) -> tuple[Config, str]:
    """Công thức chung kết: kết hợp có macro-F1 val cao nhất, chỉ nhận nếu > mean T00 + std;
    nếu không kết hợp nào vượt nhiễu thì giữ T00 (ghi rõ trong báo cáo)."""
    mean, std, _ = noise_std(paths)
    best_cfg, best_f1, reason = t00_config(paths, backbone), mean, f"giữ T00 (mean 3 seed {mean:.4f})"
    for cfg in combo_configs(paths, backbone):
        s = load_summary(paths, cfg.exp_id, 0)
        if s and s["val_macro_f1"] > best_f1 and s["val_macro_f1"] - mean > std:
            best_cfg, best_f1 = cfg, s["val_macro_f1"]
            reason = (f"{cfg.exp_id} ({cfg.desc}) macro-F1 val {s['val_macro_f1']:.4f} > mean T00 {mean:.4f} "
                      f"+ std {std:.4f}")
    return best_cfg, reason


def final_configs(paths: Paths, recipe: Config) -> list[Config]:
    return [dataclasses.replace(recipe, exp_id="F01", desc="final", seed=s) for s in FINAL_SEEDS]


# --------------------------------------------------------------------------- #
# Chạy hàng loạt (bỏ qua lần chạy đã xong; một lần hỏng không làm dừng cả loạt)
# --------------------------------------------------------------------------- #
def run_many(cfgs: list[Config]) -> list[dict]:
    out = []
    for cfg in cfgs:
        try:
            out.append(run(cfg))
        except Exception as e:  # noqa: BLE001  ghi lại thất bại để báo cáo trung thực
            import traceback
            rd = run_dir(cfg)
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"!!! {cfg.exp_id} seed{cfg.seed} THẤT BẠI: {e}")
    return out


def save_decision(paths: Paths, name: str, payload: dict) -> None:
    """Ghi lại mọi quyết định (kèm thời điểm) vào work/decisions.json để truy vết trong báo cáo."""
    import time
    p = Path(paths.work_dir) / "decisions.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d[name] = {**payload, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_decisions(paths: Paths) -> dict:
    p = Path(paths.work_dir) / "decisions.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

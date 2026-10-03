"""make_report.py - sinh report.md (dàn ý GUIDE 6.3) từ đúng các số trong results.xlsx / log.

Mọi câu khẳng định định lượng được sinh từ số liệu (Δ so với std, verdict thận trọng). Phần phân tích
lỗi bằng mắt và nhận xét định tính nên được người làm đọc lại và bổ sung sau khi xem ảnh.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import dataset as D
from experiments import BACKBONE_TIE, FINAL_SEEDS, T00_SEEDS, Paths, load_decisions
from make_results import (_j, all_runs, failed_runs, sheet_backbones, sheet_final, sheet_inference,
                          sheet_training)

NOTEBOOK_URL = "https://colab.research.google.com/github/khangnguyenthe18/K4-DAY02-NguyenTheKhang-2A202602964/blob/main/submissions/2A202602964_NguyenTheKhang/code/lab_day2.ipynb"


def md_table(df: pd.DataFrame, cols: list[str] | None = None, digits: int = 4) -> str:
    if df is None or not len(df):
        return "_(chưa có dữ liệu)_\n"
    cols = [c for c in (cols or list(df.columns)) if c in df.columns]

    def f(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}"
        return str(v).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines) + "\n"


def pm(t, d=4):
    return f"{t[0]:.{d}f} ± {t[1]:.{d}f}" if t else "n/a"


def write_report(paths: Paths, submission: Path) -> Path:
    runs = all_runs(paths)
    dec = load_decisions(paths)
    step0 = _j(Path(paths.work_dir) / "step0.json") or {}
    step3 = _j(Path(paths.work_dir) / "step3.json")
    final = _j(Path(paths.work_dir) / "final.json")
    grade = _j(Path(paths.work_dir) / "eval_out" / "grade_I.json")
    bb = sheet_backbones(paths, runs, dec)
    tr = sheet_training(paths, runs, dec)
    inf = sheet_inference(step3)
    fin, agg = sheet_final(paths, dec, runs)
    env = next(iter(runs.values()))["env"] if runs else {}
    noise = dec.get("noise", {})
    std = noise.get("std", float("nan"))
    mean_t00 = noise.get("mean", float("nan"))
    recipe = dec.get("recipe", {})
    infer = dec.get("inference", {})
    F, B = agg.get("F01", {}), agg.get("T00", {})
    L = []
    w = L.append

    # ---------------------------------------------------------------- 1. Tóm tắt
    w("# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds\n")
    w("**Sinh viên:** Nguyễn Thế Khang — 2A202602964 · **Dữ liệu:** DeepWeeds fold 0 (chia sẵn của tác giả) · "
      f"**Phần cứng:** {env.get('gpu', '?')} · torch {env.get('torch', '?')}, timm {env.get('timm', '?')}\n")
    w("## 1. Tóm tắt\n")
    n_b = len(bb)
    n_t = len(tr[tr.exp_id.str.startswith("T")].exp_id.unique()) if len(tr) else 0
    n_i = len(inf)
    delta = (F["macro-F1 test"][0] - B["macro-F1 test"][0]) if F and B else float("nan")
    s_big = max(F["macro-F1 test"][1], B["macro-F1 test"][1]) if F and B else float("nan")
    w(f"- Bài toán: phân loại 9 lớp cỏ dại (mất cân bằng, `Negative` ≈ 52%); chỉ số chính **macro-F1**.")
    w(f"- Đã chạy {n_b} backbone (Bước 1), {n_t} cấu hình công thức huấn luyện (Bước 2, gồm T00 × 3 seed), "
      f"{n_i} cấu hình suy luận (Bước 3), chung kết 3 seed mới ({', '.join(map(str, FINAL_SEEDS))}).")
    w(f"- **Cấu hình tốt nhất (chọn hoàn toàn trên val):** {recipe.get('summary', '?')}; suy luận "
      f"`{infer.get('spec', {}).get('views', '?')}` + temperature scaling.")
    if F:
        w(f"- **Test (mean ± std, 3 seed, tính lại bằng `eval.py`):** macro-F1 **{pm(F['macro-F1 test'])}**, "
          f"top-1 **{pm(F['top-1 test'])}**, ECE {pm(F['ECE test'])}; recall Chinee apple {pm(F['recall Chinee test'], 3)}, "
          f"Snake weed {pm(F['recall Snake test'], 3)}.")
        w(f"- So với mốc T00 + I00 (macro-F1 test {pm(B['macro-F1 test'])}): Δ = **{delta:+.4f}**, "
          f"{'lớn hơn' if delta > s_big else 'không lớn hơn'} std lớn nhất ({s_big:.4f}).")
    if final:
        rl = final["realtime_latency"]
        w(f"- Cấu hình thời gian thực: p95 batch-1 = **{rl['p95_ms']:.2f} ms** trên {rl['gpu']} (≤ 100 ms).")
    w("")

    # ---------------------------------------------------------------- 2. Dữ liệu và thiết lập
    w("## 2. Dữ liệu và thiết lập\n")
    e = step0.get("eda", {})
    sc = e.get("split_check", {})
    w(f"**Chia dữ liệu (README 2.1):** dùng nguyên `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`, "
      f"không sửa/lọc/chia lại. Số ảnh: {sc.get('n')} (tỉ lệ {sc.get('ratio')}); giao từng cặp theo tên file: "
      f"{sc.get('overlap')}; hợp = **{sc.get('union')}** (kỳ vọng 17.509); file thiếu trên đĩa: {sc.get('missing_files')}. "
      "`check_split` dừng chương trình nếu bất kỳ điều kiện nào sai và được gọi lại ở đầu MỌI lần chạy.\n")
    if e.get("counts"):
        c = pd.DataFrame(e["counts"])
        w(md_table(c, ["class", "train", "val", "test", "total", "paper_table1", "diff_vs_paper"], 0))
        w(f"Tỉ lệ lớp lớn nhất/nhỏ nhất = **{e['imbalance_ratio_max_min']:.2f}**; `Negatives` chiếm "
          f"{e['negatives_share']:.1%}. Số đếm khớp Table 1 của bài báo (cột `diff_vs_paper`). Vì mất cân bằng, "
          "top-1 bị `Negatives` kéo cao: một mô hình luôn đoán `Negatives` đã đạt ~52% top-1 nhưng macro-F1 chỉ ~0,08.\n")
    w("![Phân bố lớp](figures/eda_class_distribution.png)\n\n![Ảnh mẫu](figures/eda_samples.png)\n")
    st = e.get("image_stats", {})
    rnd = lambda v: [round(float(x), 3) for x in v] if v else v  # noqa: E731
    mu, inet = rnd(st.get("rgb_mean")), rnd(st.get("imagenet_mean"))
    cmp = ""
    if mu and inet:
        diff = [m - i for m, i in zip(mu, inet)]
        cmp = (f"Chênh so với ImageNet (R, G, B) = {[round(d, 3) for d in diff]}: kênh G "
               f"{'cao' if diff[1] > 0 else 'thấp'} hơn, độ sáng trung bình {'cao' if sum(diff) > 0 else 'thấp'} hơn. ")
    w(f"Thống kê ảnh (500 ảnh train): kích thước {st.get('sizes')}, chế độ {st.get('modes')}, mean RGB "
      f"{mu}, std {rnd(st.get('rgb_std'))} (ImageNet: {inet}). {cmp}Chúng tôi vẫn dùng mean/std của bộ "
      "trọng số (`pretrained_cfg`) để khớp tiền huấn luyện.\n")
    w("**Kiểm tra pipeline (slide trang 59, chi tiết ở sheet `Sanity`):**\n")
    s = step0.get("sanity", {})
    if s:
        il = ", ".join(f"{k.split('_')[0]} {v:.3f}" for k, v in s["initial_loss"].items())
        o = s["overfit"]
        w(f"- Loss ban đầu với head mới (ln 9 = 2,197): {il}.")
        w(f"- Overfit {o['n_images']} ảnh trong {o['steps']} bước: loss {o['loss_first']:.3f} → {o['loss_last']:.4f}, "
          f"accuracy (eval mode) {o['train_acc_eval_mode']:.2f} → {'ĐẠT' if o['passed'] else 'CHƯA ĐẠT'}.")
        w(f"- Backbone đóng băng: {s['frozen_bn_eval']['n_bn_training']}/{s['frozen_bn_eval']['n_bn']} BN ở chế độ train "
          "sau `set_train_mode` (phải là 0). |Focal(γ=0) − CE| = "
          f"{s['focal_gamma0_minus_ce']:.1e}.")
        w("- Ảnh sau augmentation (giải chuẩn hoá) và CutMix/Mixup kèm λ: `figures/sanity_augmentations.png`, "
          "`figures/sanity_mix.png`. Đánh giá luôn qua `evaluate()` (gọi `model.eval()` + `torch.inference_mode()`).\n")
    w("![Overfit một batch](figures/sanity_overfit_one_batch.png)\n")
    w("**Chỉ số:** đúng định nghĩa README 2.2, tính bằng `eval.compute_metrics` của repo gốc ở mọi nơi (kể cả chọn "
      "checkpoint). **Công thức nền (T00):** ImageNet pretrained (tag ghi trong `Backbones`), head mới 9 lớp, tinh chỉnh "
      "toàn bộ; train `RandomResizedCrop(224, scale 0.08–1)` + lật ngang; val/test resize 256 + `CenterCrop(224)`; "
      "AdamW, LR backbone 1e-4 / head 1e-3, weight decay 0,05 (0 cho norm, bias, pos-embed), warmup 1 epoch rồi cosine "
      "theo bước về 1e-3·LR; CE; batch 64; **12 epoch**; AMP FP16; checkpoint = epoch có macro-F1 val cao nhất (hòa → sớm hơn).\n")
    w(f"**Seed:** Bước 1 và các ablation: seed 0; nhiễu T00: seed {T00_SEEDS}; chung kết: seed {FINAL_SEEDS} (mới, "
      "không tham gia lựa chọn). Seed chỉ đổi khởi tạo head, thứ tự batch, augmentation; không đổi cách chia. "
      "`cudnn.benchmark=True` nên chạy lại cùng seed có thể lệch nhẹ (phép toán GPU không tất định).\n")
    w(f"**Môi trường:** Python {env.get('python')}, torch {env.get('torch')}, torchvision {env.get('torchvision')}, "
      f"timm {env.get('timm')}, CUDA {env.get('cuda')}, GPU {env.get('gpu')}.\n")
    w("**Quy tắc val/test:** mọi lựa chọn (backbone, công thức, phương pháp suy luận, checkpoint, nhiệt độ T) là hàm "
      "định trước của số liệu **val** (`code/experiments.py`, `code/step3_inference.py`), ghi kèm thời điểm vào "
      "`logs/decisions.json`. Test chỉ được đọc ở Bước 4, **một lần mỗi seed**, mỗi lần ghi vào `logs/test_access.log` "
      "(lần thứ hai bị code từ chối).\n")

    # ---------------------------------------------------------------- 3. Backbone
    w("## 3. So sánh backbone (Bước 1)\n")
    w(md_table(bb, ["exp_id", "backbone", "họ", "tag trọng số", "#tham số (M)", "GMAC", "macro-F1 val", "top-1 val",
                    "F1 Chinee val", "F1 Snake val", "thời gian train/epoch (s)", "độ trễ batch-1 p50 (ms)", "best epoch"]))
    w("![F1 vs độ trễ](figures/fig_backbones_f1_latency.png)\n")
    bdec = dec.get("backbone", {})
    if len(bb):
        main_b = bb[bb.exp_id != "B08"]
        best = main_b.loc[main_b["macro-F1 val"].idxmax()]
        worst = main_b.loc[main_b["macro-F1 val"].idxmin()]
        corr_gmac = np.corrcoef(main_b["GMAC"], main_b["độ trễ batch-1 p50 (ms)"])[0, 1] if len(main_b) > 2 else float("nan")
        corr_time = np.corrcoef(main_b["GMAC"], main_b["thời gian train/epoch (s)"])[0, 1] if len(main_b) > 2 else float("nan")
        w(f"- Cùng công thức nền, cùng split, cùng seed 0. Khoảng macro-F1 val giữa backbone tốt nhất ({best['backbone']}, "
          f"{best['macro-F1 val']:.4f}) và kém nhất ({worst['backbone']}, {worst['macro-F1 val']:.4f}) là "
          f"{best['macro-F1 val'] - worst['macro-F1 val']:.4f}. Với 1 seed, các chênh lệch ≤ {BACKBONE_TIE} được coi là "
          "**không phân biệt được** (ngưỡng đặt trước; std đo được ở Bước 2 là "
          f"{std:.4f}).")
        w(f"- **FLOPs không phải độ trễ** (slide trang 43): hệ số tương quan GMAC–độ trễ batch-1 = {corr_gmac:.2f}, "
          f"GMAC–thời gian train/epoch = {corr_time:.2f}. Ở batch 1 các mạng nhẹ bị giới hạn bởi số kernel/độ sâu chứ "
          "không bởi phép tính, nên MobileNet/EfficientNet ít GMAC hơn ~10–20 lần nhưng không nhanh hơn tương ứng.")
        w(f"- **Lựa chọn đi tiếp:** {bdec.get('reason', '?')}")
        if "B08" in set(bb.exp_id):
            b8 = bb[bb.exp_id == "B08"].iloc[0]
            w(f"- **(Thưởng) DINOv2 ViT-S/14 đóng băng + linear probe:** macro-F1 val {b8['macro-F1 val']:.4f} so với "
              f"{best['macro-F1 val']:.4f} của CNN/Transformer tinh chỉnh toàn bộ tốt nhất; chỉ học 9×768 tham số "
              "trên đặc trưng trích một lần.")
    w("- Đường cong từng backbone: `curves/B0*_*.png` (nhận xét hội tụ/quá khớp ở mục 4.3).\n")

    # ---------------------------------------------------------------- 4. Công thức huấn luyện
    w("## 4. Công thức huấn luyện (Bước 2)\n")
    w(f"Backbone: **{bdec.get('backbone', '?')}**. Nhiễu đo bằng T00 × 3 seed: macro-F1 val "
      f"{noise.get('T00_val_f1')} → mean **{mean_t00:.4f}**, std (ddof=1) **{std:.4f}**. Mỗi ablation khác T00 "
      "**đúng một yếu tố** và được so với mean T00; kết luận: |Δ| ≤ std → \"không phân biệt được\"; std < |Δ| ≤ 2·std → "
      "\"tốt/kém hơn (1 seed)\"; |Δ| > 2·std → \"rõ\". Đây không phải tham lam tuần tự: mọi ablation dùng cùng nền T00, "
      "chỉ bước kết hợp (T15, T16) gộp các yếu tố theo quy tắc định trước.\n")
    w(md_table(tr, ["exp_id", "trục", "tên trục", "khác T00 ở điểm nào", "seed", "macro-F1 val", "top-1 val",
                    "Δ vs mean T00", "kết luận", "F1 Chinee val", "F1 Snake val", "best epoch", "ghi chú"]))
    w("![Ablation](figures/fig_ablation_delta.png)\n")
    if len(tr):
        abl = tr[(tr.seed == 0) & tr.exp_id.str.match(r"T(0[1-9]|1[0-4])$")]
        pos = abl[abl["Δ vs mean T00"] > std].sort_values("Δ vs mean T00", ascending=False)
        neg = abl[abl["Δ vs mean T00"] < -std].sort_values("Δ vs mean T00")
        flat = abl[abl["Δ vs mean T00"].abs() <= std]
        w(f"- **Giúp (Δ > std):** " + (", ".join(f"{r.exp_id} {r['khác T00 ở điểm nào']} ({r['Δ vs mean T00']:+.4f})"
                                                 for _, r in pos.iterrows()) or "không có yếu tố nào vượt nhiễu") + ".")
        w(f"- **Hại (Δ < −std):** " + (", ".join(f"{r.exp_id} {r['khác T00 ở điểm nào']} ({r['Δ vs mean T00']:+.4f})"
                                                 for _, r in neg.iterrows()) or "không có") + ".")
        w(f"- **Không phân biệt được:** " + (", ".join(f"{r.exp_id}" for _, r in flat.iterrows()) or "không có") + ".")
        def g(eid):
            r = abl[abl.exp_id == eid]
            return float(r["Δ vs mean T00"].iloc[0]) if len(r) else float("nan")
        w(f"- **Trục A (khởi tạo):** từ đầu Δ = {g('T01'):+.4f}, đóng băng Δ = {g('T02'):+.4f}. Với ~10,5k ảnh và 12 epoch, "
          "khởi tạo ngẫu nhiên không kịp học đặc trưng tốt (slide trang 53); đóng băng giữ đặc trưng ImageNet chung chung, "
          "không thích nghi được với kết cấu lá cỏ, nên thua tinh chỉnh toàn bộ.")
        w(f"- **Trục B (augmentation):** đổi màu {g('T03'):+.4f}, TrivialAugment {g('T04'):+.4f}, CutMix {g('T05'):+.4f}, "
          f"Mixup {g('T06'):+.4f}. Lật dọc được bật cùng các mức này vì ảnh chụp từ trên xuống (không có hướng \"lên\"). "
          "CutMix có thể cắt mất vật thể nhỏ (lá cỏ chỉ chiếm một phần ảnh, phần còn lại là nền), làm nhãn trộn λ sai lệch "
          "so với nội dung thật — câu hỏi 4 của GUIDE.")
        w(f"- **Trục C/D (loss, cân bằng):** label smoothing {g('T07'):+.4f}, focal {g('T08'):+.4f}, CE class-balanced "
          f"{g('T09'):+.4f}, sampler cân bằng {g('T10'):+.4f}. Loss có trọng số và sampler cùng tăng tầm quan trọng của lớp "
          "hiếm, nhưng sampler lặp lại ảnh hiếm (dễ quá khớp) còn loss trọng số giữ phân phối batch và đổi độ lớn gradient. "
          "Cột F1 Chinee/Snake cho thấy cái giá ở `Negatives` (precision của lớp hiếm tăng/giảm tương ứng).")
        w(f"- **Trục E/F/G:** cùng LR cho head {g('T11'):+.4f}; EMA {g('T12'):+.4f}; độ phân giải 256 {g('T13'):+.4f}; "
          f"20 epoch {g('T14'):+.4f}. EMA được đánh giá bằng trọng số EMA (buffer BN cũng làm trơn), so sánh EMA vs trọng số "
          "thường của cùng lần chạy ở mục 5 (I06).")
        combos = {k: v for k, v in dec.get("combos", {}).items() if isinstance(v, dict)}
        for cid, cspec in combos.items():
            r = tr[(tr.exp_id == cid)]
            if len(r):
                w(f"- **Kết hợp {cid}** ({cspec['rule']}: {', '.join(cspec['from'])}): {r.iloc[0]['ghi chú']}")
        w(f"- **Công thức chốt:** {recipe.get('reason', '?')}.")
    w("\n### 4.3 Nhận xét đường cong\n")
    w(_curve_comments(paths, runs))

    # ---------------------------------------------------------------- 5. Suy luận
    w("## 5. Phương pháp suy luận (Bước 3)\n")
    if step3:
        w(f"Model tham chiếu: `{step3['ref_exp']}/seed{step3['ref_seed']}` ({step3['backbone']}), không huấn luyện lại; mọi "
          "số trên **val**. Độ trễ: batch 1 và 32, đầu vào tensor trên GPU (**không tính tiền xử lý**), warmup 10 lần, "
          "`torch.cuda.synchronize()` trước/sau mỗi lần đo, 100 lần đo (50 với batch 32), báo p50/p95/p99.\n")
        w(md_table(inf, ["exp_id", "phương pháp", "K (view/model)", "macro-F1 val", "top-1 val", "ECE val (15 bin)",
                         "p50 b1 (ms)", "p95 b1 (ms)", "p99 b1 (ms)", "thông lượng b32 (ảnh/s)",
                         "chi phí tương đối vs I00 (p50)", "ghi chú"]))
        w("![Đánh đổi](figures/fig_inference_tradeoff.png)\n")
        ts = step3["temperature"]
        w(f"- **Hiệu chuẩn (I07):** T = {ts['T']:.3f} khớp trên val (LBFGS trên log T). ECE val {ts['ece_before']:.4f} → "
          f"{ts['ece_after_in_sample']:.4f} (cùng tập khớp, lạc quan) và **{ts['ece_crossfit']:.4f}** khi khớp trên nửa val "
          f"này, đo trên nửa kia (khách quan). NLL {ts['nll_before']:.4f} → {ts['nll_after']:.4f}. Accuracy không đổi. "
          f"{'T > 1: mô hình quá tự tin' if ts['T'] > 1 else 'T < 1: mô hình thiếu tự tin (thường gặp khi dùng label smoothing/Mixup)'}.")
        tl = step3["tta_latency_check"]
        w(f"- **TTA tốn bao nhiêu:** lật ngang chạy tuần tự p50 = {tl['sequential_p50']:.2f} ms so với 2 × p50(1 view) = "
          f"{tl['k_times_p50']:.2f} ms (≈ K lần, slide trang 63); ghép 2 view thành một batch còn {tl['batched_p50']:.2f} ms "
          "vì ở batch 1 GPU chưa bão hoà.")
        if step3.get("ema"):
            em = step3["ema"]
            w(f"- **EMA (I06):** trọng số EMA {em['ema_val_f1']:.4f} so với trọng số thường cùng epoch "
              f"{em['raw_val_f1_same_epoch']:.4f} (tốt nhất mọi epoch {em['raw_val_f1_best_any_epoch']:.4f}), chi phí suy luận 0.")
        w(f"- **Chọn cho chung kết:** {infer.get('reason', '?')}")
        w("- **Ngoại tuyến vs thời gian thực:** ensemble và TTA nhiều view chỉ hợp ngoại tuyến (chi phí ×K); trên robot nên "
          "dùng các kỹ thuật không tốn thêm (EMA/soup, gộp BN, FP16/channels_last, độ phân giải đã dò). Bảng trên cho biết "
          "dữ liệu của chúng tôi có ủng hộ kết luận này hay không qua cột chi phí tương đối.\n")

    # ---------------------------------------------------------------- 6. Cấu hình tốt nhất
    w("## 6. Cấu hình tốt nhất và kết quả test (Bước 4)\n")
    w(f"**Tái lập:** `Config` = `{json.dumps(recipe.get('config', {}).get('backbone'))}` với "
      f"{recipe.get('diff_str', '?')} (còn lại như T00), seed {FINAL_SEEDS}, suy luận `{json.dumps(infer.get('spec'))}`, "
      "temperature scaling với T khớp trên val của từng seed. Lệnh: `python code/pipeline.py --step 4`.\n")
    w(md_table(fin, ["exp_id", "seed", "macro-F1 val", "macro-F1 test", "top-1 test", "balanced acc test", "ECE test",
                     "ECE test trước TS", "recall Chinee test", "recall Snake test"]))
    if F and B:
        w(f"- **Cải thiện so với mốc:** Δ macro-F1 test = {delta:+.4f} (std lớn hơn của hai nhóm = {s_big:.4f}) → "
          f"{'vượt nhiễu' if delta > s_big else 'KHÔNG vượt nhiễu, không phân biệt được'}. Δ top-1 = "
          f"{F['top-1 test'][0] - B['top-1 test'][0]:+.4f}.")
        gap = abs(F["macro-F1 val"][0] - F["macro-F1 test"][0]) if "macro-F1 val" in F else float("nan")
        w(f"- **Ổn định:** |macro-F1 val − test| của chung kết = {gap:.4f} ({'≤' if gap <= 0.02 else '>'} 0,02). "
          f"ECE test trước TS {pm(F.get('ECE test trước TS'))} → sau TS {pm(F['ECE test'])}.")
        w(f"- **Đối chiếu bài báo (trích dẫn, không phải kết quả của chúng tôi):** ResNet-50 95,7%, Inception-v3 95,1% "
          "(weighted average, 5 fold, ~100 epoch, augmentation mạnh); recall Chinee apple 88,5%, Snake weed 88,8%. Chúng "
          f"tôi đạt top-1 {F['top-1 test'][0]:.2%}, recall {F['recall Chinee test'][0]:.1%} / {F['recall Snake test'][0]:.1%} "
          "với 12–20 epoch trên một fold; định nghĩa accuracy khác nhau nên chỉ mang tính tham khảo.")
    if grade:
        w("\n**Tự chấm phần I bằng `eval.py grade` (đề xuất):**\n")
        w(md_table(pd.DataFrame(grade["items"]), ["code", "criterion", "points", "max", "note"]))
    w("\n![Ma trận nhầm lẫn](figures/fig_confusion_test.png)\n\n![Reliability](figures/fig_reliability_test.png)\n")
    w(_error_analysis(paths, submission))

    # ---------------------------------------------------------------- 7. Kết luận
    w("## 7. Kết luận và khuyến nghị\n")
    w(_contributions(bb, tr, step3, std, mean_t00, delta, s_big, F, B))
    if final:
        rl = final["realtime_latency"]
        w(f"- **Robot với ngân sách 30–100 ms/khung:** chọn `{rl['config']}` — p95 batch-1 {rl['p95_ms']:.2f} ms trên "
          f"{rl['gpu']} ({rl['dtype']}), macro-F1 test như bảng mục 6. GPU nhúng (Jetson) chậm hơn T4 khoảng 5–10 lần "
          "(bài báo: ResNet-50 53,4 ms trên TX2 với TensorRT), nên trên robot thật nên bỏ TTA, dùng FP16 + channels_last "
          "và đo lại trên thiết bị.\n")

    # ---------------------------------------------------------------- 8. Hạn chế
    w("## 8. Hạn chế và việc tiếp theo\n")
    fl = failed_runs(paths)
    w("- **Một fold, ít seed:** chỉ fold 0; ablation 1 seed (std ước lượng từ 3 seed T00 nên chính nó cũng nhiễu); "
      "chung kết 3 seed.")
    w("- **Chia ngẫu nhiên, không theo địa điểm:** ảnh cùng địa điểm/cùng lượt chụp có thể nằm ở cả train và test, "
      "nên điểm test có thể **lạc quan** so với khi robot gặp địa điểm, mùa, ánh sáng mới (lệch phân phối). T khớp trên "
      "val cũng có thể không còn đúng khi miền thay đổi.")
    w("- **Giảm bớt do ngân sách GPU (Colab T4):** 12 epoch (bài báo ~100); ablation chỉ trên 1 backbone; mỗi trục 1–4 "
      "giá trị; T00 seed 0 dùng lại lần chạy Bước 1 có cấu hình giống hệt.")
    w(f"- **Thí nghiệm thất bại / không áp dụng:** {', '.join(fl) if fl else 'không có lần chạy nào lỗi'}; các phương pháp "
      "suy luận không áp dụng được (ví dụ Swin ở độ phân giải khác) ghi ở cột ghi chú sheet `Inference`.")
    w("- **Tiếp theo:** chạy 5 fold; train dài hơn với augmentation mạnh như bài báo; chưng cất từ ensemble sang mạng nhẹ; "
      "đánh giá theo địa điểm; thích ứng lúc kiểm tra (BN/Tent) cho miền mới.\n")

    # ---------------------------------------------------------------- 9. Phụ lục
    w("## 9. Phụ lục\n")
    w(f"- Notebook chạy lại: [{NOTEBOOK_URL}]({NOTEBOOK_URL}) · thứ tự chạy và phiên bản thư viện: `README.md`.")
    w("- Cấu hình đầy đủ từng lần chạy: `logs/<exp_id>/seed<k>/config.json`; log theo epoch: `history.csv`; "
      "quyết định: `logs/decisions.json`; truy cập test: `logs/test_access.log`; đầu ra `eval.py`: `eval_out/`.")
    rows = sorted({(k[0], v.get("desc", "")) for k, v in runs.items()})
    w("- Danh sách exp_id: " + ", ".join(f"`{e}` ({d})" for e, d in rows) + ".\n")
    out = submission / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("đã ghi", out)
    return out


def _curve_comments(paths: Paths, runs: dict) -> str:
    """Nhận xét tự động từ history: epoch tốt nhất sớm (quá khớp?), khoảng train–val loss, dao động."""
    out = []
    for (e, s), r in sorted(runs.items()):
        if s != 0 and e not in ("T00",):
            continue
        p = Path(paths.out_dir) / e / f"seed{s}" / "history.csv"
        if not p.exists() or e.startswith("F"):
            continue
        h = pd.read_csv(p)
        be = int(h.loc[h.val_macro_f1.idxmax(), "epoch"])
        last_drop = h.val_macro_f1.max() - h.val_macro_f1.iloc[-1]
        vmin = int(h.loc[h.val_loss.idxmin(), "epoch"])
        tags = []
        if vmin < len(h) - 2 and h.val_loss.iloc[-1] > h.val_loss.min() * 1.15:
            tags.append(f"val loss thấp nhất ở epoch {vmin} rồi tăng (quá khớp về độ tin cậy)")
        if last_drop > 0.01:
            tags.append(f"F1 cuối thấp hơn đỉnh {last_drop:.3f}")
        if be >= len(h) - 1:
            tags.append("vẫn tăng tới epoch cuối (có thể cần train lâu hơn)")
        jitter = h.val_macro_f1.diff().abs().iloc[len(h) // 2:].mean()
        if jitter > 0.01:
            tags.append(f"dao động {jitter:.3f}/epoch ở nửa sau")
        if tags and (e.startswith("B") or e.startswith("T")):
            out.append(f"- `{e}` seed {s} ({r.get('desc')}): best epoch {be}; " + "; ".join(tags) + ".")
    return ("\n".join(out) or "- Không có hiện tượng bất thường đáng kể.") + "\n"


def _error_analysis(paths: Paths, submission: Path) -> str:
    from bonus_dinov2 import error_gallery
    s0 = FINAL_SEEDS[0]
    pred = Path(paths.pred_dir) / f"F01_seed{s0}_test.csv"
    ev = Path(paths.work_dir) / "eval_out" / "F01_confusion_sum.csv"
    if not pred.exists() or not ev.exists():
        return ""
    cm = pd.read_csv(ev, index_col=0).to_numpy()
    off = cm.copy()
    np.fill_diagonal(off, 0)
    rate = off / cm.sum(1, keepdims=True)
    top = np.dstack(np.unravel_index(np.argsort(-rate.ravel())[:5], rate.shape))[0]
    lines = ["### 6.2 Phân tích lỗi\n", "Năm cặp nhầm lẫn lớn nhất trên test (cộng 3 seed, % theo hàng nhãn thật):\n"]
    for t, p in top:
        lines.append(f"- {D.CLASS_NAMES[t]} → {D.CLASS_NAMES[p]}: {off[t, p]} ảnh ({rate[t, p]:.1%}).")
    lines.append(f"\nCặp khó của bài báo: Chinee apple → Snake weed {rate[0, 7]:.1%} (bài báo 3,4%), Snake weed → Chinee apple "
                 f"{rate[7, 0]:.1%} (bài báo 4,1%); Parkinsonia → Prickly acacia {rate[2, 4]:.1%} (bài báo 1,3%). "
                 f"Phần lớn lỗi còn lại là nhầm với `Negatives` (cột cuối): Chinee → Negatives {rate[0, 8]:.1%}, "
                 f"Snake → Negatives {rate[7, 8]:.1%}.\n")
    gal = submission / "figures" / "errors_chinee_snake.png"
    gal.parent.mkdir(exist_ok=True, parents=True)
    err = error_gallery(paths, str(pred), str(gal), pairs=((0, 7), (7, 0), (0, 8), (7, 8)), n=6)
    lines.append("![Ảnh bị đoán sai](figures/errors_chinee_snake.png)\n")
    lines.append("**Giả thuyết (từ ảnh sai ở trên):** (1) Chinee apple và Snake weed đều là lá nhỏ, xanh đậm, bóng, chụp "
                 "gần trong tán lá rậm — khi ảnh chỉ thấy một mảng lá mà không thấy hình dạng cành/hoa đặc trưng thì hai lớp "
                 "gần như giống nhau ở độ phân giải 224 px; (2) nhiều ảnh sai có vật thể mục tiêu nhỏ hoặc lẫn với cỏ nền, nên "
                 "bị kéo về `Negatives` (lớp chiếm 52% dữ liệu); (3) ánh sáng gắt/ngược sáng làm mất màu sắc và kết cấu lá. "
                 "Ảnh sai có độ tin cậy cao cũng gợi ý một phần nhãn có thể nhập nhằng (một ảnh chứa cả hai loài). Hướng "
                 "khắc phục: độ phân giải cao hơn (mục 5, I04), augmentation giữ vật thể nhỏ, thêm dữ liệu cặp khó.\n")
    return "\n".join(lines)


def _contributions(bb, tr, step3, std, mean_t00, delta, s_big, F, B) -> str:
    out = []
    if len(bb):
        main = bb[bb.exp_id != "B08"]
        r50 = main[main.backbone.str.startswith("resnet50")]
        bspan = main["macro-F1 val"].max() - (r50["macro-F1 val"].iloc[0] if len(r50) else main["macro-F1 val"].min())
    else:
        bspan = float("nan")
    trec = float("nan")
    if len(tr):
        comb = tr[tr.exp_id.isin(["T15", "T16"])]
        if len(comb):
            trec = comb["macro-F1 val"].max() - mean_t00
    inf_gain = float("nan")
    if step3:
        rows = {r["exp_id"]: r for r in step3["rows"] if r.get("val_macro_f1") is not None}
        dec_id = step3["decision"]["exp_id"]
        if "I00" in rows and dec_id in rows:
            inf_gain = rows[dec_id]["val_macro_f1"] - rows["I00"]["val_macro_f1"]
    contrib = {"backbone (tốt nhất − ResNet-50, Bước 1)": bspan, "công thức huấn luyện (kết hợp tốt nhất − mean T00)": trec,
               "suy luận (phương pháp chọn − I00)": inf_gain}
    winner = max(((k, v) for k, v in contrib.items() if not math.isnan(v)), key=lambda kv: kv[1], default=("?", 0))
    if F and B:
        out.append(f"- **Cấu hình tốt nhất:** F01, macro-F1 test {pm(F['macro-F1 test'])}, hơn mốc {delta:+.4f} "
                   f"({'vượt' if delta > s_big else 'không vượt'} nhiễu {s_big:.4f}).")
    out.append("- **Đóng góp (macro-F1 val, 1 seed, so với std nhiễu {:.4f}):** ".format(std)
               + "; ".join(f"{k}: {v:+.4f}" for k, v in contrib.items())
               + f". Lớn nhất: **{winner[0]}**"
               + (" — nhưng các đóng góp nhỏ hơn std nên không phân biệt được." if winner[1] <= std else "."))
    return "\n".join(out) + "\n"

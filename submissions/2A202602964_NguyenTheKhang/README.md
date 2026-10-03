# Lab Day 2 — DeepWeeds · Nguyễn Thế Khang · 2A202602964

Bài nộp cho Lab Day 2 (backbone, công thức huấn luyện, suy luận trên DeepWeeds, fold 0).

| Sản phẩm | Vị trí |
|---|---|
| Notebook chạy lại (Colab, GPU T4) | [code/lab_day2.ipynb](code/lab_day2.ipynb) · [Mở trên Colab](https://colab.research.google.com/github/khangnguyenthe18/K4-DAY02-NguyenTheKhang-2A202602964/blob/main/submissions/2A202602964_NguyenTheKhang/code/lab_day2.ipynb) |
| Bảng so sánh | `results.xlsx` (Summary, Backbones, Training, Inference, Final, PerClass, Latency, Sanity) |
| Báo cáo | `report.md` |
| Biểu đồ training (mỗi `exp_id` một ảnh) | `curves/<exp_id>_<mota>.png` |
| Dự đoán test/val của chung kết và mốc | `predictions/` (`F01_seed{10,11,12}_{test,val}.csv`, `F01uncal_seed*_test.csv`, `T00_seed{0,1,2}_{test,val}.csv`) |
| Đầu ra `eval.py` gốc | `eval_out/` (`score_*.md`, `grade.md`, `*_summary.json`) |
| Log truy vết | `logs/<exp_id>/seed<k>/{config.json, summary.json, history.csv}`, `logs/decisions.json`, `logs/test_access.log` |
| Code | `code/` |

## Cách chạy lại

**Colab (khuyến nghị):** mở notebook, chọn *Runtime → T4 GPU*, *Run all*. Notebook clone repo này, tải DeepWeeds từ
Zenodo (kiểm tra MD5 `b7b30f96d466fba86016aa5a26606e0f`) và CSV fold 0 từ GitHub của tác giả, rồi chạy
`code/pipeline.py` từng bước. Kết quả ghi ra Google Drive (`MyDrive/deepweeds_lab_work`), nên nếu Colab ngắt chỉ cần
*Run all* lại: lần chạy đã xong được bỏ qua, lần đang dở tiếp tục từ epoch cuối (`last.pt`).

**Dòng lệnh (máy có GPU):**

```bash
pip install torch torchvision timm fvcore openpyxl matplotlib pandas scikit-learn
cd submissions/2A202602964_NguyenTheKhang/code
python -m unittest discover -s tests          # kiểm tra tự viết (CPU, ~10 s)
python pipeline.py --step all --images <thư mục .jpg> --labels <thư mục labels> --cache data/cache --work work
# hoặc từng bước: --step 0 | 1 | 2 | 3 | 4 | 5
python train.py --set exp_id=B01 backbone=resnet50 seed=0 images_dir=... labels_dir=...   # một thí nghiệm lẻ
```

**Thứ tự và thời lượng ước tính trên T4** (ước lượng trước khi chạy, thời gian đo thật nằm ở cột
`thời gian train/epoch (s)` của `results.xlsx`):

| Bước | Nội dung | Số lần huấn luyện |
|---|---|---|
| 0 | EDA, kiểm tra chia dữ liệu, loss ban đầu, overfit 1 batch, ảnh augmentation | 0 |
| 1 | 7 backbone × 12 epoch (seed 0) + DINOv2 linear probe | 7 (+ probe) |
| 2 | T00 seed 1, 2 (seed 0 dùng lại B-run giống hệt) + 14 ablation + tối đa 2 kết hợp | 18 |
| 3 | 20+ cấu hình suy luận trên val + đo độ trễ | 0 |
| 4 | Chung kết 3 seed mới (10, 11, 12), test một lần mỗi seed, `eval.py score/grade` | 3 |
| 5 | `results.xlsx`, `curves/`, `figures/`, `report.md` | 0 |

## Seed và tái lập

- Bước 1 và ablation Bước 2: seed 0. Nhiễu T00: seed 0, 1, 2. Chung kết F01: seed **10, 11, 12** (mới, không tham gia
  bất kỳ lựa chọn nào). Mốc T00 + I00: seed 0, 1, 2.
- `set_seed` cố định `random`, `numpy`, `torch` (CPU + CUDA), `Generator` và `worker_init_fn` của DataLoader. Seed không
  đổi cách chia (dùng nguyên CSV fold 0). `cudnn.benchmark=True` để nhanh, nên chạy lại có thể lệch nhẹ do GPU.
- Phiên bản thư viện thực tế của từng lần chạy được ghi trong `logs/<exp_id>/seed<k>/config.json` (khoá `env`) và ở
  mục 2 của `report.md`. Tag trọng số `timm` ghi trong sheet `Backbones`.

## Quy tắc val/test được code bảo đảm

- `dataset.check_split` chạy ở đầu mọi lần huấn luyện: giao từng cặp tập rỗng, hợp = 17.509, mọi file tồn tại, tỉ lệ
  60/20/20 ± 1 điểm %.
- Chọn backbone, công thức, phương pháp suy luận, checkpoint, nhiệt độ T: hàm định trước của số liệu **val**
  (`experiments.py`, `step3_inference.py`), quyết định ghi vào `logs/decisions.json` kèm thời điểm.
- Test chỉ được đọc trong `final.py` (Bước 4), một lần mỗi seed; mỗi lần ghi vào `logs/test_access.log`, lần thứ
  hai bị từ chối (`train._log_test_access`).
- `eval.py` gốc không sửa; file dự đoán ghi bằng `eval.save_predictions`.

## Cấu trúc `code/`

| File | Nội dung |
|---|---|
| `dataset.py` | đọc/kiểm tra split, transform theo mức aug, cache ảnh đã giải mã, DataLoader có seed, sampler cân bằng |
| `model.py` | backbone `timm`, đóng băng (BN ở eval), 4 nhóm tham số (không weight decay cho norm/bias/pos-embed), đếm params/GMAC |
| `losses.py` | label smoothing, focal, trọng số class-balanced, Mixup/CutMix (λ theo diện tích thật) |
| `train.py` | `Config` + một hàm `run(cfg)`: AMP, warmup + cosine theo bước, EMA, chọn checkpoint theo macro-F1 val, tiếp tục khi ngắt, vẽ đường cong |
| `inference.py` | TTA (lật, 5/10-crop, đa tỉ lệ), gộp xác suất/logit, ensemble, temperature scaling, gộp BN, model soup |
| `benchmark.py` | đo độ trễ: warmup, `cuda.synchronize`, ≥ 50 lần, p50/p95/p99 |
| `experiments.py` | danh sách thí nghiệm B/T/F và quy tắc chọn trên val |
| `step0_checks.py`, `step3_inference.py`, `final.py` | Bước 0, 3, 4 |
| `bonus_dinov2.py` | điểm thưởng: DINOv2 đóng băng + linear probe; ảnh lỗi để phân tích |
| `make_results.py`, `make_report.py` | tạo `results.xlsx`, `curves/`, `figures/`, `report.md` từ log |
| `pipeline.py`, `lab_day2.ipynb` | chạy toàn bộ theo thứ tự |
| `tests/test_code.py` | 31 kiểm tra: focal γ=0 ≡ CE, CutMix λ, gộp BN chính xác, TS, lịch LR, EMA, rò rỉ split, ... |

Không commit dataset và checkpoint (`.gitignore`).

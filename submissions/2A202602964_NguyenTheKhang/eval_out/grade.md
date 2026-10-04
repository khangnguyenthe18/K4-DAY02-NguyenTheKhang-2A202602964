## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 97.12% (mean 3 seed) |
| I2 | Macro-F1 cải thiện so với mốc | 5 | 5 | final 0.9608, mốc 0.9236, Δ=+0.0372, s=0.0096 |
| I3 | Recall hai lớp khó | 4 | 4 | Chinee Apple 91.6% (mốc 88.5%), Snake Weed 92.6% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.2749, sau 0.2026 |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9600, test 0.9608, chênh 0.0008 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 24.5 ms (ngân sách 100 ms), đo đúng cách |

**Tổng các ý đã chấm: 20 / 20** (phần I tối đa 20).

Ngưỡng điểm là TẠM THỜI (xem khối hằng số đầu file eval.py và RUBRIC.md mục I).

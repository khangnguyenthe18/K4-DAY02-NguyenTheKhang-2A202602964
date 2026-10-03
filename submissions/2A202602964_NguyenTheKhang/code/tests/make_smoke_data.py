"""make_smoke_data.py - dữ liệu GIẢ để kiểm tra end-to-end pipeline trên CPU (không dùng cho kết quả).

Tải các CSV chia fold THẬT của tác giả (để check_split chạy đúng trên 17.509 tên file), rồi tạo ảnh
256x256 tổng hợp có màu phụ thuộc lớp (để mô hình tí hon học được chút tín hiệu).
    python tests/make_smoke_data.py --out ../../smoke_data
    LAB_SMOKE=1 python pipeline.py --step all --images ../../smoke_data/images --labels ../../smoke_data/labels \
        --cache ../../smoke_data/cache --work ../../smoke_work --submission ../../smoke_submission --no-dinov2 --workers 0
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BASE = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smoke_data")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    for name in ("labels", "train_subset0", "val_subset0", "test_subset0"):
        p = out / "labels" / f"{name}.csv"
        if not p.exists():
            urllib.request.urlretrieve(f"{BASE}/{name}.csv", p)
    df = pd.read_csv(out / "labels" / "labels.csv")
    rng = np.random.default_rng(0)
    palette = rng.integers(40, 215, size=(9, 3))
    for f, lab in zip(df.Filename, df.Label):
        p = out / "images" / f
        if p.exists():
            continue
        img = rng.normal(palette[lab], 45, size=(32, 32, 3)).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).resize((256, 256), Image.NEAREST).save(p, quality=70)
    print(len(df), "ảnh giả tại", out / "images")

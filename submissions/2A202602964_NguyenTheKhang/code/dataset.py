"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Hoàn thiện từ starter/dataset.py. Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Bổ sung (được phép theo README 2.4):
    ImageCache      : giải mã toàn bộ ảnh JPEG một lần vào file .npy uint8 (memmap). Colab chỉ có
                      2 nhân CPU, giải mã JPEG mỗi epoch sẽ làm GPU chờ; cache này không đổi nội dung
                      ảnh (giải mã PIL giống hệt đường không cache), chỉ đổi chỗ đọc.
    eval_transform  : transform đánh giá có tham số crop_pct (dùng cho dò độ phân giải I04).

Lựa chọn tiền xử lý (ghi vào báo cáo):
    - Train "basic": RandomResizedCrop(img_size, scale=(0.08, 1)) + lật ngang.
    - Val/test: Resize(round(img_size / crop_pct)) + CenterCrop(img_size), crop_pct = 0.875.
      Với img_size = 224 và ảnh gốc 256x256, đây đúng là "resize 256 rồi CenterCrop 224".
    - Lật dọc: ảnh chụp từ trên xuống mặt đất nên về mặt ngữ nghĩa lật dọc hợp lệ; nhưng để giữ
      đúng công thức nền của GUIDE (chỉ lật ngang), lật dọc chỉ nằm trong mức "color" trở lên.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509
# Table 1 của bài báo (Olsen et al. 2019), dùng để đối chiếu ở bước EDA.
PAPER_TABLE1 = {
    "Chinee Apple": 1125, "Lantana": 1064, "Parkinsonia": 1031, "Parthenium": 1022,
    "Prickly Acacia": 1062, "Rubber Vine": 1009, "Siam Weed": 1074, "Snake Weed": 1016,
    "Negatives": 9106,
}
AUG_LEVELS = ("basic", "color", "trivial", "randaug")


# --------------------------------------------------------------------------- #
# Chia dữ liệu
# --------------------------------------------------------------------------- #
def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame, KHÔNG sửa, lọc hay chia lại.
    """
    labels_dir = Path(labels_dir)
    out = []
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_subset{fold}.csv"
        if not path.exists():
            raise FileNotFoundError(f"không thấy {path}; hãy tải từ github.com/AlexOlsen/DeepWeeds/labels")
        df = pd.read_csv(path)
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
        df["Label"] = df["Label"].astype(int)
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, expected_total: int = TOTAL_IMAGES, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir`
    Lỗi ở bất kỳ ý nào -> AssertionError (dừng ngay).
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    sets = {k: set(v["Filename"]) for k, v in splits.items()}
    for k, df in splits.items():
        assert not df["Filename"].duplicated().any(), f"{k}: có Filename trùng trong cùng tập"
        assert df["Label"].between(0, NUM_CLASSES - 1).all(), f"{k}: Label ngoài 0..8"

    n = {k: int(len(v)) for k, v in splits.items()}
    total = sum(n.values())
    ratio = {k: n[k] / total for k in n}
    per_class = {k: {CLASS_NAMES[c]: int((v["Label"] == c).sum()) for c in range(NUM_CLASSES)}
                 for k, v in splits.items()}
    overlap = {
        "train&val": len(sets["train"] & sets["val"]),
        "train&test": len(sets["train"] & sets["test"]),
        "val&test": len(sets["val"] & sets["test"]),
    }
    union = len(sets["train"] | sets["val"] | sets["test"])

    images_dir = Path(images_dir)
    on_disk = set(os.listdir(images_dir)) if images_dir.exists() else set()
    missing_files = sorted((sets["train"] | sets["val"] | sets["test"]) - on_disk)

    report = {
        "n": n, "ratio": {k: round(v, 4) for k, v in ratio.items()}, "per_class": per_class,
        "overlap": overlap, "union": union, "expected_total": expected_total,
        "missing_files": len(missing_files), "missing_examples": missing_files[:5],
        "per_class_total": {c: sum(per_class[s][c] for s in splits) for c in CLASS_NAMES},
    }
    if verbose:
        print(f"Số ảnh: {n}  (tỉ lệ {report['ratio']}), hợp = {union}")
        print("Giao từng cặp:", overlap, "| file thiếu trên đĩa:", len(missing_files))
        print(pd.DataFrame(per_class).assign(total=lambda d: d.sum(1)).to_string())

    assert all(v == 0 for v in overlap.values()), f"giao giữa các tập khác rỗng: {overlap}"
    assert union == expected_total, f"hợp ba tập = {union}, kỳ vọng {expected_total}"
    assert not missing_files, f"{len(missing_files)} file trong CSV không có trong {images_dir}: {missing_files[:5]}"
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert abs(ratio[k] - target) <= 0.01, f"tỉ lệ {k} = {ratio[k]:.3f} lệch quá 1 điểm % khỏi {target}"
    return report


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def eval_transform(img_size: int = 224, crop_pct: float = 0.875, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Transform đánh giá (không ngẫu nhiên): Resize(img_size / crop_pct) + CenterCrop(img_size)."""
    import torch
    from torchvision.transforms import v2

    resize = int(round(img_size / crop_pct))
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(resize, antialias=True),
        v2.CenterCrop(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_pct: float = 0.875):
    """Tạo transform theo `train` và mức `aug` (trục B của GUIDE.md mục 3).

    basic   : RandomResizedCrop + lật ngang                                   (công thức nền)
    color   : basic + lật dọc + ColorJitter(0.3, 0.3, 0.3, 0.05)
    trivial : basic + lật dọc + TrivialAugmentWide
    randaug : basic + lật dọc + RandAugment(num_ops=2, magnitude=9)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.
    Đánh giá: KHÔNG augmentation ngẫu nhiên, xem eval_transform.
    """
    import torch
    from torchvision.transforms import v2

    if not train:
        return eval_transform(img_size, crop_pct, mean, std)
    if aug not in AUG_LEVELS:
        raise ValueError(f"aug phải thuộc {AUG_LEVELS}, nhận {aug!r}")
    ops = [v2.ToImage(), v2.RandomResizedCrop(img_size, scale=(0.08, 1.0), antialias=True),
           v2.RandomHorizontalFlip()]
    if aug != "basic":
        ops.append(v2.RandomVerticalFlip())
    if aug == "color":
        ops.append(v2.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(v2.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(v2.RandAugment(num_ops=2, magnitude=9))
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(ops)


# --------------------------------------------------------------------------- #
# Cache ảnh đã giải mã
# --------------------------------------------------------------------------- #
class ImageCache:
    """Mảng uint8 (N, H, W, 3) trên đĩa, đọc bằng memmap; index: Filename -> hàng.

    Tạo một lần bằng ImageCache.build(...). Mỗi worker DataLoader tự mở memmap (lazy) nên không
    phải copy 3,4 GB vào từng tiến trình.
    """

    def __init__(self, npy_path: str | Path, index: dict[str, int]):
        self.npy_path = str(npy_path)
        self.index = index
        self._arr = None

    @property
    def arr(self):
        if self._arr is None:
            self._arr = np.load(self.npy_path, mmap_mode="r")
        return self._arr

    def get(self, filename: str) -> np.ndarray:
        return np.array(self.arr[self.index[filename]])

    def __getstate__(self):  # không pickle memmap sang worker
        d = dict(self.__dict__)
        d["_arr"] = None
        return d

    @classmethod
    def build(cls, filenames, images_dir: str | Path, cache_dir: str | Path, size: int = 256) -> "ImageCache":
        from PIL import Image

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        npy, idx_path = cache_dir / f"images_{size}.npy", cache_dir / f"images_{size}_index.json"
        filenames = sorted(set(filenames))
        if npy.exists() and idx_path.exists():
            index = json.loads(idx_path.read_text(encoding="utf-8"))
            if set(index) >= set(filenames):
                return cls(npy, index)
        arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(len(filenames), size, size, 3))
        for i, f in enumerate(filenames):
            with Image.open(Path(images_dir) / f) as im:
                im = im.convert("RGB")
                if im.size != (size, size):
                    im = im.resize((size, size), Image.BILINEAR)
                arr[i] = np.asarray(im)
        arr.flush()
        del arr
        index = {f: i for i, f in enumerate(filenames)}
        idx_path.write_text(json.dumps(index), encoding="utf-8")
        return cls(npy, index)


# --------------------------------------------------------------------------- #
# Dataset và DataLoader
# --------------------------------------------------------------------------- #
try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # cho phép import module không cần torch (test của repo gốc)
    _TorchDataset = object


class DeepWeedsDataset(_TorchDataset):
    """Dataset đọc ảnh theo DataFrame (Filename, Label). __getitem__ -> (tensor, int(label), filename)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: ImageCache | None = None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.cache = cache
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def load_image(self, i: int):
        """Ảnh gốc: ndarray HWC uint8 (có cache) hoặc PIL RGB."""
        f = self.filenames[i]
        if self.cache is not None:
            return self.cache.get(f)
        from PIL import Image
        with Image.open(self.images_dir / f) as im:
            return im.convert("RGB")

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), self.filenames[i]


def seed_worker(worker_id: int) -> None:
    """Seed numpy/random trong mỗi worker từ seed torch (đã được Generator của DataLoader cố định)."""
    import torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                cache: ImageCache | None = None, seed: int = 0):
    """Tạo DataLoader.

    - train=True: shuffle (hoặc sampler "balanced"), drop_last=True để BatchNorm không gặp batch 1-2 ảnh
    - train=False: không shuffle, giữ đúng thứ tự df (ghép logit với Filename)
    - sampler="balanced": WeightedRandomSampler, trọng số 1/(số ảnh của lớp), có hoàn lại
    - Generator + worker_init_fn cố định seed để tái lập
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform, cache)
    g = torch.Generator()
    g.manual_seed(seed)
    smp, shuffle = None, train
    if train and sampler == "balanced":
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES).astype(np.float64)
        w = 1.0 / counts[ds.labels]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                    replacement=True, generator=g)
        shuffle = False
    elif sampler not in (None, "none", "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler!r}")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=smp, drop_last=train,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=g, persistent_workers=num_workers > 0,
    )


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Tensor (C,H,W) hoặc (N,C,H,W) đã chuẩn hoá -> [0,1] để vẽ."""
    import torch
    m = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    s = torch.tensor(std, device=x.device).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1)

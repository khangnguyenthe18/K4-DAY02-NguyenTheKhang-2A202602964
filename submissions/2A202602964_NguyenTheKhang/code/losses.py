"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Hoàn thiện từ starter/losses.py. Giao diện giữ nguyên:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar

Lựa chọn cài đặt:
    - Label smoothing: TỰ CÀI ĐẶT theo công thức q'(k) = (1-eps)*1[k==y] + eps/K (không dùng tham số
      label_smoothing của PyTorch) để kiểm tra được bằng test: eps = 0 cho đúng CE; và kết quả khớp
      torch.nn.CrossEntropyLoss(label_smoothing=eps).
    - Mixup/CutMix: loss = lam * L(y_a) + (1 - lam) * L(y_b) (tổng có trọng số hai CE, tương đương
      CE với nhãn mềm vì CE tuyến tính theo nhãn).
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    _Module = nn.Module
except ImportError:  # cho phép import module không cần torch
    torch = None
    _Module = object

LOSS_KINDS = ("ce", "ls", "focal", "ce_weighted")


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor (bắt buộc với ce_weighted).
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        w = kw.get("weight")
        if w is None:
            raise ValueError("ce_weighted cần weight=class_weights(train_counts, beta)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))
    raise ValueError(f"loss phải thuộc {LOSS_KINDS}, nhận {kind!r}")


class LabelSmoothingCE(_Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        k = logits.shape[-1]
        nll = -logp.gather(1, target.view(-1, 1)).squeeze(1)
        uniform = -logp.mean(dim=-1)        # = sum_k (1/K) * (-log p_k)
        return ((1 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(_Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    alpha: None hoặc vector trọng số theo lớp (độ dài K). Trung bình theo batch.
    gamma = 0 và alpha = None cho đúng cross-entropy (có unit test).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target.view(-1, 1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t).clamp(min=0) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: w_c ∝ 1 / n_c, chuẩn hoá về trung bình 1
    - beta > 0: class-balanced (Cui et al. 2019): w_c = (1 - beta) / (1 - beta ** n_c),
      chuẩn hoá tổng trọng số về số lớp
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("mọi lớp phải có ít nhất 1 ảnh trong train")
    if beta and beta > 0:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w * len(n) / w.sum()
    else:
        w = 1.0 / n
        w = w / w.mean()
    return torch.as_tensor(w, dtype=torch.float32)


def rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator):
    """Hộp CutMix có diện tích ~ (1 - lam), tâm đều trên ảnh, cắt theo biên. Trả về (y1, y2, x1, x2)."""
    cut = math.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = int(rng.integers(h)), int(rng.integers(w))
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    return y1, y2, x1, x2


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix", rng: np.random.Generator | None = None):
    """Trộn một batch ảnh và nhãn. Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].

    - lam ~ Beta(alpha, alpha)
    - mixup : x_mix = lam * x + (1 - lam) * x[perm]
    - cutmix: dán hộp từ x[perm] vào x, rồi lam = 1 - (diện tích THỰC của hộp sau khi cắt biên) / (H*W)
    rng: numpy Generator (mặc định dùng np.random toàn cục, đã được set_seed cố định).
    """
    rng = rng or np.random.default_rng(np.random.randint(2**31))
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = rand_bbox(h, w, lam, rng)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    else:
        raise ValueError(f"mode phải là mixup hoặc cutmix, nhận {mode!r}")
    return x_mix, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

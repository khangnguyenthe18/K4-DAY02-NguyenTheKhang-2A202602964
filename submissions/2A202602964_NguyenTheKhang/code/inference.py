"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Hoàn thiện từ starter/inference.py. Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views_fn)   -> (filenames, y_true, [logits_view_k])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
    uniform_soup(state_dicts)                        -> state_dict (model soup, slide trang 67)

Mọi hàm chạy ở eval(), không gradient. Chọn phương pháp CHỈ trên val; T khớp trên VAL.
"""
from __future__ import annotations

import copy

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:
    torch = None


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _autocast(device, amp: bool):
    dev = device.type if hasattr(device, "type") else str(device).split(":")[0]
    return torch.autocast("cuda", dtype=torch.float16, enabled=(amp and dev == "cuda"))


@torch.inference_mode() if torch else (lambda f: f)
def predict_views(model, loader, device, views_fn, amp: bool = True):
    """Chạy model trên loader; views_fn(x) -> list K batch. Trả về (filenames, y_true, [logits_k])."""
    model.eval()
    names, ys, outs = [], [], None
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        with _autocast(device, amp):
            vs = views_fn(x)
            logits = [model(v).float().cpu() for v in vs]
        if outs is None:
            outs = [[] for _ in logits]
        for k, lg in enumerate(logits):
            outs[k].append(lg)
        ys.append(torch.as_tensor(y))
        names.extend(f)
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = True):
    """Chạy model trên loader, gom logit theo đúng thứ tự file. `view`: hàm biến đổi batch hoặc None."""
    view = view or view_identity
    names, y, (logits,) = predict_views(model, loader, device, lambda x: [view(x)], amp)
    return names, y, logits


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) trên chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[3])


def views_hflip(x):
    """TTA K = 2: ảnh gốc + lật ngang."""
    return [x, view_hflip(x)]


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop` từ batch lớn hơn; flip=True thêm bản lật (K = 10)."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop {crop} lớn hơn ảnh {h}x{w}")
    t, l = (h - crop) // 2, (w - crop) // 2
    boxes = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), (t, l)]
    out = [x[..., i:i + crop, j:j + crop] for i, j in boxes]
    if flip:
        out += [view_hflip(v) for v in out]
    return out


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias). Trả về list các batch.

    CNN có global pooling chạy được mọi kích thước. ViT/DeiT cần dynamic_img_size=True
    (model.build_model đã bật). Swin cố định kích thước cửa sổ: có thể lỗi, khi đó ghi "không áp dụng".
    """
    return [x if x.shape[-1] == s else F.interpolate(x, size=(s, s), mode="bilinear",
                                                       align_corners=False, antialias=True)
            for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy TTA (slide trang 62): "prob" = trung bình softmax; "logit" = trung bình logit rồi softmax."""
    arr = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])
    if space == "prob":
        p = np.mean([softmax_np(a) for a in arr], axis=0)
    elif space == "logit":
        p = softmax_np(arr.mean(0))
    else:
        raise ValueError("space phải là 'prob' hoặc 'logit'")
    return p / p.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    arr = np.stack([np.asarray(p, dtype=np.float64) for p in list_of_probs])
    p = arr.mean(0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels, max_iter: int = 200) -> float:
    """T > 0 cực tiểu NLL trên VAL của softmax(logit / T) (slide trang 69).

    Tối ưu LBFGS trên log T (đảm bảo T > 0), khởi đầu từ T tốt nhất của lưới thô 0.05..20.
    Accuracy không đổi vì argmax không đổi. KHÔNG gọi hàm này trên test.
    """
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 80))
    nll = [F.cross_entropy(z / t, y).item() for t in grid]
    log_t = torch.tensor([np.log(grid[int(np.argmin(nll))])], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float):
    """softmax(logits / T) dạng numpy float64."""
    return softmax_np(np.asarray(logits, dtype=np.float64) / float(T))


def crossfit_temperature_ece(val_logits, val_labels, ece_fn, seed: int = 0) -> dict:
    """ECE sau TS trên val không lạc quan: chia val làm 2 nửa, khớp T trên nửa này, đo trên nửa kia."""
    rng = np.random.default_rng(seed)
    n = len(val_labels)
    idx = rng.permutation(n)
    a, b = idx[: n // 2], idx[n // 2:]
    eces = []
    for fit_idx, ev_idx in ((a, b), (b, a)):
        t = fit_temperature(val_logits[fit_idx], val_labels[fit_idx])
        eces.append(ece_fn(apply_temperature(val_logits[ev_idx], t), val_labels[ev_idx]) * len(ev_idx))
    return {"ece_crossfit": float(sum(eces) / n)}


def _fuse(conv: "nn.Conv2d", bn: "nn.BatchNorm2d") -> "nn.Conv2d":
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64,
                                                                             device=w.device)
    scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode).to(conv.weight.device)
    fused.weight.data = (w * scale.view(-1, 1, 1, 1)).to(conv.weight.dtype)
    fused.bias.data = (bn.bias.detach().double() + (b - bn.running_mean.double()) * scale).to(conv.weight.dtype)
    # giữ hành vi forward đặc biệt của timm (ví dụ Conv2dSame pad động)
    if type(conv) is not nn.Conv2d:
        new = copy.deepcopy(conv)
        new.weight.data, new.bias = fused.weight.data, nn.Parameter(fused.bias.data)
        return new
    return fused


def _bn_replacement(bn):
    """BN thường -> Identity; timm BatchNormAct2d (BN + drop + act trong một module) -> drop + act."""
    if hasattr(bn, "act") and hasattr(bn, "drop"):
        return nn.Sequential(bn.drop, bn.act)
    return nn.Identity()


def fuse_conv_bn(model, check: bool = True, img_size: int = 224) -> "nn.Module":
    """Gộp BatchNorm vào tích chập liền trước (slide trang 71, 75), trên một BẢN SAO ở eval():

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Cặp (Conv2d, BatchNorm2d) liền kề theo thứ tự đăng ký module con (đúng với ResNet/ResNeXt/
    EfficientNet/MobileNetV3 của timm). check=True: in sai số lớn nhất trước/sau gộp.
    Kiến trúc không có BatchNorm (ViT, Swin, ConvNeXt dùng LayerNorm): trả về bản sao, n_fused = 0.
    """
    ref = model.eval()
    fused_model = copy.deepcopy(model).eval()
    n = 0
    for parent in fused_model.modules():
        names = list(parent._modules.keys())
        for a, b in zip(names, names[1:]):
            conv, bn = parent._modules[a], parent._modules[b]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d) and bn.track_running_stats \
                    and bn.num_features == conv.out_channels:
                parent._modules[a] = _fuse(conv, bn)
                parent._modules[b] = _bn_replacement(bn)
                n += 1
    fused_model.n_fused = n
    fused_model.max_abs_diff = 0.0
    if check:
        dev = next(model.parameters()).device
        x = torch.randn(2, 3, img_size, img_size, device=dev)
        with torch.inference_mode():
            d = (ref(x).float() - fused_model(x).float()).abs().max().item()
        fused_model.max_abs_diff = d
        print(f"fuse_conv_bn: gộp {n} cặp Conv-BN, sai số lớn nhất = {d:.2e}")
    return fused_model


def uniform_soup(state_dicts):
    """Model soup đồng đều: trung bình mọi tensor số thực của các state_dict cùng kiến trúc."""
    out = {}
    for k in state_dicts[0]:
        v0 = state_dicts[0][k]
        if torch.is_floating_point(v0):
            out[k] = sum(sd[k].double() for sd in state_dicts).div(len(state_dicts)).to(v0.dtype)
        else:
            out[k] = v0.clone()
    return out


@torch.no_grad() if torch else (lambda f: f)
def recalibrate_bn(model, loader, device, max_batches: int = 50):
    """Tính lại running mean/var của BN trên ảnh TRAIN (dùng cho soup/EMA, slide trang 56)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return model
    for m in bns:
        m.reset_running_stats()
        m.momentum = None   # trung bình tích luỹ
    model.train()
    for i, (x, _, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(x.to(device))
    model.eval()
    return model

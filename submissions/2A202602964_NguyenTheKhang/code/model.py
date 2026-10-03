"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Hoàn thiện từ starter/model.py. Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float

Bổ sung: weight_tag(model), data_config(model), set_train_mode(model), is_vit_like(name).
"""
from __future__ import annotations

# Gợi ý backbone (GUIDE.md mục 2.1). Tag thực sự được tải ghi trong config.json / results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}

INIT_MODES = ("scratch", "frozen", "finetune")


def is_vit_like(name: str) -> bool:
    """ViT/DeiT/DINOv2 dùng pos-embedding cố định: cần dynamic_img_size để chạy độ phân giải khác."""
    return any(k in name for k in ("vit_", "deit_", "dinov2")) or name.startswith("test_vit")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", drop_path_rate: float = 0.0,
                img_size: int | None = None):
    """Tạo model phân loại 9 lớp qua timm (head mới khởi tạo ngẫu nhiên).

    `init` (trục A): "scratch" (pretrained=False) | "frozen" (chỉ train head) | "finetune".
    Tag trọng số thực sự được tải: weight_tag(model).
    """
    import timm

    if init not in INIT_MODES:
        raise ValueError(f"init phải thuộc {INIT_MODES}")
    kw = dict(num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate:
        kw["drop_path_rate"] = drop_path_rate
    if is_vit_like(name):
        kw["dynamic_img_size"] = True       # cho phép TTA đa tỉ lệ / dò độ phân giải
        if img_size:
            kw["img_size"] = img_size
    model = timm.create_model(name, pretrained=(pretrained and init != "scratch"), **kw)
    init_head(model)
    model.init_mode = init
    if init == "frozen":
        freeze_backbone(model)
    return model


def init_head(model, std: float = 1e-3) -> None:
    """Khởi tạo lại head giống nhau cho mọi backbone: trọng số N(0, std) cắt cụt, bias 0.

    Init mặc định của timm khác nhau theo họ mạng (EfficientNet/MobileNetV3 dùng uniform theo
    fan_out = 9 lớp, rất lớn), làm loss ban đầu lệch xa ln 9 (Bước 0 đo được 4,9 và 6,9). Head
    nhỏ cho logit ≈ 0 nên loss ban đầu ≈ ln 9 với mọi backbone, so sánh công bằng hơn.
    """
    import torch

    for m in model.get_classifier().modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


def weight_tag(model) -> str:
    """Tên đầy đủ của bộ trọng số đã tải, ví dụ 'resnet50.a1_in1k' ('scratch' nếu không tải)."""
    if getattr(model, "init_mode", "finetune") == "scratch":
        return "scratch (random init)"
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    arch = cfg.get("architecture", type(model).__name__)
    tag = cfg.get("tag")
    return f"{arch}.{tag}" if tag else str(arch)


def data_config(model) -> dict:
    """mean/std/crop_pct đúng với trọng số đang dùng (timm pretrained_cfg)."""
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    return {"mean": tuple(cfg.get("mean", (0.485, 0.456, 0.406))),
            "std": tuple(cfg.get("std", (0.229, 0.224, 0.225))),
            "crop_pct": float(cfg.get("crop_pct", 0.875))}


def _head_param_ids(model) -> set[int]:
    head = model.get_classifier()
    return {id(p) for p in head.parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    BatchNorm của phần đóng băng phải ở chế độ eval khi train (GUIDE 3.2): nếu không, running
    mean/var vẫn bị cập nhật bằng thống kê của DeepWeeds trong khi gamma/beta không đổi, làm lệch
    đặc trưng mà head đang học. set_train_mode() lo việc này sau mỗi lần model.train().
    """
    head_ids = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone đóng băng thì đưa mọi module ngoài head về eval()."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        head = model.get_classifier()
        head_modules = set(head.modules())
        for m in model.modules():
            if m not in head_modules:
                m.eval()
        head.train()      # model.modules() gồm cả gốc: eval() ở gốc đã lan xuống head


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số theo slide trang 52 (thêm nhóm bias của head để bias không bị weight decay):

    - backbone ndim > 1                 : lr_backbone, weight_decay
    - backbone norm/bias (ndim <= 1),
      pos_embed / cls_token (timm no_weight_decay): lr_backbone, 0
    - head weight                       : lr_head, weight_decay
    - head bias                         : lr_head, 0
    Bỏ qua tham số requires_grad == False. Không trả nhóm rỗng.
    """
    head_ids = _head_param_ids(model)
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    buckets = {k: [] for k in ("bb_decay", "bb_no_decay", "head_decay", "head_no_decay")}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name in skip or name.split(".")[0] in skip
        part = "head" if id(p) in head_ids else "bb"
        buckets[f"{part}_{'no_decay' if no_decay else 'decay'}"].append(p)
    spec = {
        "bb_decay": (lr_backbone, weight_decay), "bb_no_decay": (lr_backbone, 0.0),
        "head_decay": (lr_head, weight_decay), "head_no_decay": (lr_head, 0.0),
    }
    return [{"params": ps, "lr": spec[k][0], "weight_decay": spec[k][1], "name": k}
            for k, ps in buckets.items() if ps]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Công cụ: fvcore.FlopCountAnalysis (fvcore đếm 1 "flop" cho mỗi phép nhân-cộng, tức là MAC,
    khớp quy ước của slide). Nếu không có fvcore: tự đếm bằng hook trên Conv2d và Linear (bỏ qua
    matmul của attention, nên thấp hơn thực tế với transformer; khi đó ghi chú trong xlsx).
    """
    import copy

    import torch

    m = copy.deepcopy(model).float().eval().cpu()
    x = torch.zeros(1, 3, img_size, img_size)
    try:
        import logging

        from fvcore.nn import FlopCountAnalysis
        logging.getLogger("fvcore").setLevel(logging.ERROR)
        fca = FlopCountAnalysis(m, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return float(fca.total()) / 1e9
    except Exception:
        pass
    macs = [0]

    def conv_hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
        macs[0] += out.numel() * k

    def lin_hook(mod, inp, out):
        macs[0] += out.numel() * mod.in_features

    hooks = []
    for mod in m.modules():
        if isinstance(mod, torch.nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
    with torch.no_grad():
        m(x)
    for h in hooks:
        h.remove()
    return macs[0] / 1e9

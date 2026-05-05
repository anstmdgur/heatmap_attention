import torch
import torch.nn.functional as F


TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
FOCAL_TVERSKY_GAMMA = 0.75
HEATMAP_LOSS_WEIGHT = 0.3
HEATMAP_DICE_WEIGHT = 0.5
HEATMAP_BCE_WEIGHT = 0.5
HEATMAP_BCE_POS_WEIGHT = 2.0


def focal_tversky_loss(logits, targets, smooth=1e-6):
    # 협착은 배경 비율이 매우 크므로 FN을 더 강하게 보는 Tversky 계열 loss를 사용합니다.
    probs = torch.sigmoid(logits.float()).flatten(1)
    targets = targets.float().flatten(1)

    true_pos = (probs * targets).sum(dim=1)
    false_pos = (probs * (1.0 - targets)).sum(dim=1)
    false_neg = ((1.0 - probs) * targets).sum(dim=1)

    score = (true_pos + smooth) / (
        true_pos
        + TVERSKY_ALPHA * false_pos
        + TVERSKY_BETA * false_neg
        + smooth
    )
    return (1.0 - score).clamp_min(1e-8).pow(FOCAL_TVERSKY_GAMMA).mean()


def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits.float()).flatten(1)
    targets = targets.float().flatten(1)

    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def weighted_bce_loss(logits, targets, pos_weight=HEATMAP_BCE_POS_WEIGHT):
    pos_weight_tensor = torch.tensor(
        float(pos_weight),
        device=logits.device,
        dtype=logits.dtype,
    )
    return F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        pos_weight=pos_weight_tensor,
    )


def heatmap_loss(logits, targets):
    return (
        HEATMAP_DICE_WEIGHT * dice_loss(logits, targets)
        + HEATMAP_BCE_WEIGHT * weighted_bce_loss(logits, targets)
    )


def _gaussian_kernel2d(kernel_size, sigma, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - ((kernel_size - 1) / 2.0)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


@torch.no_grad()
def build_heatmap_target(masks, output_size):
    # bbox 중심이 아니라 stenosis mask 자체를 부드러운 localization target으로 바꿉니다.
    target = F.interpolate(masks.float(), size=output_size, mode="area").clamp(0.0, 1.0)

    # 너무 얇은 stenosis가 128 scale에서 사라지지 않도록 주변 영역을 조금 넓힙니다.
    dilation_kernel = 5
    target = F.max_pool2d(
        target,
        kernel_size=dilation_kernel,
        stride=1,
        padding=dilation_kernel // 2,
    )

    # hard 0/1 target의 계단을 줄이기 위해 Gaussian blur를 적용합니다.
    blur_kernel = 9
    blur_sigma = 2.0
    kernel = _gaussian_kernel2d(blur_kernel, blur_sigma, masks.device, target.dtype)
    target = F.conv2d(target, kernel, padding=blur_kernel // 2).clamp(0.0, 1.0)

    # 이미지마다 병변 크기가 다르므로 max를 1로 맞춰 heatmap scale을 통일합니다.
    max_value = target.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
    target = torch.where(max_value > 1e-6, target / max_value.clamp_min(1e-6), target)
    return target.clamp(0.0, 1.0)

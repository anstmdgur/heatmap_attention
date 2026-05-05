import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arcade_metric import (
    add_mask_prediction,
    evaluate_arcade_stenosis_predictions,
    load_arcade_prediction_template,
    save_arcade_prediction_and_metrics,
)
from dataset import denormalize_image
from loss import HEATMAP_LOSS_WEIGHT, build_heatmap_target, focal_tversky_loss, heatmap_loss as heatmap_loss_fn
from metrics import (
    BinarySegmentationMeter,
    HEATMAP_PRED_THRESHOLD,
    HEATMAP_TARGET_THRESHOLD,
    SEG_THRESHOLD,
)


class EarlyStopping:
    def __init__(self, patience, checkpoint_path):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_score = None
        self.bad_epochs = 0
        self.early_stop = False

    def __call__(self, score, epoch, model, optimizer, scheduler, extra):
        # val F1이 좋아질 때만 checkpoint를 갱신합니다.
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_score": self.best_score,
                "extra": extra,
            }, self.checkpoint_path)
            print(f"  [*] checkpoint 저장: {self.checkpoint_path}")
        else:
            self.bad_epochs += 1
            print(f"  [*] early stopping count: {self.bad_epochs}/{self.patience}")
            if self.bad_epochs >= self.patience:
                self.early_stop = True


def compute_model_loss(outputs, masks):
    stenosis_logits = outputs["stenosis"]
    seg_loss = focal_tversky_loss(stenosis_logits, masks)

    heatmap_loss = torch.tensor(0.0, device=masks.device)
    heatmap_target = None
    if outputs["heatmap"] is not None:
        heatmap_target = build_heatmap_target(masks, outputs["heatmap"].shape[2:])
        heatmap_loss = heatmap_loss_fn(outputs["heatmap"], heatmap_target)

    heatmap_loss = HEATMAP_LOSS_WEIGHT * heatmap_loss
    total_loss = seg_loss + heatmap_loss
    return total_loss, seg_loss, heatmap_loss, heatmap_target


def train_one_epoch(loader, model, optimizer, device, model_name, epoch):
    model.train()
    total_loss_sum = 0.0
    seg_loss_sum = 0.0
    heat_loss_sum = 0.0
    seg_meter = BinarySegmentationMeter(threshold=SEG_THRESHOLD)
    heat_meter = BinarySegmentationMeter(
        threshold=HEATMAP_PRED_THRESHOLD,
        target_threshold=HEATMAP_TARGET_THRESHOLD,
    )

    pbar = tqdm(loader, desc=f"Train {epoch}")
    for images, masks, _filenames in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        total_loss, seg_loss, heat_loss, heatmap_target = compute_model_loss(outputs, masks)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss_sum += float(total_loss.detach().item())
        seg_loss_sum += float(seg_loss.detach().item())
        heat_loss_sum += float(heat_loss.detach().item())
        seg_meter.update_from_logits(outputs["stenosis"], masks)
        if outputs["heatmap"] is not None and heatmap_target is not None:
            heat_meter.update_from_logits(outputs["heatmap"], heatmap_target)

        pbar.set_postfix({
            "Loss": f"{total_loss.detach().item():.4f}",
            "Seg": f"{seg_loss.detach().item():.4f}",
            "Heat": f"{heat_loss.detach().item():.4f}",
        })

    batches = max(len(loader), 1)
    seg_metrics = seg_meter.compute()
    heat_metrics = heat_meter.compute() if model_name == "heatmap" else None
    return {
        "loss": total_loss_sum / batches,
        "seg_loss": seg_loss_sum / batches,
        "heatmap_loss": heat_loss_sum / batches,
        "seg": seg_metrics,
        "heatmap": heat_metrics,
    }


@torch.no_grad()
def evaluate_one_epoch(loader, model, device, model_name, epoch, desc="Val", arcade_annotation_path=None):
    model.eval()
    total_loss_sum = 0.0
    seg_loss_sum = 0.0
    heat_loss_sum = 0.0
    seg_meter = BinarySegmentationMeter(threshold=SEG_THRESHOLD)
    heat_meter = BinarySegmentationMeter(
        threshold=HEATMAP_PRED_THRESHOLD,
        target_threshold=HEATMAP_TARGET_THRESHOLD,
    )
    arcade_prediction = None
    if arcade_annotation_path is not None:
        arcade_prediction = load_arcade_prediction_template(arcade_annotation_path)

    pbar = tqdm(loader, desc=f"{desc} {epoch}")
    for images, masks, filenames in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        total_loss, seg_loss, heat_loss, heatmap_target = compute_model_loss(outputs, masks)

        total_loss_sum += float(total_loss.detach().item())
        seg_loss_sum += float(seg_loss.detach().item())
        heat_loss_sum += float(heat_loss.detach().item())
        seg_meter.update_from_logits(outputs["stenosis"], masks)
        if arcade_prediction is not None:
            stenosis_prob = torch.sigmoid(outputs["stenosis"])
            for batch_idx, filename in enumerate(filenames):
                raw_pred_mask = (
                    stenosis_prob[batch_idx].detach().cpu().squeeze().numpy()
                    > SEG_THRESHOLD
                )
                add_mask_prediction(arcade_prediction, filename, raw_pred_mask)
        if outputs["heatmap"] is not None and heatmap_target is not None:
            heat_meter.update_from_logits(outputs["heatmap"], heatmap_target)

    batches = max(len(loader), 1)
    seg_metrics = seg_meter.compute()
    heat_metrics = heat_meter.compute() if model_name == "heatmap" else None
    arcade_metrics = None
    if arcade_prediction is not None:
        arcade_metrics = evaluate_arcade_stenosis_predictions(
            arcade_annotation_path,
            arcade_prediction,
        )
    return {
        "loss": total_loss_sum / batches,
        "seg_loss": seg_loss_sum / batches,
        "heatmap_loss": heat_loss_sum / batches,
        "seg": seg_metrics,
        "heatmap": heat_metrics,
        "arcade": arcade_metrics,
    }


def _mask_overlay(gray_image, mask, color=(0.85, 0.10, 0.10), alpha=0.35):
    rgb = np.stack([gray_image, gray_image, gray_image], axis=-1)
    mask = mask.astype(bool)
    out = rgb.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * np.array(color)
    return np.clip(out, 0.0, 1.0)


def _heatmap_overlay(gray_image, heatmap, alpha=0.38):
    rgb = np.stack([gray_image, gray_image, gray_image], axis=-1)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    color = plt.get_cmap("jet")(heatmap)[..., :3]
    alpha_map = (heatmap * alpha)[..., None]
    out = rgb * (1.0 - alpha_map) + color * alpha_map
    return np.clip(out, 0.0, 1.0)


def _gaussian_blur_numpy(image, kernel_size=11, sigma=1.8):
    coords = np.arange(kernel_size, dtype=np.float32) - ((kernel_size - 1) / 2.0)
    kernel_1d = np.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / np.maximum(kernel_1d.sum(), 1e-12)
    kernel_2d = np.outer(kernel_1d, kernel_1d).astype(np.float32)

    pad = kernel_size // 2
    tensor = torch.from_numpy(image.astype(np.float32)).view(1, 1, *image.shape)
    kernel = torch.from_numpy(kernel_2d).view(1, 1, kernel_size, kernel_size)
    tensor = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(tensor, kernel)
    return blurred.squeeze().numpy()


def _binary_closing_numpy(mask, kernel_size=3):
    tensor = torch.from_numpy(mask.astype(np.float32)).view(1, 1, *mask.shape)
    pad = kernel_size // 2
    dilated = F.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=pad)
    closed = 1.0 - F.max_pool2d(1.0 - dilated, kernel_size=kernel_size, stride=1, padding=pad)
    return closed.squeeze().numpy() > 0.5


def _save_three_panel(path, left, middle, right, titles):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=400)
    images = [left, middle, right]
    for axis, image, title in zip(axes, images, titles):
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axis.imshow(image)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    plt.tight_layout(pad=0.2)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


@torch.no_grad()
def test_and_save(loader, model, device, model_name, output_dir, arcade_annotation_path=None):
    model.eval()

    seg_dir = os.path.join(output_dir, "test_images", "stenosis_seg")
    os.makedirs(seg_dir, exist_ok=True)
    heat_dir = None
    if model_name == "heatmap":
        heat_dir = os.path.join(output_dir, "test_images", "stenosis_heatmap")
        os.makedirs(heat_dir, exist_ok=True)

    seg_meter = BinarySegmentationMeter(threshold=SEG_THRESHOLD)
    heat_meter = BinarySegmentationMeter(
        threshold=HEATMAP_PRED_THRESHOLD,
        target_threshold=HEATMAP_TARGET_THRESHOLD,
    )
    inference_time_sum = 0.0
    inference_image_count = 0
    arcade_prediction = None
    if arcade_annotation_path is not None:
        arcade_prediction = load_arcade_prediction_template(arcade_annotation_path)

    for images, masks, filenames in tqdm(loader, desc="Test"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time_sum += time.perf_counter() - start_time
        inference_image_count += int(images.shape[0])

        stenosis_logits = outputs["stenosis"]
        seg_meter.update_from_logits(stenosis_logits, masks)

        heatmap_target = None
        if outputs["heatmap"] is not None:
            heatmap_target = build_heatmap_target(masks, outputs["heatmap"].shape[2:])
            heat_meter.update_from_logits(outputs["heatmap"], heatmap_target)

        stenosis_prob = torch.sigmoid(stenosis_logits)
        for batch_idx, filename in enumerate(filenames):
            stem = os.path.splitext(filename)[0] + ".png"
            image_np = denormalize_image(images[batch_idx])
            gt_mask = (masks[batch_idx].detach().cpu().squeeze().numpy() > 0.5)
            raw_pred_mask = (stenosis_prob[batch_idx].detach().cpu().squeeze().numpy() > SEG_THRESHOLD)
            if arcade_prediction is not None:
                add_mask_prediction(arcade_prediction, filename, raw_pred_mask)
            pred_mask = _binary_closing_numpy(raw_pred_mask, kernel_size=3)

            gt_overlay = _mask_overlay(image_np, gt_mask)
            pred_overlay = _mask_overlay(image_np, pred_mask, color=(0.05, 0.45, 0.95))
            _save_three_panel(
                os.path.join(seg_dir, stem),
                image_np,
                gt_overlay,
                pred_overlay,
                ["Original", "Original + GT", "Original + Pred"],
            )

            if heat_dir is not None and outputs["heatmap"] is not None and heatmap_target is not None:
                gt_heat = F.interpolate(
                    heatmap_target[batch_idx:batch_idx + 1],
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                ).detach().cpu().squeeze().numpy()
                pred_heat = F.interpolate(
                    torch.sigmoid(outputs["heatmap"][batch_idx:batch_idx + 1]),
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                ).detach().cpu().squeeze().numpy()
                pred_heat = _gaussian_blur_numpy(pred_heat, kernel_size=11, sigma=1.8)

                _save_three_panel(
                    os.path.join(heat_dir, stem),
                    image_np,
                    _heatmap_overlay(image_np, gt_heat),
                    _heatmap_overlay(image_np, pred_heat),
                    ["Original", "Original + GT Heatmap", "Original + Pred Heatmap"],
                )

    arcade_metrics = None
    if arcade_prediction is not None:
        arcade_metrics = evaluate_arcade_stenosis_predictions(
            arcade_annotation_path,
            arcade_prediction,
        )
        save_arcade_prediction_and_metrics(output_dir, arcade_prediction, arcade_metrics)

    return {
        "seg": seg_meter.compute(),
        "heatmap": heat_meter.compute() if model_name == "heatmap" else None,
        "arcade": arcade_metrics,
        "inference_time_per_image": inference_time_sum / max(inference_image_count, 1),
    }

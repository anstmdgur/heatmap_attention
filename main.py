import argparse
import csv
import os
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import make_stenosis_loaders
from loss import (
    FOCAL_TVERSKY_GAMMA,
    HEATMAP_BCE_POS_WEIGHT,
    HEATMAP_BCE_WEIGHT,
    HEATMAP_DICE_WEIGHT,
    HEATMAP_LOSS_WEIGHT,
    TVERSKY_ALPHA,
    TVERSKY_BETA,
)
from metrics import HEATMAP_PRED_THRESHOLD, HEATMAP_TARGET_THRESHOLD, SEG_THRESHOLD
from model import UNetPlusPlusBaseline, UNetPlusPlusHeatmapAttention
from train import EarlyStopping, evaluate_one_epoch, test_and_save, train_one_epoch


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "data", "ARCADE"))

# 논문 실험에서 고정할 기본값입니다. 자주 바꿀 필요가 없는 값은 상수로 둡니다.
EPOCHS = 250
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 30
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.2
SEED = 40


def parse_args():
    parser = argparse.ArgumentParser(description="ARCADE stenosis segmentation/heatmap training")
    parser.add_argument("--model", choices=["baseline", "heatmap"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name):
    if model_name == "baseline":
        return UNetPlusPlusBaseline()
    return UNetPlusPlusHeatmapAttention()


def metric_values(metrics):
    if metrics is None:
        return ["", "", "", ""]
    return [
        f"{metrics['iou']:.6f}",
        f"{metrics['precision']:.6f}",
        f"{metrics['recall']:.6f}",
        f"{metrics['f1']:.6f}",
    ]


def write_result_txt(path, args, checkpoint, test_metrics):
    seg = test_metrics["seg"]
    heatmap = test_metrics["heatmap"]
    arcade = test_metrics["arcade"]

    with open(path, "w", encoding="utf-8") as file:
        file.write("[Experiment Config]\n")
        file.write(f"model: {args.model}\n")
        file.write(f"data_dir: {args.data_dir}\n")
        file.write(f"output_dir: {args.output_dir}\n")
        file.write(f"test_only: {args.test_only}\n")
        file.write(f"epochs: {args.epochs}\n")
        file.write(f"batch_size: {args.batch_size}\n")
        file.write(f"num_workers: {args.num_workers}\n")
        file.write(f"lr: {args.lr:.4f}\n")
        file.write(f"weight_decay: {args.weight_decay:.4f}\n")
        file.write("optimizer: AdamW\n")
        file.write(
            "scheduler: ReduceLROnPlateau("
            f"factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})\n"
        )
        file.write(
            "early_stopping: "
            f"monitor=val_f1, patience={EARLY_STOP_PATIENCE}\n"
        )
        file.write(
            "seg_loss: "
            f"focal_tversky(alpha={TVERSKY_ALPHA}, beta={TVERSKY_BETA}, "
            f"gamma={FOCAL_TVERSKY_GAMMA})\n"
        )
        checkpoint_seed = checkpoint.get("extra", {}).get("seed", args.seed)
        file.write(f"seed: {checkpoint_seed}\n")
        if args.model == "heatmap":
            file.write(
                "heatmap_loss: "
                f"{HEATMAP_LOSS_WEIGHT:.2f} * "
                f"({HEATMAP_DICE_WEIGHT:.2f} * dice + "
                f"{HEATMAP_BCE_WEIGHT:.2f} * weighted_bce"
                f"(pos_weight={HEATMAP_BCE_POS_WEIGHT:.2f}))\n"
            )
        else:
            file.write("heatmap_loss: none\n")
        file.write(f"seg_threshold: {SEG_THRESHOLD:.2f}\n")
        file.write(f"heatmap_pred_threshold: {HEATMAP_PRED_THRESHOLD:.2f}\n")
        file.write(f"heatmap_target_threshold: {HEATMAP_TARGET_THRESHOLD:.2f}\n")
        file.write(
            "inference_time_per_image_sec: "
            f"{test_metrics['inference_time_per_image']:.6f}\n"
        )
        checkpoint_monitor = checkpoint.get("extra", {}).get("checkpoint_monitor", "val_seg_f1")
        file.write(f"checkpoint_monitor: {checkpoint_monitor}\n")
        file.write(f"best_epoch: {checkpoint['epoch'] + 1}\n")
        file.write(f"best_checkpoint_score: {checkpoint['best_score']:.4f}\n\n")

        if arcade is not None:
            file.write("[Test Stenosis ARCADE Official Polygon-Instance Metric]\n")
            file.write(f"Total Mean F1: {arcade['total_mean']:.4f}\n")
            file.write(
                "Evaluation note: COCO polygon instance F1, "
                "matching the official ARCADE stenosis evaluator logic.\n\n"
            )

        file.write("[Test Stenosis Segmentation]\n")
        file.write(f"IoU: {seg['iou']:.4f}\n")
        file.write(f"Precision: {seg['precision']:.4f}\n")
        file.write(f"Recall: {seg['recall']:.4f}\n")
        file.write(f"F1: {seg['f1']:.4f}\n")
        file.write(f"Image-wise F1: {seg['image_f1']:.4f}\n\n")

        if heatmap is not None:
            file.write("[Test Stenosis Heatmap]\n")
            file.write(f"IoU: {heatmap['iou']:.4f}\n")
            file.write(f"Precision: {heatmap['precision']:.4f}\n")
            file.write(f"Recall: {heatmap['recall']:.4f}\n")
            file.write(f"F1: {heatmap['f1']:.4f}\n")
            file.write(f"Image-wise F1: {heatmap['image_f1']:.4f}\n")


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = args.checkpoint_path or os.path.join(args.output_dir, "checkpoint.pt")
    history_path = os.path.join(args.output_dir, "training_history.csv")
    result_path = os.path.join(args.output_dir, "result.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_stenosis_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_arcade_annotation_path = os.path.join(
        args.data_dir,
        "stenosis",
        "val",
        "annotations",
        "val.json",
    )

    model = build_model(args.model).to(device)

    if args.test_only:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        print("=== Test Only Started ===")
        print(f"model: {args.model}")
        print(f"checkpoint: {checkpoint_path}")
        print(f"output_dir: {args.output_dir}")
        print(f"device: {device}")

        arcade_annotation_path = os.path.join(
            args.data_dir,
            "stenosis",
            "test",
            "annotations",
            "test.json",
        )
        test_metrics = test_and_save(
            test_loader,
            model,
            device,
            args.model,
            args.output_dir,
            arcade_annotation_path=arcade_annotation_path,
        )
        write_result_txt(result_path, args, checkpoint, test_metrics)

        print("=== Test Results ===")
        if test_metrics["arcade"] is not None:
            print(
                f"ARCADE Official Polygon-Instance F1: "
                f"{test_metrics['arcade']['total_mean']:.4f}"
            )
        print(
            f"Stenosis IoU/F1/Image-wise F1: "
            f"{test_metrics['seg']['iou']:.4f}/"
            f"{test_metrics['seg']['f1']:.4f}/"
            f"{test_metrics['seg']['image_f1']:.4f}"
        )
        if test_metrics["heatmap"] is not None:
            print(
                f"Heatmap IoU/F1/Image-wise F1: "
                f"{test_metrics['heatmap']['iou']:.4f}/"
                f"{test_metrics['heatmap']['f1']:.4f}/"
                f"{test_metrics['heatmap']['image_f1']:.4f}"
            )
        print(f"result.txt: {result_path}")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )
    early_stopping = EarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        checkpoint_path=checkpoint_path,
    )

    with open(history_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Epoch", "LR",
            "Train_Loss", "Train_Seg_Loss", "Train_Heatmap_Loss",
            "Train_Seg_IoU", "Train_Seg_Precision", "Train_Seg_Recall", "Train_Seg_F1",
            "Train_Heatmap_IoU", "Train_Heatmap_Precision", "Train_Heatmap_Recall", "Train_Heatmap_F1",
            "Val_Loss", "Val_Seg_Loss", "Val_Heatmap_Loss",
            "Val_Seg_IoU", "Val_Seg_Precision", "Val_Seg_Recall", "Val_Seg_F1",
            "Val_Heatmap_IoU", "Val_Heatmap_Precision", "Val_Heatmap_Recall", "Val_Heatmap_F1",
            "Val_ARCADE_Polygon_Instance_F1",
        ])

        print("=== Training Started ===")
        print(f"model: {args.model}")
        print(f"output_dir: {args.output_dir}")
        print(f"device: {device}")

        for epoch in range(args.epochs):
            current_lr = optimizer.param_groups[0]["lr"]
            train_metrics = train_one_epoch(
                train_loader, model, optimizer, device, args.model, epoch + 1
            )
            val_metrics = evaluate_one_epoch(
                val_loader,
                model,
                device,
                args.model,
                epoch + 1,
                desc="Val",
                arcade_annotation_path=val_arcade_annotation_path,
            )
            scheduler.step(val_metrics["loss"])
            val_arcade_f1 = (
                val_metrics["arcade"]["total_mean"]
                if val_metrics["arcade"] is not None
                else val_metrics["seg"]["f1"]
            )

            train_heat_values = metric_values(train_metrics["heatmap"])
            val_heat_values = metric_values(val_metrics["heatmap"])
            writer.writerow([
                epoch + 1,
                f"{current_lr:.8f}",
                f"{train_metrics['loss']:.6f}",
                f"{train_metrics['seg_loss']:.6f}",
                f"{train_metrics['heatmap_loss']:.6f}",
                *metric_values(train_metrics["seg"]),
                *train_heat_values,
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['seg_loss']:.6f}",
                f"{val_metrics['heatmap_loss']:.6f}",
                *metric_values(val_metrics["seg"]),
                *val_heat_values,
                f"{val_arcade_f1:.6f}",
            ])
            csv_file.flush()

            print(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"Train Loss {train_metrics['loss']:.4f} "
                f"SegF1 {train_metrics['seg']['f1']:.4f} | "
                f"Val Loss {val_metrics['loss']:.4f} "
                f"SegF1 {val_metrics['seg']['f1']:.4f} "
                f"ARCADE-F1 {val_arcade_f1:.4f} "
                f"P/R {val_metrics['seg']['precision']:.4f}/{val_metrics['seg']['recall']:.4f}"
            )
            if args.model == "heatmap":
                print(
                    f"  Heatmap Val IoU/F1 "
                    f"{val_metrics['heatmap']['iou']:.4f}/{val_metrics['heatmap']['f1']:.4f}"
                )

            early_stopping(
                score=val_arcade_f1,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra={
                    "model": args.model,
                    "seed": args.seed,
                    "checkpoint_monitor": "val_arcade_polygon_instance_f1",
                    "val_loss": val_metrics["loss"],
                    "val_seg_f1": val_metrics["seg"]["f1"],
                    "val_arcade_polygon_instance_f1": val_arcade_f1,
                },
            )
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print("=== Test Started ===")
    arcade_annotation_path = os.path.join(
        args.data_dir,
        "stenosis",
        "test",
        "annotations",
        "test.json",
    )
    test_metrics = test_and_save(
        test_loader,
        model,
        device,
        args.model,
        args.output_dir,
        arcade_annotation_path=arcade_annotation_path,
    )
    write_result_txt(result_path, args, checkpoint, test_metrics)

    print("=== Test Results ===")
    if test_metrics["arcade"] is not None:
        print(
            f"ARCADE Official Polygon-Instance F1: "
            f"{test_metrics['arcade']['total_mean']:.4f}"
        )
    print(
        f"Stenosis IoU/F1/Image-wise F1: "
        f"{test_metrics['seg']['iou']:.4f}/"
        f"{test_metrics['seg']['f1']:.4f}/"
        f"{test_metrics['seg']['image_f1']:.4f}"
    )
    if test_metrics["heatmap"] is not None:
        print(
            f"Heatmap IoU/F1/Image-wise F1: "
            f"{test_metrics['heatmap']['iou']:.4f}/"
            f"{test_metrics['heatmap']['f1']:.4f}/"
            f"{test_metrics['heatmap']['image_f1']:.4f}"
        )
    print(f"result.txt: {result_path}")


if __name__ == "__main__":
    main()

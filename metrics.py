import torch


SEG_THRESHOLD = 0.5
HEATMAP_PRED_THRESHOLD = 0.35
HEATMAP_TARGET_THRESHOLD = 0.25


class BinarySegmentationMeter:
    def __init__(self, threshold=SEG_THRESHOLD, target_threshold=0.5):
        self.threshold = threshold
        self.target_threshold = target_threshold
        self.reset()

    def reset(self):
        self.iou_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.f1_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update_from_logits(self, logits, targets):
        probs = torch.sigmoid(logits.detach().float())
        self.update_from_probs(probs, targets)

    @torch.no_grad()
    def update_from_probs(self, probs, targets):
        preds = (probs > self.threshold).float().flatten(1)
        targets = (targets.detach().float() > self.target_threshold).float().flatten(1)

        true_pos = (preds * targets).sum(dim=1)
        false_pos = (preds * (1.0 - targets)).sum(dim=1)
        false_neg = ((1.0 - preds) * targets).sum(dim=1)

        iou = true_pos / (true_pos + false_pos + false_neg + 1e-7)
        precision = true_pos / (true_pos + false_pos + 1e-7)
        recall = true_pos / (true_pos + false_neg + 1e-7)
        f1 = (2.0 * true_pos) / ((2.0 * true_pos) + false_pos + false_neg + 1e-7)

        # GT와 prediction이 모두 비어 있는 이미지는 맞춘 것으로 처리합니다.
        empty = (true_pos + false_pos + false_neg) <= 0
        iou = torch.where(empty, torch.ones_like(iou), iou)
        precision = torch.where(empty, torch.ones_like(precision), precision)
        recall = torch.where(empty, torch.ones_like(recall), recall)
        f1 = torch.where(empty, torch.ones_like(f1), f1)

        self.iou_sum += float(iou.sum().item())
        self.precision_sum += float(precision.sum().item())
        self.recall_sum += float(recall.sum().item())
        self.f1_sum += float(f1.sum().item())
        self.count += int(probs.shape[0])

    def compute(self):
        count = max(self.count, 1)
        image_f1 = self.f1_sum / count
        return {
            "iou": self.iou_sum / count,
            "precision": self.precision_sum / count,
            "recall": self.recall_sum / count,
            "f1": image_f1,
            "image_f1": image_f1,
            "count": self.count,
        }

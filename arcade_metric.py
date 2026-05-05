import json
import os
from statistics import mean

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon


STENOSIS_CATEGORY_ID = 26


def _calculate_polygon_f1(gt_mask, pred_mask):
    pred_list = []
    for i in range(0, len(pred_mask), 2):
        pred_list.append([pred_mask[i], pred_mask[i + 1]])
    gt_list = []
    for i in range(0, len(gt_mask), 2):
        gt_list.append([gt_mask[i], gt_mask[i + 1]])

    checker = False
    try:
        intersection = Polygon(pred_list).intersection(Polygon(gt_list))
    except Exception:
        try:
            pred_polygon = Polygon(pred_list).buffer(0)
            gt_polygon = Polygon(gt_list).buffer(0)
            intersection = pred_polygon.intersection(gt_polygon)
            checker = True
        except Exception:
            return 0

    if str(type(intersection)) == "<class 'shapely.geometry.multipolygon.MultiPolygon'>":
        intersection = MultiPolygon([
            item for item in intersection.geoms
            if str(type(item)) == "<class 'shapely.geometry.polygon.Polygon'>"
        ])
        polygons = list(intersection.geoms)
        if len(polygons) == 0:
            return 0
        areas = [item.area for item in polygons]
        for index in range(0, len(polygons)):
            if polygons[index].area == max(areas):
                intersection = polygons[index]

    tp = intersection.area
    if tp == 0:
        return 0

    if checker:
        fp = (Polygon(pred_list).buffer(0) - intersection).area
        fn = (Polygon(gt_list).buffer(0) - intersection).area
    else:
        fp = (Polygon(pred_list).buffer(0) - intersection).area
        fn = (Polygon(gt_list).buffer(0) - intersection).area

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)


def _attach_file_names(annotation):
    file_name_by_id = {
        image["id"]: image["file_name"]
        for image in annotation["images"]
    }
    for item in annotation["annotations"]:
        item["file_name"] = file_name_by_id[item["image_id"]]


def _build_image_information(gt_annotation, pred_annotation):
    list_images_information = []
    for image in gt_annotation["images"]:
        list_images_information.append({
            "image_id": image["id"],
            "image_name": image["file_name"],
        })

    for item in list_images_information:
        gt_classes = []
        pred_classes = []
        for annotation in gt_annotation["annotations"]:
            if annotation["image_id"] == item["image_id"]:
                gt_classes.append(annotation["category_id"])
        for annotation in pred_annotation["annotations"]:
            if annotation["file_name"] == item["image_name"]:
                pred_classes.append(annotation["category_id"])
        item["num_gt_masks"] = len(gt_classes)
        item["num_pred_masks"] = len(pred_classes)

    return list_images_information


def evaluate_arcade_stenosis_predictions(gt_annotation_path, pred_annotation):
    with open(gt_annotation_path, "r", encoding="utf-8") as file:
        gt_annotation = json.load(file)

    if len(gt_annotation["images"]) != len(pred_annotation["images"]):
        raise ValueError("GT and prediction image counts do not match.")
    if len(gt_annotation["categories"]) != len(pred_annotation["categories"]):
        raise ValueError("GT and prediction category counts do not match.")

    _attach_file_names(gt_annotation)
    _attach_file_names(pred_annotation)
    image_information = _build_image_information(gt_annotation, pred_annotation)

    for item in image_information:
        f1_scores = []
        for pred_item in pred_annotation["annotations"]:
            f1_scores_for_pred = []
            if item["image_name"] == pred_item["file_name"]:
                for gt_item in gt_annotation["annotations"]:
                    if pred_item["file_name"] == gt_item["file_name"]:
                        f1_scores_for_pred.append(
                            _calculate_polygon_f1(
                                pred_item["segmentation"][0],
                                gt_item["segmentation"][0],
                            )
                        )
            if len(f1_scores_for_pred) > 0:
                f1_scores.append(max(f1_scores_for_pred))

        for _ in range(0, item["num_gt_masks"] - len(f1_scores)):
            f1_scores.append(0)

        item["f1_scores"] = f1_scores

    all_means = []
    for item in image_information:
        item["mean_f1_score"] = mean(item["f1_scores"]) if len(item["f1_scores"]) > 0 else 0.0
        all_means.append(item["mean_f1_score"])

    return {
        "total_mean": mean(all_means) if len(all_means) > 0 else 0.0,
        "all_means_per_image": all_means,
        "total_info": image_information,
    }


def load_arcade_prediction_template(gt_annotation_path):
    with open(gt_annotation_path, "r", encoding="utf-8") as file:
        gt_annotation = json.load(file)
    return {
        "licenses": gt_annotation.get("licenses", []),
        "info": gt_annotation.get("info", {}),
        "categories": gt_annotation["categories"],
        "images": gt_annotation["images"],
        "annotations": [],
    }


def add_mask_prediction(pred_annotation, filename, mask, min_area=1.0):
    image_id_by_name = {
        image["file_name"]: image["id"]
        for image in pred_annotation["images"]
    }
    if filename not in image_id_by_name:
        return

    mask_uint8 = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        if contour.shape[0] < 3:
            continue
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        points = contour.reshape(-1, 2).astype(float)
        segmentation = []
        for x_coord, y_coord in points:
            segmentation.extend([float(x_coord), float(y_coord)])
        if len(segmentation) < 6:
            continue

        x_coord, y_coord, width, height = cv2.boundingRect(contour)
        pred_annotation["annotations"].append({
            "id": len(pred_annotation["annotations"]) + 1,
            "image_id": image_id_by_name[filename],
            "category_id": STENOSIS_CATEGORY_ID,
            "segmentation": [segmentation],
            "area": area,
            "bbox": [
                float(x_coord),
                float(y_coord),
                float(width),
                float(height),
            ],
            "iscrowd": 0,
        })


def save_arcade_prediction_and_metrics(output_dir, pred_annotation, metrics):
    pred_path = os.path.join(output_dir, "test_arcade_predictions.json")
    metrics_path = os.path.join(output_dir, "test_arcade_metrics.json")
    with open(pred_path, "w", encoding="utf-8") as file:
        json.dump(pred_annotation, file)
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file)
    return pred_path, metrics_path

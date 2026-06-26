import argparse
import os

import pandas as pd


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def load_gt(gt_file):
    gt = pd.read_csv(gt_file, header=None)

    # In this dataset GT has MOT-like format:
    # frame, id, x, y, w, h, mark, -1, -1, -1
    # We keep only valid marked objects.
    if gt.shape[1] >= 7:
        gt = gt[gt[6] == 1]

    return gt


def load_det(det_file, min_confidence):
    det = pd.read_csv(det_file, header=None)

    # MOT det format:
    # frame, id, x, y, w, h, confidence, -1, -1, -1
    det = det[det[6] >= min_confidence]

    return det


def evaluate_detection(gt_file, det_file, iou_threshold=0.5, min_confidence=0.0):
    gt = load_gt(gt_file)
    det = load_det(det_file, min_confidence)

    frames = sorted(set(gt[0].unique()) | set(det[0].unique()))

    tp = 0
    fp = 0
    fn = 0

    for frame in frames:
        frame_gt = gt[gt[0] == frame]
        frame_det = det[det[0] == frame]

        gt_boxes = frame_gt[[2, 3, 4, 5]].values.tolist()
        det_boxes = frame_det[[2, 3, 4, 5]].values.tolist()

        matched_gt = set()
        matched_det = set()

        pairs = []

        for det_idx, det_box in enumerate(det_boxes):
            for gt_idx, gt_box in enumerate(gt_boxes):
                score = bbox_iou(det_box, gt_box)
                pairs.append((score, det_idx, gt_idx))

        pairs.sort(reverse=True)

        for score, det_idx, gt_idx in pairs:
            if score < iou_threshold:
                continue

            if det_idx in matched_det:
                continue

            if gt_idx in matched_gt:
                continue

            matched_det.add(det_idx)
            matched_gt.add(gt_idx)
            tp += 1

        fp += len(det_boxes) - len(matched_det)
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--det-file", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    metrics = evaluate_detection(
        gt_file=args.gt_file,
        det_file=args.det_file,
        iou_threshold=args.iou_threshold,
        min_confidence=args.min_confidence,
    )

    print("Detection evaluation")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"TP: {metrics['tp']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()

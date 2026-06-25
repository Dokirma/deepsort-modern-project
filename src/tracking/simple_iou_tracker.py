import argparse
import os

import numpy as np
import pandas as pd


def iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
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


def run_iou_tracker(det_file, output_file, min_confidence=0.0, iou_threshold=0.3, max_age=5):
    det = pd.read_csv(det_file, header=None)

    # frame, id, x, y, w, h, confidence, -1, -1, -1
    det = det[det[6] >= min_confidence]

    active_tracks = {}
    next_track_id = 1
    results = []

    frames = sorted(det[0].unique())

    for frame in frames:
        frame_dets = det[det[0] == frame]

        detections = []
        for _, row in frame_dets.iterrows():
            detections.append({
                "bbox": [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
                "confidence": float(row[6]),
            })

        assigned_tracks = set()
        assigned_detections = set()

        track_ids = list(active_tracks.keys())

        pairs = []
        for track_id in track_ids:
            track_box = active_tracks[track_id]["bbox"]
            for det_idx, detection in enumerate(detections):
                pairs.append((iou(track_box, detection["bbox"]), track_id, det_idx))

        pairs.sort(reverse=True)

        for score, track_id, det_idx in pairs:
            if score < iou_threshold:
                continue
            if track_id in assigned_tracks:
                continue
            if det_idx in assigned_detections:
                continue

            active_tracks[track_id]["bbox"] = detections[det_idx]["bbox"]
            active_tracks[track_id]["age"] = 0

            assigned_tracks.add(track_id)
            assigned_detections.add(det_idx)

        for det_idx, detection in enumerate(detections):
            if det_idx in assigned_detections:
                continue

            active_tracks[next_track_id] = {
                "bbox": detection["bbox"],
                "age": 0,
            }
            assigned_tracks.add(next_track_id)
            next_track_id += 1

        for track_id in list(active_tracks.keys()):
            if track_id not in assigned_tracks:
                active_tracks[track_id]["age"] += 1

            if active_tracks[track_id]["age"] > max_age:
                del active_tracks[track_id]

        for track_id, track in active_tracks.items():
            x, y, w, h = track["bbox"]
            results.append([frame, track_id, x, y, w, h, 1, -1, -1, -1])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for row in results:
            f.write(",".join(map(str, row)) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=5)
    args = parser.parse_args()

    run_iou_tracker(
        det_file=args.det_file,
        output_file=args.output_file,
        min_confidence=args.min_confidence,
        iou_threshold=args.iou_threshold,
        max_age=args.max_age,
    )


if __name__ == "__main__":
    main()

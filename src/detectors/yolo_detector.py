import argparse
import configparser
import os
import time

import cv2
from ultralytics import YOLO


def read_seqinfo(sequence_dir):
    seqinfo_path = os.path.join(sequence_dir, "seqinfo.ini")
    config = configparser.ConfigParser()
    config.read(seqinfo_path)

    info = config["Sequence"]
    return {
        "im_dir": info.get("imDir", "img1"),
        "seq_length": info.getint("seqLength"),
        "im_ext": info.get("imExt", ".jpg"),
    }


def run_yolo_detector(sequence_dir, output_file, model_name, conf, iou, imgsz):
    info = read_seqinfo(sequence_dir)

    model = YOLO(model_name)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    rows = []
    total_time = 0.0
    processed_frames = 0

    for frame_id in range(1, info["seq_length"] + 1):
        image_name = f"{frame_id:06d}{info['im_ext']}"
        image_path = os.path.join(sequence_dir, info["im_dir"], image_name)

        image = cv2.imread(image_path)
        if image is None:
            continue

        start = time.time()

        results = model.predict(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=[0],  
            verbose=False,
        )

        elapsed = time.time() - start
        total_time += elapsed
        processed_frames += 1

        result = results[0]

        if result.boxes is None:
            continue

        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0].cpu().numpy())

            w = x2 - x1
            h = y2 - y1

            rows.append([
                frame_id,
                -1,
                float(x1),
                float(y1),
                float(w),
                float(h),
                score,
                -1,
                -1,
                -1,
            ])

    with open(output_file, "w") as f:
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")

    fps = processed_frames / total_time if total_time > 0 else 0.0

    print(f"Model: {model_name}")
    print(f"Sequence: {sequence_dir}")
    print(f"Frames: {processed_frames}")
    print(f"Detections: {len(rows)}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"FPS: {fps:.2f}")
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    run_yolo_detector(
        sequence_dir=args.sequence_dir,
        output_file=args.output_file,
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()

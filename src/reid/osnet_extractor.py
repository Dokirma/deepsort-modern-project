import argparse
import configparser
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchreid
from torchvision import transforms


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


def build_model(model_name, device):
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=1000,
        pretrained=True,
    )
    model.eval()
    model.to(device)
    return model


def crop_person(image, bbox):
    x, y, w, h = bbox

    h_img, w_img = image.shape[:2]

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(w_img, int(round(x + w)))
    y2 = min(h_img, int(round(y + h)))

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def extract_features(sequence_dir, det_file, output_npy, model_name="osnet_x0_25", batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    info = read_seqinfo(sequence_dir)
    detections = pd.read_csv(det_file, header=None)

    model = build_model(model_name, device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    all_features = []
    valid_rows = []

    total_time = 0.0

    for frame_id in sorted(detections[0].unique()):
        image_name = f"{int(frame_id):06d}{info['im_ext']}"
        image_path = os.path.join(sequence_dir, info["im_dir"], image_name)

        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_dets = detections[detections[0] == frame_id]

        crops = []
        rows = []

        for _, row in frame_dets.iterrows():
            bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
            crop = crop_person(image_rgb, bbox)

            if crop is None:
                continue

            crops.append(transform(crop))
            rows.append(row.values.astype(np.float32))

        if not crops:
            continue

        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            batch_rows = rows[start:start + batch_size]

            batch_tensor = torch.stack(batch).to(device)

            t0 = time.time()
            with torch.no_grad():
                features = model(batch_tensor)
                features = F.normalize(features, p=2, dim=1)
            total_time += time.time() - t0

            features = features.cpu().numpy().astype(np.float32)

            all_features.append(features)
            valid_rows.extend(batch_rows)

    valid_rows = np.asarray(valid_rows, dtype=np.float32)
    all_features = np.concatenate(all_features, axis=0)

    detection_mat = np.concatenate([valid_rows, all_features], axis=1)

    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    np.save(output_npy, detection_mat)

    fps_like = len(valid_rows) / total_time if total_time > 0 else 0.0

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Detections with features: {len(valid_rows)}")
    print(f"Feature dim: {all_features.shape[1]}")
    print(f"ReID time: {total_time:.2f} sec")
    print(f"Detections/sec: {fps_like:.2f}")
    print(f"Saved: {output_npy}")
    print(f"Shape: {detection_mat.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--det-file", required=True)
    parser.add_argument("--output-npy", required=True)
    parser.add_argument("--model", default="osnet_x0_25")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    extract_features(
        sequence_dir=args.sequence_dir,
        det_file=args.det_file,
        output_npy=args.output_npy,
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

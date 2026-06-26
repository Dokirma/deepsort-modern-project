import argparse
import os

import numpy as np
import pandas as pd


def make_pseudo_feature(row, feature_dim=128):
    frame = float(row[0])
    x = float(row[2])
    y = float(row[3])
    w = float(row[4])
    h = float(row[5])
    conf = float(row[6])

    cx = x + w / 2.0
    cy = y + h / 2.0
    area = w * h
    aspect = w / h if h > 0 else 0.0

    base = np.array([
        frame / 1000.0,
        x / 1000.0,
        y / 1000.0,
        w / 1000.0,
        h / 1000.0,
        cx / 1000.0,
        cy / 1000.0,
        area / 100000.0,
        aspect,
        conf / 100.0,
    ], dtype=np.float32)

    feature = np.tile(base, int(np.ceil(feature_dim / len(base))))[:feature_dim]

    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm

    return feature


def convert(det_txt, output_npy, feature_dim=128):
    det = pd.read_csv(det_txt, header=None)

    features = []
    for _, row in det.iterrows():
        features.append(make_pseudo_feature(row, feature_dim=feature_dim))

    features = np.asarray(features, dtype=np.float32)
    det_values = det.values.astype(np.float32)

    detection_mat = np.concatenate([det_values, features], axis=1)

    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    np.save(output_npy, detection_mat)

    print(f"Saved: {output_npy}")
    print(f"Shape: {detection_mat.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-txt", required=True)
    parser.add_argument("--output-npy", required=True)
    parser.add_argument("--feature-dim", type=int, default=128)
    args = parser.parse_args()

    convert(
        det_txt=args.det_txt,
        output_npy=args.output_npy,
        feature_dim=args.feature_dim,
    )


if __name__ == "__main__":
    main()

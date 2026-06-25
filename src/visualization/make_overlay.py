import argparse
import configparser
import os

import cv2
import pandas as pd


def read_seqinfo(sequence_dir):
    seqinfo_path = os.path.join(sequence_dir, "seqinfo.ini")
    config = configparser.ConfigParser()
    config.read(seqinfo_path)

    info = config["Sequence"]
    return {
        "im_dir": info.get("imDir", "img1"),
        "frame_rate": info.getint("frameRate", 25),
        "seq_length": info.getint("seqLength"),
        "im_width": info.getint("imWidth"),
        "im_height": info.getint("imHeight"),
        "im_ext": info.get("imExt", ".jpg"),
    }


def make_overlay(sequence_dir, tracks_file, output_video):
    info = read_seqinfo(sequence_dir)

    tracks = pd.read_csv(tracks_file, header=None)
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        info["frame_rate"],
        (info["im_width"], info["im_height"]),
    )

    for frame_id in range(1, info["seq_length"] + 1):
        image_name = f"{frame_id:06d}{info['im_ext']}"
        image_path = os.path.join(sequence_dir, info["im_dir"], image_name)

        frame = cv2.imread(image_path)
        if frame is None:
            continue

        frame_tracks = tracks[tracks[0] == frame_id]

        for _, row in frame_tracks.iterrows():
            track_id = int(row[1])
            x, y, w, h = map(float, row[[2, 3, 4, 5]])

            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        writer.write(frame)

    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--tracks-file", required=True)
    parser.add_argument("--output-video", required=True)
    args = parser.parse_args()

    make_overlay(
        sequence_dir=args.sequence_dir,
        tracks_file=args.tracks_file,
        output_video=args.output_video,
    )


if __name__ == "__main__":
    main()

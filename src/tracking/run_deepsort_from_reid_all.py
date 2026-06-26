import argparse
import os
import subprocess
import sys


SEQUENCES = [
    "KITTI-17",
    "MOT16-09",
    "MOT16-11",
    "PETS09-S2L1",
    "TUD-Campus",
    "TUD-Stadtmitte",
]


def run_command(command):
    print(" ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reid-dir", required=True)
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--max-cosine-distance", default="0.30")
    parser.add_argument("--nn-budget", default="100")
    args = parser.parse_args()

    os.makedirs(f"outputs/tracks/{args.tracker_name}", exist_ok=True)

    for seq in SEQUENCES:
        print("=" * 80)
        print(f"Running DeepSORT with ReID features for {seq}")

        detection_file = f"{args.reid_dir}/{seq}.npy"
        sequence_dir = f"data/mot/videos/{seq}"
        output_file = f"outputs/tracks/{args.tracker_name}/{seq}.txt"

        run_command([
            sys.executable,
            "deep_sort_app.py",
            "--sequence_dir", sequence_dir,
            "--detection_file", detection_file,
            "--output_file", output_file,
            "--min_confidence", "0",
            "--min_detection_height", "0",
            "--nms_max_overlap", "1.0",
            "--max_cosine_distance", args.max_cosine_distance,
            "--nn_budget", args.nn_budget,
            "--display", "False",
        ])

        print(f"Saved result: {output_file}")


if __name__ == "__main__":
    main()

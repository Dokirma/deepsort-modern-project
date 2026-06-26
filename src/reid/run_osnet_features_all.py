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
    parser.add_argument("--det-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="osnet_x0_25")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for seq in SEQUENCES:
        print("=" * 80)
        print(f"Extracting ReID features for {seq}")

        run_command([
            sys.executable,
            "src/reid/osnet_extractor.py",
            "--sequence-dir", f"data/mot/videos/{seq}",
            "--det-file", f"{args.det_dir}/{seq}.txt",
            "--output-npy", f"{args.output_dir}/{seq}.npy",
            "--model", args.model,
            "--batch-size", str(args.batch_size),
        ])


if __name__ == "__main__":
    main()

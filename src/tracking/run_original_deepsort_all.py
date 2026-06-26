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
    os.makedirs("outputs/detections", exist_ok=True)
    os.makedirs("outputs/tracks/original_deepsort", exist_ok=True)

    for seq in SEQUENCES:
        print("=" * 80)
        print(f"Running original DeepSORT baseline for {seq}")

        det_txt = f"data/mot/videos/{seq}/det/det.txt"
        det_npy = f"outputs/detections/{seq}_pseudo_features.npy"
        sequence_dir = f"data/mot/videos/{seq}"
        output_file = f"outputs/tracks/original_deepsort/{seq}.txt"

        run_command([
            sys.executable,
            "src/data/convert_det_to_npy.py",
            "--det-txt", det_txt,
            "--output-npy", det_npy,
        ])

        run_command([
            sys.executable,
            "deep_sort_app.py",
            "--sequence_dir", sequence_dir,
            "--detection_file", det_npy,
            "--output_file", output_file,
            "--min_confidence", "0",
            "--min_detection_height", "0",
            "--nms_max_overlap", "1.0",
            "--max_cosine_distance", "0.2",
            "--nn_budget", "100",
            "--display", "False",
        ])

        print(f"Saved result: {output_file}")


if __name__ == "__main__":
    main()

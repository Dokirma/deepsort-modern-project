import argparse
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
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="yolo11n_conf040")
    args = parser.parse_args()

    for seq in SEQUENCES:
        print("=" * 80)
        print(f"Running {args.model} on {seq}")

        run_command([
            sys.executable,
            "src/detectors/yolo_detector.py",
            "--sequence-dir", f"data/mot/videos/{seq}",
            "--output-file", f"outputs/detections/{args.name}/{seq}.txt",
            "--model", args.model,
            "--conf", str(args.conf),
            "--iou", str(args.iou),
            "--imgsz", str(args.imgsz),
        ])


if __name__ == "__main__":
    main()

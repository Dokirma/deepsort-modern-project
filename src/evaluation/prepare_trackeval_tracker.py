import argparse
import os
import shutil

import pandas as pd


def prepare_gt_for_trackeval(src_gt, dst_gt):
    gt = pd.read_csv(src_gt, header=None)

    # TrackEval expects MOTChallenge-like GT:
    # frame, id, x, y, w, h, mark, class, visibility
    gt = gt.iloc[:, :9].copy()
    gt[6] = 1
    gt[7] = 1
    gt[8] = 1.0

    gt.to_csv(dst_gt, header=False, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--tracks-dir", required=True)
    parser.add_argument("--seqmap-file", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    args = parser.parse_args()

    gt_root = "data/trackeval/gt/DLCV-train"
    tracker_root = f"data/trackeval/trackers/DLCV-train/{args.tracker_name}/data"
    seqmap_dir = "data/trackeval/seqmaps"

    os.makedirs(gt_root, exist_ok=True)
    os.makedirs(tracker_root, exist_ok=True)
    os.makedirs(seqmap_dir, exist_ok=True)

    for seq in args.sequences:
        src_seq_dir = f"data/mot/videos/{seq}"

        dst_seq_dir = f"{gt_root}/{seq}"
        dst_gt_dir = f"{dst_seq_dir}/gt"

        os.makedirs(dst_gt_dir, exist_ok=True)

        prepare_gt_for_trackeval(
            src_gt=f"{src_seq_dir}/gt/gt.txt",
            dst_gt=f"{dst_gt_dir}/gt.txt",
        )

        shutil.copyfile(
            f"{src_seq_dir}/seqinfo.ini",
            f"{dst_seq_dir}/seqinfo.ini",
        )

        shutil.copyfile(
            f"{args.tracks_dir}/{seq}.txt",
            f"{tracker_root}/{seq}.txt",
        )

        print(f"Prepared {seq}")

    seqmap_path = f"{seqmap_dir}/{args.seqmap_file}"
    with open(seqmap_path, "w") as f:
        f.write("name\n")
        for seq in args.sequences:
            f.write(seq + "\n")

    print(f"Saved seqmap: {seqmap_path}")


if __name__ == "__main__":
    main()

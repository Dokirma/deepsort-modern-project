import os
import shutil


SEQUENCES = [
    "KITTI-17",
    "MOT16-09",
    "MOT16-11",
    "PETS09-S2L1",
    "TUD-Campus",
    "TUD-Stadtmitte",
]


def main():
    gt_root = "data/trackeval/gt/DLCV-train"
    tracker_root = "data/trackeval/trackers/DLCV-train/original_deepsort/data"
    seqmap_dir = "data/trackeval/seqmaps"

    os.makedirs(gt_root, exist_ok=True)
    os.makedirs(tracker_root, exist_ok=True)
    os.makedirs(seqmap_dir, exist_ok=True)

    for seq in SEQUENCES:
        src_gt = f"data/mot/videos/{seq}/gt/gt.txt"
        dst_gt_dir = f"{gt_root}/{seq}/gt"
        dst_gt = f"{dst_gt_dir}/gt.txt"

        os.makedirs(dst_gt_dir, exist_ok=True)
        shutil.copyfile(src_gt, dst_gt)

        src_track = f"outputs/tracks/original_deepsort/{seq}.txt"
        dst_track = f"{tracker_root}/{seq}.txt"

        shutil.copyfile(src_track, dst_track)

        print(f"Prepared {seq}")

    seqmap_path = f"{seqmap_dir}/DLCV-train.txt"
    with open(seqmap_path, "w") as f:
        f.write("name\n")
        for seq in SEQUENCES:
            f.write(seq + "\n")

    print(f"Saved seqmap: {seqmap_path}")


if __name__ == "__main__":
    main()

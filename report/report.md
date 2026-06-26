# DeepSORT Modernization Project Report

## 1. Project goal

The goal of this project is to modify the original DeepSORT implementation by adding modern person detection and person ReID models, evaluating the system on MOT-style videos, and comparing the modified system with the original baseline.

The project uses the original DeepSORT repository as the basis. 

## 2. Dataset

The dataset was provided as a Google Drive archive instead of being downloaded from the MOTChallenge website.

The archive contains six MOT-style sequences:

| Sequence | Frames | GT rows | Unique IDs |
|---|---:|---:|---:|
| KITTI-17 | 145 | 782 | 9 |
| MOT16-09 | 525 | 8830 | 43 |
| MOT16-11 | 900 | 10076 | 80 |
| PETS09-S2L1 | 795 | 4650 | 19 |
| TUD-Campus | 71 | 359 | 8 |
| TUD-Stadtmitte | 179 | 1156 | 10 |

Each sequence contains frames in `img1/`, ground-truth annotations in `gt/gt.txt`, detections in `det/det.txt`, and sequence metadata in `seqinfo.ini`.

## 3. Initial detection baseline

Before replacing the detector with modern models, the provided `det.txt` files were evaluated against ground-truth bounding boxes.

The evaluation used IoU threshold 0.5. A detection was counted as a true positive if it matched a ground-truth bounding box in the same frame with IoU ≥ 0.5.

| Sequence | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| KITTI-17 | 432 | 119 | 251 | 0.7840 | 0.6325 | 0.7002 |
| MOT16-09 | 2989 | 877 | 2268 | 0.7732 | 0.5686 | 0.6553 |
| MOT16-11 | 4948 | 796 | 4226 | 0.8614 | 0.5394 | 0.6634 |
| PETS09-S2L1 | 4082 | 1495 | 394 | 0.7319 | 0.9120 | 0.8121 |
| TUD-Campus | 230 | 92 | 129 | 0.7143 | 0.6407 | 0.6755 |
| TUD-Stadtmitte | 833 | 295 | 323 | 0.7385 | 0.7206 | 0.7294 |

The results show that the provided detections are usable but not perfect. The main weakness is recall, especially on MOT16-09 and MOT16-11, where many ground-truth objects are missed. This motivates the use of stronger modern detectors in the next experiments.

## 4. Current implementation state

At this stage, the project contains:

- the original DeepSORT implementation;
- dataset downloading and extraction code;
- MOT-style dataset inspection;
- a simple IoU tracker baseline;
- overlay visualization for tracking results;
- detection Precision / Recall / F1 evaluation.

The simple IoU tracker is not the final method. It is used as an early technical baseline to verify that detections, tracking output format, and overlay generation work correctly.


## 5. Original DeepSORT baseline reproduction

The original DeepSORT implementation was successfully merged into the project while preserving the original repository history.

The original `deep_sort_app.py` expects a detection matrix where the first 10 columns follow the MOTChallenge detection format and the remaining columns contain appearance descriptors. Since the provided `det.txt` files contain only MOT-format detections, a converter from `det.txt` to `.npy` was implemented.

At this stage, pseudo appearance descriptors were used only to verify that the original DeepSORT pipeline can run on the provided data. This is not the final ReID solution.

The original DeepSORT baseline was successfully executed on `TUD-Campus`, and an overlay video was generated:

- tracking result: `outputs/tracks/original_deepsort/TUD-Campus.txt`
- overlay video: `outputs/videos/TUD-Campus_original_deepsort.mp4`

This confirms that the original DeepSORT baseline pipeline works on the provided MOT-style data.


## 6. Original DeepSORT baseline outputs

The original DeepSORT baseline was executed on all six required sequences. The generated tracking outputs are listed below.

| Sequence | Tracking output rows |
|---|---:|
| KITTI-17 | 498 |
| MOT16-09 | 3983 |
| MOT16-11 | 5830 |
| PETS09-S2L1 | 5492 |
| TUD-Campus | 285 |
| TUD-Stadtmitte | 1140 |

These files are stored in `outputs/tracks/original_deepsort/`. Since the `outputs/` directory is ignored by Git, the numerical summary is saved in the report folder for reproducibility.


## 6. Original DeepSORT baseline outputs

The original DeepSORT baseline was executed on all six required sequences. The generated tracking outputs are listed below.

| Sequence | Tracking output rows |
|---|---:|
| KITTI-17 | 498 |
| MOT16-09 | 3983 |
| MOT16-11 | 5830 |
| PETS09-S2L1 | 5492 |
| TUD-Campus | 285 |
| TUD-Stadtmitte | 1140 |



## 6. Original DeepSORT baseline outputs

The original DeepSORT baseline was executed on all six required sequences. The tracking outputs are listed below.

| Sequence | Tracking output rows |
|---|---:|
| KITTI-17 | 498 |
| MOT16-09 | 3983 |
| MOT16-11 | 5830 |
| PETS09-S2L1 | 5492 |
| TUD-Campus | 285 |
| TUD-Stadtmitte | 1140 |



## 7. Original DeepSORT baseline HOTA evaluation

The original DeepSORT baseline was evaluated using TrackEval with HOTA, CLEAR and Identity metrics.

| Sequence | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|---:|
| KITTI-17 | 38.351 | 39.150 | 37.752 | 63.594 | 45.396 |
| MOT16-09 | 25.344 | 28.163 | 22.935 | 31.999 | 31.869 |
| MOT16-11 | 26.957 | 36.284 | 20.095 | 32.202 | 40.363 |
| PETS09-S2L1 | 41.363 | 52.649 | 32.724 | 53.757 | 63.462 |
| TUD-Campus | 35.183 | 42.198 | 29.556 | 52.174 | 44.290 |
| TUD-Stadtmitte | 37.729 | 41.582 | 34.684 | 55.052 | 43.166 |
| **COMBINED** | **31.257** | **37.293** | **26.310** | **39.665** | **41.949** |

The combined HOTA of the original baseline is 31.257. The weakest results are observed on MOT16-09 and MOT16-11. These sequences have low association accuracy and relatively low detection recall, which indicates that both the detector and the appearance/ReID component should be improved in the modified system.

# DeepSORT Modernization Project Report

## 1. Project goal

The goal of this project is to modify the original DeepSORT implementation by adding modern person detection and person ReID models, evaluating the system on MOT-style videos, and comparing the modified system with the original baseline.

The project uses the original DeepSORT repository as the basis. 

## 2. Dataset

The evaluation was performed on six MOT-style video sequences required by the assignment.

| Sequence | Source group | Role in evaluation |
|---|---|---|
| KITTI-17 | MOT15-style data | Outdoor street scene with moving pedestrians |
| MOT16-09 | MOT16 | Crowded urban pedestrian tracking sequence |
| MOT16-11 | MOT16 | Crowded pedestrian tracking sequence with many identities |
| PETS09-S2L1 | MOT15 | Public surveillance-style pedestrian sequence |
| TUD-Campus | MOT15-style data | Short campus sequence used for detector and ReID parameter tests |
| TUD-Stadtmitte | MOT15-style data | Urban pedestrian sequence used in the final benchmark |

The sequences were stored in MOTChallenge-style format. Each sequence contains image frames, sequence metadata, provided detections and ground-truth annotations. The expected structure is:

```text
sequence/
  img1/
  det/det.txt
  gt/gt.txt
  seqinfo.ini
```

The dataset was provided through a Google Drive archive instead of downloading directly from the MOTChallenge website. The archive contains all six required sequences:

```text
KITTI-17
MOT16-09
MOT16-11
PETS09-S2L1
TUD-Campus
TUD-Stadtmitte
```

The same six sequences were used for the original DeepSORT baseline, detector-only DeepSORT experiment, ReID-based DeepSORT experiment and the final full benchmark. This makes the comparison fair because all configurations were evaluated on the same data split.

The main evaluation metric was HOTA averaged over all six videos. Additional metrics from TrackEval were also reported: DetA, AssA, IDF1 and MOTA. Detector quality was evaluated separately with Precision, Recall and F1 using ground-truth bounding boxes and IoU threshold 0.5.

## 3. Initial detection baseline

The first step was to run the original DeepSORT-style baseline on all six sequences. This baseline used the original/provided detection input and original-style or pseudo appearance descriptors.

The baseline was evaluated with TrackEval using HOTA, CLEAR, Identity and Count metrics.

| Sequence | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|---:|
| KITTI-17 | 46.932 | 51.542 | 42.788 | 64.384 | 58.824 |
| MOT16-09 | 29.571 | 30.091 | 29.212 | 33.438 | 36.512 |
| MOT16-11 | 32.120 | 42.168 | 24.523 | 28.963 | 39.113 |
| PETS09-S2L1 | 33.969 | 55.338 | 21.095 | 30.922 | 76.538 |
| TUD-Campus | 41.390 | 40.565 | 42.357 | 53.028 | 45.404 |
| TUD-Stadtmitte | 43.668 | 58.736 | 32.582 | 58.301 | 72.837 |
| **COMBINED** | **31.257** | **39.811** | **25.401** | **39.665** | **41.949** |

The original baseline achieved combined HOTA 31.257. This value was used as the reference for all later comparisons.


## 4. Current implementation state

The original DeepSORT pipeline depends strongly on the quality of input detections. Therefore, the first modernization step was to replace the original detection input with modern YOLO-based person detectors.

The detector script supports model selection before execution. Three detector models were tested, satisfying the requirement to support not less than three detection models.

| Detector model | Repository/family | Purpose |
|---|---|---|
| YOLO11n | Ultralytics | Main final detector candidate |
| YOLOv8n | Ultralytics | Lightweight detector candidate |
| YOLOv5nu | Ultralytics | Additional lightweight detector candidate |

The selected detector confidence threshold for the main experiments was 0.40. The detector IoU threshold was set to 0.60. The image size was 640.

The detector was evaluated independently before the full tracking benchmark. This was done because the detection stage influences both tracking quality and speed. For detector-only evaluation, predicted boxes were matched with ground-truth boxes using IoU threshold 0.5. The following metrics were reported:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Detector speed was also measured in frames per second. All tested lightweight YOLO detectors were faster than the required 5 FPS threshold in Colab.


## 5. Original DeepSORT baseline reproduction

The original DeepSORT implementation was successfully merged into the project while preserving the original repository history.

Detector quality was evaluated separately from tracking quality. The goal of this experiment was to compare candidate person detectors before using them inside DeepSORT.

The matching criterion was IoU threshold 0.5 between predicted bounding boxes and ground-truth bounding boxes. Detections were evaluated using Precision, Recall and F1.

### 5.1 YOLO11n detector examples

YOLO11n was selected as the final detector because it was fully evaluated in the tracking pipeline on all six required sequences and gave the best complete final benchmark result with OSNet x0.5.

Example YOLO11n detector results:

| Sequence | Frames | Detections | FPS | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PETS09-S2L1 | 795 | 4566 | 59.40 | 4232 | 507 | 414 | 0.8930 | 0.9109 | 0.9018 |
| TUD-Campus | 71 | 382 | 33.62 | 292 | 90 | 67 | 0.7644 | 0.8134 | 0.7881 |
| TUD-Stadtmitte | 179 | 992 | 53.14 | 962 | 30 | 194 | 0.9698 | 0.8322 | 0.8957 |

These results show that YOLO11n is fast enough for real-time use in Colab and provides strong detection quality, especially on PETS09-S2L1 and TUD-Stadtmitte.

### 5.2 Detector model comparison on TUD-Campus


| Detector | Sequence | Frames | Detections | FPS | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | TUD-Campus | 71 | 382 | 33.62 | 0.7644 | 0.8134 | 0.7881 |
| YOLOv8n | TUD-Campus | 71 | 379 | 48.96 | 0.7863 | 0.8301 | 0.8076 |
| YOLOv5nu | TUD-Campus | 71 | 376 | 47.42 | 0.7713 | 0.8078 | 0.7891 |

YOLOv8n achieved the best standalone detector result on TUD-Campus: its F1 score was 0.8076, compared with 0.7881 for YOLO11n and 0.7891 for YOLOv5nu. Therefore, YOLOv8n was additionally tested inside the full DeepSORT pipeline with OSNet x0.5 on TUD-Campus.

In this candidate tracking experiment, YOLOv8n + OSNet x0.5 slightly improved HOTA on TUD-Campus from 48.294 to 48.450 and also improved MOTA from 55.710 to 57.939. However, IDF1 decreased from 67.218 to 64.666, which means that identity preservation became worse. In addition, this candidate was tested only on TUD-Campus, while YOLO11n + OSNet x0.5 was evaluated on all six required sequences.

For this reason, YOLOv8n is reported as the best standalone detector candidate and as a promising tracking candidate for TUD-Campus, but the final selected full-benchmark configuration remains YOLO11n + OSNet x0.5.

### 5.3 Detector influence on tracking

Replacing the original detection input with YOLO11n produced a large improvement even before adding modern ReID descriptors.

| Method | Combined HOTA | Combined IDF1 | Combined MOTA |
|---|---:|---:|---:|
| Original DeepSORT | 31.257 | 39.665 | 41.949 |
| DeepSORT + YOLO11n | 42.369 | 49.166 | 53.116 |

This means that the detector update alone increased combined HOTA by 11.112 points.


## 6. Detector-only DeepSORT tracking benchmark

After replacing the detector with YOLO11n, DeepSORT was evaluated with external detector outputs and pseudo/original-style appearance features.

Configuration:

```text
detector: YOLO11n
confidence threshold: 0.40
detector IoU threshold: 0.60
ReID: pseudo/original-style features
```

| Sequence | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|---:|
| KITTI-17 | 53.283 | 53.568 | 53.016 | 77.457 | 64.194 |
| MOT16-09 | 33.007 | 32.728 | 33.465 | 37.369 | 37.508 |
| MOT16-11 | 42.320 | 45.566 | 39.692 | 45.346 | 49.127 |
| PETS09-S2L1 | 50.293 | 65.713 | 38.575 | 63.337 | 83.484 |
| TUD-Campus | 46.566 | 44.274 | 49.137 | 65.385 | 53.482 |
| TUD-Stadtmitte | 53.033 | 61.426 | 46.025 | 64.299 | 77.336 |
| **COMBINED** | **42.369** | **45.813** | **39.615** | **49.166** | **53.116** |

Replacing only the detector increased combined HOTA from 31.257 to 42.369. This confirmed that modern detector quality had a strong positive influence on the final tracker.

parison, OSNet x0.5 was evaluated on all six sequences.
## 7. ReID modernization

The next step was to replace the original-style appearance representation with modern ReID descriptors. The ReID extraction script supports model selection before execution.

The following ReID models were tested:

| ReID model | Feature dimension | Notes |
|---|---:|---|
| OSNet x0.25 | 512 | Lightweight OSNet model |
| OSNet x0.5 | 512 | Stronger OSNet model selected for final full benchmark |
| MobileNetV2 x1.0 | 1280 | Fast candidate model tested on TUD-Campus |

OSNet models were loaded through Torchreid. The MobileNetV2 model produced a warning about weights, so this result was treated as an experimental candidate and not selected as the final full-benchmark model.

## 8. ReID model comparison on TUD-Campus

A ReID model comparison was performed on TUD-Campus with the same YOLO11n detections and DeepSORT parameters.

Configuration:

```text
detector: YOLO11n
detector confidence threshold: 0.40
DeepSORT max_cosine_distance: 0.30
DeepSORT nn_budget: 100
```

| Detector | ReID model | Feature dim | HOTA | DetA | AssA | IDF1 | MOTA | ReID detections/sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | OSNet x0.25 | 512 | 44.197 | 47.869 | 41.160 | 60.606 | 53.482 | 224.80 |
| YOLO11n | OSNet x0.5 | 512 | 48.294 | 48.359 | 48.454 | 67.218 | 55.710 | 252.67 |
| YOLO11n | MobileNetV2 x1.0 | 1280 | 50.421 | 48.135 | 52.991 | 72.099 | 56.546 | 462.50 |

MobileNetV2 x1.0 achieved the best TUD-Campus result among the three tested ReID models, but because of weights warning it was not chosen as the final result.

OSNet x0.5 was selected for the full final benchmark because it improved over OSNet x0.25 and was fully evaluated on all required sequences.

## 9. Parameter tuning

The main DeepSORT association parameter tuned during the experiments was `max_cosine_distance`. The parameter was tested on TUD-Campus using YOLO11n detections and OSNet x0.25 descriptors.

| Configuration | max_cosine_distance | HOTA | IDF1 | MOTA | IDs |
|---|---:|---:|---:|---:|---:|
| YOLO11n + OSNet x0.25 | 0.30 | 44.197 | 60.606 | 53.482 | 9 |
| YOLO11n + OSNet x0.25 | 0.40 | 44.197 | 60.606 | 53.482 | 9 |
| YOLO11n + OSNet x0.25 | 0.50 | 44.098 | 60.440 | 52.925 | 8 |

The best parameter value from this tuning was `max_cosine_distance = 0.30`. This value was used in the final configuration.

Final selected parameters:

```text
detector: YOLO11n
detector confidence threshold: 0.40
detector IoU threshold: 0.60
ReID model: OSNet x0.5
DeepSORT max_cosine_distance: 0.30
DeepSORT nn_budget: 100
```

The parameter tuning was not a full exhaustive grid over all possible detector and tracker parameters, but it demonstrates the evolution of association parameters and their influence on tracking metrics.

## 10. Full benchmark with YOLO11n + OSNet x0.25

Before selecting OSNet x0.5, YOLO11n + OSNet x0.25 was evaluated on all six sequences.

| Sequence | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|---:|
| KITTI-17 | 53.287 | 53.561 | 53.031 | 77.279 | 63.939 |
| MOT16-09 | 32.969 | 32.978 | 33.148 | 36.615 | 38.029 |
| MOT16-11 | 44.625 | 45.868 | 43.795 | 50.324 | 49.365 |
| PETS09-S2L1 | 52.779 | 65.660 | 42.520 | 68.561 | 83.613 |
| TUD-Campus | 44.197 | 47.869 | 41.160 | 60.606 | 53.482 |
| TUD-Stadtmitte | 53.554 | 61.276 | 46.946 | 72.183 | 77.595 |
| **COMBINED** | **43.816** | **46.058** | **42.130** | **52.271** | **53.414** |

This configuration improved the detector-only result and served as a strong ReID baseline.

## 11. Final full benchmark with YOLO11n + OSNet x0.5

After the TUD-Campus ReID comparison, OSNet x0.5 was evaluated on all six required sequences.

Final configuration:

```text
detector: YOLO11n
detector confidence threshold: 0.40
detector IoU threshold: 0.60
ReID model: OSNet x0.5
ReID feature dimension: 512
DeepSORT max_cosine_distance: 0.30
DeepSORT nn_budget: 100
```

| Sequence | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|---:|
| KITTI-17 | 53.418 | 53.696 | 53.159 | 77.569 | 64.450 |
| MOT16-09 | 33.100 | 32.986 | 33.403 | 38.168 | 37.701 |
| MOT16-11 | 46.015 | 45.851 | 46.510 | 53.131 | 49.375 |
| PETS09-S2L1 | 55.085 | 65.709 | 46.261 | 69.370 | 83.763 |
| TUD-Campus | 48.294 | 48.359 | 48.454 | 67.218 | 55.710 |
| TUD-Stadtmitte | 55.625 | 61.397 | 50.607 | 71.034 | 77.422 |
| **COMBINED** | **45.111** | **46.087** | **44.574** | **54.055** | **53.371** |

The final configuration achieved the best combined HOTA among all fully evaluated configurations.

## 12. Comparison with the original baseline

| Method | Combined HOTA | Combined IDF1 | Combined MOTA |
|---|---:|---:|---:|
| Original DeepSORT | 31.257 | 39.665 | 41.949 |
| DeepSORT + YOLO11n | 42.369 | 49.166 | 53.116 |
| DeepSORT + YOLO11n + OSNet x0.25 | 43.816 | 52.271 | 53.414 |
| DeepSORT + YOLO11n + OSNet x0.5 | 45.111 | 54.055 | 53.371 |

The final system improved combined HOTA from 31.257 to 45.111.

Per-sequence HOTA comparison:

| Sequence | Original HOTA | Final HOTA | Improvement |
|---|---:|---:|---:|
| KITTI-17 | 46.932 | 53.418 | +6.486 |
| MOT16-09 | 29.571 | 33.100 | +3.529 |
| MOT16-11 | 32.120 | 46.015 | +13.895 |
| PETS09-S2L1 | 33.969 | 55.085 | +21.116 |
| TUD-Campus | 41.390 | 48.294 | +6.904 |
| TUD-Stadtmitte | 43.668 | 55.625 | +11.957 |
| **COMBINED** | **31.257** | **45.111** | **+13.854** |

The final configuration improves HOTA on every required sequence.

## 13. Candidate YOLOv8n + OSNet x0.5 experiment

Since YOLOv8n achieved better standalone detection F1 than YOLO11n on TUD-Campus, an additional combined tracking experiment was performed with YOLOv8n and OSNet x0.5 on TUD-Campus.

| Detector | ReID model | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---|---:|---:|---:|---:|---:|
| YOLO11n | OSNet x0.5 | 48.294 | 48.359 | 48.454 | 67.218 | 55.710 |
| YOLOv8n | OSNet x0.5 | 48.450 | 50.721 | 46.365 | 64.666 | 57.939 |

YOLOv8n + OSNet x0.5 slightly improved HOTA and MOTA on TUD-Campus, but IDF1 decreased compared with YOLO11n + OSNet x0.5. Since this candidate was not evaluated on all six sequences and the identity metric was worse, it was not selected as the final full-benchmark configuration.

## 14. Overlay videos

Overlay videos were generated for both the initial baseline and the final best modified version.

The videos are not stored directly in the GitHub repository because they are generated output files. After running the final Colab notebook, they are created in the Colab runtime under:

```text
outputs/videos_web/

The generated runtime paths are:

| Sequence | Original DeepSORT overlay | Best modified overlay |
|---|---|---|
| KITTI-17 | `outputs/videos_web/original_deepsort/KITTI-17_original_deepsort.mp4` | `outputs/videos_web/best_yolo11n_osnet_x0_5/KITTI-17_best_yolo11n_osnet_x0_5.mp4` |
| MOT16-09 | `outputs/videos_web/original_deepsort/MOT16-09_original_deepsort.mp4` | `outputs/videos_web/best_yolo11n_osnet_x0_5/MOT16-09_best_yolo11n_osnet_x0_5.mp4` |
| MOT16-11 | `outputs/videos_web/original_deepsort/MOT16-11_original_deepsort.mp4` | `outputs/videos_web/best_yolo11n_osnet_x0_5/MOT16-11_best_yolo11n_osnet_x0_5.mp4` |
| PETS09-S2L1 | `outputs/videos_web/original_deepsort/PETS09-S2L1_original_deepsort.mp4` | `outputs/videos_web/best_yolo11n_osnet_x0_5/PETS09-S2L1_best_yolo11n_osnet_x0_5.mp4` |
| TUD-Campus | `outputs/videos_web/original_deepsort/TUD-Campus_original_deepsort.mp4` | `outputs/videos_web/best_yolo11n_osnet_x0_5/TUD-Campus_best_yolo11n_osnet_x0_5.mp4` |
| TUD-Stadtmitte | `outputs/videos_web/original_deepsort/TUD-Stadtmitte_original_deepsort.mp4` | `outputs/videos_web/best_yolo11n_osnet_x0_5/TUD-Stadtmitte_best_yolo11n_osnet_x0_5.mp4` |

In total, the final notebook creates 12 web-compatible overlay videos:

```text
6 original DeepSORT overlays
6 final best modified DeepSORT overlays
```

For direct visual demonstration in Colab, the notebook embeds two representative videos for TUD-Campus:

```text
Original DeepSORT overlay: TUD-Campus
Best modified DeepSORT overlay: TUD-Campus
```

The videos are converted to H.264 format for browser-compatible playback in Colab.

## 15. Colab reproducibility

A clean Colab notebook is provided in the repository:

```text
notebooks/DeepSORT_Project_clean_submission.ipynb
```

The notebook is designed to run in a clean Colab runtime. It performs the following steps:

1. Clone the GitHub repository.
2. Install dependencies.
3. Download and extract the dataset.
4. Run original DeepSORT tracking.
5. Run YOLO11n detections.
6. Extract OSNet x0.5 ReID features.
7. Run the final best DeepSORT configuration.
8. Generate 12 overlay videos.
9. Convert videos to web-compatible format.
10. Show two representative overlay videos directly in Colab.
11. Create a backup archive with generated outputs.

Generated outputs are not committed directly to GitHub because they include videos and feature tensors. A separate backup archive can be shared if required.

The final generated backup archive contains:

```text
12 tracking txt files
6 detection txt files
6 ReID npy files
12 web-compatible overlay mp4 videos
report files
```

## 16. Repository and development history

The repository contains:

```text
application_util/
deep_sort/
src/
report/
tools/
notebooks/
README.md
deep_sort_app.py
requirements.txt
```

The original DeepSORT implementation was preserved as the basis, and development was performed through separate commits. The repository contains the original project history and additional commits for:

```text
TrackEval runner
YOLO detector support
external detection runner
OSNet feature extraction
ReID-based DeepSORT runner
detector comparison
ReID comparison
parameter tuning
full benchmark results
overlay generation
report and README updates
clean Colab notebook
```

The large generated files are ignored by Git and are regenerated by the Colab notebook.

## 17. Limitations

The main task was completed: the detector and ReID components of DeepSORT were modernized, the system was evaluated on all required sequences, and the final configuration improved HOTA on every video compared with the original baseline.

The additional task for 9-10 points was not fully completed.

Not completed:

```text
standalone body ReID with persistent identity database
kNN identity assignment with cluster management
time-window majority voting for identity resolution
conflict resolution between active identities
segmentation model support
```

Segmentation model support was also not implemented. The project therefore targets the main-task score range rather than the additional-task score range.

## 18. Conclusion

The final selected configuration is:

```text
YOLO11n detector
confidence threshold = 0.40
detector IoU threshold = 0.60
OSNet x0.5 ReID
max_cosine_distance = 0.30
nn_budget = 100
```

This configuration achieved:

```text
Combined HOTA = 45.111
Combined IDF1 = 54.055
Combined MOTA = 53.371
```

The original DeepSORT baseline achieved:

```text
Combined HOTA = 31.257
Combined IDF1 = 39.665
Combined MOTA = 41.949
```

Thus, the final version increased combined HOTA by 13.854 points and improved tracking quality on every required video sequence.

The experiments show that the largest improvement came from replacing the original detection input with a modern YOLO-based person detector. Adding OSNet x0.5 ReID descriptors further improved association quality and IDF1. The final system is reproducible in Colab, runs with real-time detector speed above the required threshold, and provides overlay videos for both the original and final modified trackers on all six required sequences.

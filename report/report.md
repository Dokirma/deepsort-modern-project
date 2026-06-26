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


## 8. First modern detector experiment: YOLO11n

The first modern detector experiment was performed with YOLO11n on the `TUD-Campus` sequence.

Two confidence thresholds were tested: 0.25 and 0.40. The old provided `det.txt` detections were used as the baseline.

| Detector | Model | Conf | Precision | Recall | F1 | FPS |
|---|---|---:|---:|---:|---:|---:|
| Provided detections | det.txt | - | 0.7143 | 0.6407 | 0.6755 | - |
| YOLO | yolo11n.pt | 0.25 | 0.5787 | 0.8496 | 0.6885 | 24.90 |
| YOLO | yolo11n.pt | 0.40 | 0.7644 | 0.8134 | 0.7881 | 24.40 |

Increasing the confidence threshold from 0.25 to 0.40 reduced the number of false positives and improved F1 from 0.6885 to 0.7881. Compared with the provided detections, YOLO11n with confidence 0.40 improved F1 from 0.6755 to 0.7881 while keeping real-time performance above 5 FPS in Colab.

This configuration is therefore a strong first candidate for the modified DeepSORT detector.


## 9. YOLO11n detector evaluation on all sequences

YOLO11n with confidence threshold 0.40 and IoU threshold 0.60 was evaluated on all six sequences. The goal of this experiment was to test whether a modern detector can improve the detection quality before integrating it into DeepSORT.

| Sequence | Old F1 | YOLO11n F1 | YOLO11n Precision | YOLO11n Recall | FPS |
|---|---:|---:|---:|---:|---:|
| KITTI-17 | 0.7002 | 0.8692 | 0.9157 | 0.8272 | 35.14 |
| MOT16-09 | 0.6553 | 0.7370 | 0.8518 | 0.6494 | 62.57 |
| MOT16-11 | 0.6634 | 0.7276 | 0.9085 | 0.6068 | 62.89 |
| PETS09-S2L1 | 0.8121 | 0.9018 | 0.8929 | 0.9109 | 59.40 |
| TUD-Campus | 0.6755 | 0.7881 | 0.7644 | 0.8134 | 33.62 |
| TUD-Stadtmitte | 0.7294 | 0.8957 | 0.9698 | 0.8322 | 53.14 |
| **Average** | **0.7060** | **0.8199** | - | - | **51.96** |

YOLO11n improved F1 on every sequence. The average F1 increased from 0.7060 for the provided detections to 0.8199 for YOLO11n. The detector also satisfies the real-time requirement, because the FPS was higher than 5 on every sequence.

This confirms that YOLO11n with confidence 0.40 is a strong candidate detector for the modified DeepSORT pipeline.


## 10. Detector-only DeepSORT update on TUD-Campus

The first modified DeepSORT experiment replaced the provided detections with YOLO11n detections. The ReID descriptors were still pseudo descriptors at this stage, so this experiment evaluates the effect of detector replacement only.

| Tracker | Detector | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---|---:|---:|---:|---:|---:|
| Original DeepSORT baseline | provided det.txt | 35.183 | 42.198 | 29.556 | 52.174 | 44.290 |
| DeepSORT + YOLO11n | YOLO11n conf=0.40 | 46.566 | 44.274 | 49.137 | 65.385 | 53.482 |

Replacing the detector improved HOTA from 35.183 to 46.566 on TUD-Campus. IDF1 also increased from 52.174 to 65.385. This confirms that the YOLO11n detector improves not only detection F1 but also the final tracking quality.


## 11. Detector-only DeepSORT update on all sequences

The detector-only modified DeepSORT pipeline was evaluated on all six sequences. In this experiment, the original provided detections were replaced with YOLO11n detections using confidence threshold 0.40 and IoU threshold 0.60. The ReID descriptors were still pseudo descriptors, so this experiment isolates the effect of detector replacement.

| Sequence | Original HOTA | DeepSORT + YOLO11n HOTA | Original IDF1 | DeepSORT + YOLO11n IDF1 | Original MOTA | DeepSORT + YOLO11n MOTA |
|---|---:|---:|---:|---:|---:|---:|
| KITTI-17 | 38.351 | 53.283 | 63.594 | 77.457 | 45.396 | 64.194 |
| MOT16-09 | 25.344 | 33.007 | 31.999 | 37.369 | 31.869 | 37.508 |
| MOT16-11 | 26.957 | 42.320 | 32.202 | 45.346 | 40.363 | 49.127 |
| PETS09-S2L1 | 41.363 | 50.293 | 53.757 | 63.337 | 63.462 | 83.484 |
| TUD-Campus | 35.183 | 46.566 | 52.174 | 65.385 | 44.290 | 53.482 |
| TUD-Stadtmitte | 37.729 | 53.033 | 55.052 | 64.299 | 43.166 | 77.336 |
| **COMBINED** | **31.257** | **42.369** | **39.665** | **49.166** | **41.949** | **53.116** |

Replacing the detector with YOLO11n improved HOTA on every sequence. The combined HOTA increased from 31.257 to 42.369. IDF1 also improved from 39.665 to 49.166, and MOTA improved from 41.949 to 53.116.

This confirms that the modern detector improves the final tracking quality, not only the standalone detection F1 score. The next step is to replace the pseudo descriptors with real modern ReID embeddings.


## 12. First ReID experiment: OSNet x0.25 on TUD-Campus

The first modern ReID experiment used OSNet x0.25 from Torchreid. YOLO11n detections with confidence threshold 0.40 were used as input detections.

| Variant | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|---:|
| Original DeepSORT | 35.183 | 42.198 | 29.556 | 52.174 | 44.290 |
| DeepSORT + YOLO11n + pseudo features | 46.566 | 44.274 | 49.137 | 65.385 | 53.482 |
| DeepSORT + YOLO11n + OSNet x0.25 | 44.025 | 47.724 | 40.973 | 60.414 | 53.203 |

OSNet x0.25 improved over the original DeepSORT baseline, but it did not outperform the pseudo-feature detector-only variant on TUD-Campus. This is a useful negative result. A possible explanation is that the pseudo descriptors encode bbox position and size, which can accidentally help on short and simple sequences, while OSNet relies on appearance information and requires proper matching threshold tuning.

The next step is to tune DeepSORT parameters for OSNet, especially `max_cosine_distance`.


## 13. OSNet x0.25 parameter tuning on TUD-Campus

The `max_cosine_distance` parameter was tuned for the YOLO11n + OSNet x0.25 configuration on TUD-Campus.

| Model | max_cosine_distance | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|---:|---:|
| OSNet x0.25 | 0.20 | 44.025 | 47.724 | 40.973 | 60.414 | 53.203 |
| OSNet x0.25 | 0.30 | 44.197 | 47.869 | 41.160 | 60.606 | 53.482 |
| OSNet x0.25 | 0.40 | 44.197 | 47.869 | 41.160 | 60.606 | 53.482 |
| OSNet x0.25 | 0.50 | 44.098 | 47.659 | 41.155 | 60.440 | 52.925 |

The best result was obtained with `max_cosine_distance = 0.30` or `0.40`. Since both values gave the same HOTA, `0.30` was selected as the safer and stricter setting for further experiments.

This tuning slightly improved HOTA from 44.025 to 44.197 compared with the initial OSNet setting, but it still did not outperform the detector-only pseudo-feature variant on TUD-Campus.


## 14. DeepSORT with YOLO11n detector and OSNet x0.25 ReID on all sequences

The next experiment evaluated the full modified DeepSORT pipeline with both a modern detector and a modern ReID model.

Configuration:

- detector: YOLO11n
- detector confidence threshold: 0.40
- detector IoU threshold: 0.60
- ReID model: OSNet x0.25 from Torchreid
- ReID feature dimension: 512
- DeepSORT `max_cosine_distance`: 0.30
- `nn_budget`: 100

| Sequence | Original HOTA | YOLO11n + pseudo HOTA | YOLO11n + OSNet HOTA | Original IDF1 | YOLO11n + pseudo IDF1 | YOLO11n + OSNet IDF1 |
|---|---:|---:|---:|---:|---:|---:|
| KITTI-17 | 38.351 | 53.283 | 53.287 | 63.594 | 77.457 | 77.279 |
| MOT16-09 | 25.344 | 33.007 | 32.969 | 31.999 | 37.369 | 36.615 |
| MOT16-11 | 26.957 | 42.320 | 44.625 | 32.202 | 45.346 | 50.324 |
| PETS09-S2L1 | 41.363 | 50.293 | 52.779 | 53.757 | 63.337 | 68.561 |
| TUD-Campus | 35.183 | 46.566 | 44.197 | 52.174 | 65.385 | 60.606 |
| TUD-Stadtmitte | 37.729 | 53.033 | 53.554 | 55.052 | 64.299 | 72.183 |
| **COMBINED** | **31.257** | **42.369** | **43.816** | **39.665** | **49.166** | **52.271** |

The full YOLO11n + OSNet x0.25 configuration achieved the best combined HOTA so far: 43.816. This is higher than both the original DeepSORT baseline and the detector-only YOLO11n variant.

The most important improvement from OSNet appears in association-related metrics. Combined IDF1 increased from 49.166 with pseudo features to 52.271 with OSNet. This confirms that the modern ReID model improves identity consistency across the full dataset, even though it was not better on every individual sequence.

The result also shows why evaluation on all sequences is necessary: on TUD-Campus alone, pseudo features performed better, but on the full benchmark OSNet gave the stronger combined result.


## 15. Overlay videos

Two overlay videos were generated for qualitative comparison:

| Version | Sequence | Output video |
|---|---|---|
| Initial original DeepSORT baseline | TUD-Campus | `outputs/videos/TUD-Campus_original_deepsort.mp4` |
| Best current modified version | TUD-Campus | `outputs/videos/TUD-Campus_best_yolo11n_osnet_x0_25.mp4` |

The best current version uses:

- YOLO11n detector;
- confidence threshold 0.40;
- OSNet x0.25 ReID model;
- `max_cosine_distance = 0.30`;
- `nn_budget = 100`.

The videos are stored in the `outputs/videos/` directory. This directory is ignored by Git, because generated videos are large and can be recreated from the provided scripts.


## 16. Comparison of lightweight detector models on TUD-Campus

To satisfy the requirement of supporting multiple detection models, lightweight YOLO models were compared on TUD-Campus.

| Detector | Conf | Detections | FPS | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Provided det.txt | - | 322 | - | 0.7143 | 0.6407 | 0.6755 |
| YOLO11n | 0.40 | 382 | 33.62 | 0.7644 | 0.8134 | 0.7881 |
| YOLOv8n | 0.40 | 379 | 48.96 | 0.7863 | 0.8301 | 0.8076 |

YOLOv8n achieved the best F1 on TUD-Campus among the tested lightweight detectors. It also had higher FPS than YOLO11n in this experiment. This result shows that a smaller or older detector architecture can still be competitive under limited Colab resources.


## 17. Third detector model: YOLOv5nu

A third lightweight detector, YOLOv5nu, was evaluated on TUD-Campus to satisfy the requirement of supporting at least three detection models.

| Detector | Conf | Detections | FPS | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Provided det.txt | - | 322 | - | 0.7143 | 0.6407 | 0.6755 |
| YOLO11n | 0.40 | 382 | 33.62 | 0.7644 | 0.8134 | 0.7881 |
| YOLOv8n | 0.40 | 379 | 48.96 | 0.7863 | 0.8301 | 0.8076 |
| YOLOv5nu | 0.40 | 376 | 47.42 | 0.7713 | 0.8078 | 0.7891 |

All three modern detectors improved over the provided detections on TUD-Campus. YOLOv8n achieved the best F1, while YOLOv5nu was also competitive and remained real-time in Colab.

The implementation supports model selection through the `--model` argument of `src/detectors/yolo_detector.py`.


## 18. Second ReID model: OSNet x0.5

The second ReID model experiment used OSNet x0.5 from Torchreid. The same YOLO11n detections and the same DeepSORT parameters were used for fair comparison.

| Detector | ReID model | max_cosine_distance | HOTA | DetA | AssA | IDF1 | MOTA | ReID detections/sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | OSNet x0.25 | 0.30 | 44.197 | 47.869 | 41.160 | 60.606 | 53.482 | 224.80 |
| YOLO11n | OSNet x0.5 | 0.30 | 48.294 | 48.359 | 48.454 | 67.218 | 55.710 | 252.67 |

OSNet x0.5 outperformed OSNet x0.25 on TUD-Campus. HOTA increased from 44.197 to 48.294, and IDF1 increased from 60.606 to 67.218. This suggests that OSNet x0.5 provides stronger appearance descriptors for this sequence while still remaining fast enough for Colab experiments.


## 19. Third ReID model: MobileNetV2 x1.0

A third ReID model, MobileNetV2 x1.0, was evaluated on TUD-Campus. This model was selected because it is lightweight and belongs to a different architecture family than OSNet.

The experiment used:

- detector: YOLO11n;
- detector confidence threshold: 0.40;
- ReID model: MobileNetV2 x1.0;
- DeepSORT `max_cosine_distance`: 0.30;
- `nn_budget`: 100.

| Detector | ReID model | Feature dim | HOTA | DetA | AssA | IDF1 | MOTA | ReID detections/sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | OSNet x0.25 | 512 | 44.197 | 47.869 | 41.160 | 60.606 | 53.482 | 224.80 |
| YOLO11n | OSNet x0.5 | 512 | 48.294 | 48.359 | 48.454 | 67.218 | 55.710 | 252.67 |
| YOLO11n | MobileNetV2 x1.0 | 1280 | 50.421 | 48.135 | 52.991 | 72.099 | 56.546 | 462.50 |

MobileNetV2 x1.0 achieved the best result on TUD-Campus among the three tested ReID models. HOTA increased to 50.421 and IDF1 increased to 72.099.

However, Torchreid produced a warning that ImageNet pretrained weights for MobileNetV2 need to be downloaded manually. Therefore, this result is treated as an experimental candidate and should be interpreted carefully. For the most reproducible full benchmark result, OSNet x0.25 remains the already tested all-sequence ReID baseline, while OSNet x0.5 and MobileNetV2 are promising candidates for further evaluation.


## 20. Full benchmark with YOLO11n and OSNet x0.5

After the TUD-Campus ReID comparison, OSNet x0.5 was evaluated on all six sequences.

Configuration:

- detector: YOLO11n;
- detector confidence threshold: 0.40;
- detector IoU threshold: 0.60;
- ReID model: OSNet x0.5;
- ReID feature dimension: 512;
- DeepSORT `max_cosine_distance`: 0.30;
- `nn_budget`: 100.

| Sequence | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|---:|
| KITTI-17 | 53.418 | 53.696 | 53.159 | 77.569 | 64.450 |
| MOT16-09 | 33.100 | 32.986 | 33.403 | 38.168 | 37.701 |
| MOT16-11 | 46.015 | 45.851 | 46.510 | 53.131 | 49.375 |
| PETS09-S2L1 | 55.085 | 65.709 | 46.261 | 69.370 | 83.763 |
| TUD-Campus | 48.294 | 48.359 | 48.454 | 67.218 | 55.710 |
| TUD-Stadtmitte | 55.625 | 61.397 | 50.607 | 71.034 | 77.422 |
| **COMBINED** | **45.111** | **46.087** | **44.574** | **54.055** | **53.371** |

OSNet x0.5 achieved the best combined HOTA among all tested configurations. The combined HOTA increased from 31.257 for the original DeepSORT baseline to 45.111. IDF1 also increased from 39.665 to 54.055.

Compared with OSNet x0.25, OSNet x0.5 improved the combined HOTA from 43.816 to 45.111 and the combined IDF1 from 52.271 to 54.055. Therefore, the current best configuration is YOLO11n + OSNet x0.5 with `max_cosine_distance = 0.30`.


## 21. Candidate combination: YOLOv8n with OSNet x0.5 on TUD-Campus

Since YOLOv8n achieved better standalone detection F1 than YOLO11n on TUD-Campus, an additional combined tracking experiment was performed with YOLOv8n and OSNet x0.5.

| Detector | ReID model | HOTA | DetA | AssA | IDF1 | MOTA |
|---|---|---:|---:|---:|---:|---:|
| YOLO11n | OSNet x0.5 | 48.294 | 48.359 | 48.454 | 67.218 | 55.710 |
| YOLOv8n | OSNet x0.5 | 48.450 | 50.721 | 46.365 | 64.666 | 57.939 |

YOLOv8n + OSNet x0.5 slightly improved HOTA and MOTA on TUD-Campus, but IDF1 decreased compared with YOLO11n + OSNet x0.5. Since the improvement in HOTA was small and the identity metric became worse, this combination was saved as a candidate experiment but was not selected as the final full-benchmark configuration.

The current best fully evaluated configuration remains YOLO11n + OSNet x0.5, because it was evaluated on all six sequences and achieved the best combined HOTA so far.


## 22. Final best overlay update

After evaluating OSNet x0.5 on all six sequences, the best current configuration became:

- detector: YOLO11n;
- ReID model: OSNet x0.5;
- detector confidence threshold: 0.40;
- detector IoU threshold: 0.60;
- DeepSORT `max_cosine_distance`: 0.30;
- `nn_budget`: 100.

The best overlay video was therefore regenerated for this final configuration:

| Version | Sequence | Output video |
|---|---|---|
| Initial original DeepSORT baseline | TUD-Campus | `outputs/videos/TUD-Campus_original_deepsort.mp4` |
| Final best modified version | TUD-Campus | `outputs/videos/TUD-Campus_best_yolo11n_osnet_x0_5.mp4` |

The final best configuration achieved combined HOTA 45.111, compared with 31.257 for the original DeepSORT baseline.


## 23. Final conclusion

The project successfully modified the original DeepSORT implementation by replacing the original detection input with modern YOLO-based person detectors and replacing pseudo appearance descriptors with modern ReID embeddings.

The original DeepSORT baseline achieved a combined HOTA of 31.257 on the six required sequences. The best final configuration achieved a combined HOTA of 45.111.

Final selected configuration:

- detector: YOLO11n;
- detector confidence threshold: 0.40;
- detector IoU threshold: 0.60;
- ReID model: OSNet x0.5;
- ReID feature dimension: 512;
- DeepSORT `max_cosine_distance`: 0.30;
- `nn_budget`: 100.

The final configuration improved HOTA on every required sequence:

| Sequence | Original HOTA | Final HOTA |
|---|---:|---:|
| KITTI-17 | 38.351 | 53.418 |
| MOT16-09 | 25.344 | 33.100 |
| MOT16-11 | 26.957 | 46.015 |
| PETS09-S2L1 | 41.363 | 55.085 |
| TUD-Campus | 35.183 | 48.294 |
| TUD-Stadtmitte | 37.729 | 55.625 |
| **COMBINED** | **31.257** | **45.111** |

The detector experiments showed that modern YOLO detectors improve standalone detection F1 compared with the provided detections. Three detector models were tested: YOLO11n, YOLOv8n and YOLOv5nu. Although YOLOv8n achieved the best F1 on TUD-Campus, YOLO11n was selected for the final full benchmark because it had already been evaluated on all sequences and produced a stable improvement.

The ReID experiments showed that OSNet x0.5 provided better identity consistency than OSNet x0.25 on the full benchmark. The combined IDF1 increased from 39.665 for the original baseline to 54.055 for the final configuration. MobileNetV2 x1.0 was also tested as a lightweight alternative, but it was treated as an experimental candidate because Torchreid warned that its pretrained weights require manual download.

The project also includes overlay videos for qualitative comparison between the original DeepSORT baseline and the final best modified configuration.

The additional standalone body ReID task and segmentation model support were not completed in the current version. Therefore, the current work focuses on the main DeepSORT modernization task: detector replacement, ReID replacement, parameter tuning, full MOT-style evaluation, report preparation, Colab reproducibility and Git history preservation.

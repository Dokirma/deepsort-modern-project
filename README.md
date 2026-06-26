# DeepSORT Modernization Project

## Project goal

This project modifies the original DeepSORT implementation by adding modern person detection and ReID models, evaluating them on MOT-style videos, and comparing the modified system with the original DeepSORT baseline.

The original DeepSORT repository was used as the basis, and the original commit history was preserved.

## Dataset

The required MOT-style videos are provided through Google Drive instead of the MOTChallenge website.

Dataset archive link:

https://drive.google.com/file/d/1ujjjDlQZ6eEfdfWqJx-L_pgbJkSqRkU8/view?usp=sharing

The archive contains six required sequences:

- KITTI-17
- MOT16-09
- MOT16-11
- PETS09-S2L1
- TUD-Campus
- TUD-Stadtmitte

## Main results

| Method | Detector | ReID | Combined HOTA | Combined IDF1 | Combined MOTA |
|---|---|---|---:|---:|---:|
| Original DeepSORT | provided det.txt | original / pseudo features | 31.257 | 39.665 | 41.949 |
| DeepSORT + YOLO11n | YOLO11n | pseudo features | 42.369 | 49.166 | 53.116 |
| DeepSORT + YOLO11n + OSNet x0.25 | YOLO11n | OSNet x0.25 | 43.816 | 52.271 | 53.414 |
| DeepSORT + YOLO11n + OSNet x0.5 | YOLO11n | OSNet x0.5 | 45.111 | 54.055 | 53.371 |

Final selected configuration:

- detector: YOLO11n
- detector confidence threshold: 0.40
- detector IoU threshold: 0.60
- ReID model: OSNet x0.5
- DeepSORT `max_cosine_distance`: 0.30
- DeepSORT `nn_budget`: 100

The final configuration improves HOTA on every required sequence compared with the original DeepSORT baseline.

## Supported models

Detection models:

- YOLO11n
- YOLOv8n
- YOLOv5nu

ReID models:

- OSNet x0.25
- OSNet x0.5
- MobileNetV2 x1.0

## Colab execution instructions

The project is designed to run in a clean Colab runtime. 

A solution notebook is provided in:

```text

https://github.com/Dokirma/deepsort-modern-project/blob/main/notebooks/DeepSORT_solution_part.ipynb

```

A submission notebook is provided in:

```text
(https://github.com/Dokirma/deepsort-modern-project/blob/main/notebooks/DeepSORT_Project_submission_checked.ipynb)

```

### 1. Clone repository

```python
%cd /content
!rm -rf /content/deepsort-modern-project
!git clone https://github.com/Dokirma/deepsort-modern-project.git /content/deepsort-modern-project
%cd /content/deepsort-modern-project
```

### 2. Install dependencies

```python
%cd /content/deepsort-modern-project
!pip install -q -r requirements.txt
!pip install -q ultralytics torchreid gdown
```

### 3. Download and extract dataset

```python
%cd /content/deepsort-modern-project

!python src/data/download_data.py \
  --file-id 1ujjjDlQZ6eEfdfWqJx-L_pgbJkSqRkU8 \
  --output data/raw/dlcv-final-project-videos.tar.gz \
  --extract-to data/mot
```

Expected dataset folder:

```text
data/mot/videos/
```

### 4. Run original DeepSORT baseline

```python
%cd /content/deepsort-modern-project
!python src/tracking/run_original_deepsort_all.py
```

### 5. Run YOLO11n detections

```python
%cd /content/deepsort-modern-project

!python src/detectors/run_yolo_all.py \
  --model yolo11n.pt \
  --conf 0.40 \
  --iou 0.60 \
  --imgsz 640 \
  --name yolo11n_conf040
```

### 6. Extract OSNet x0.5 ReID features

```python
%cd /content/deepsort-modern-project

!python src/reid/run_osnet_features_all.py \
  --det-dir outputs/detections/yolo11n_conf040 \
  --output-dir outputs/detections_reid/yolo11n_osnet_x0_5 \
  --model osnet_x0_5 \
  --batch-size 32
```

### 7. Run final best DeepSORT version

```python
%cd /content/deepsort-modern-project

!python src/tracking/run_deepsort_from_reid_all.py \
  --reid-dir outputs/detections_reid/yolo11n_osnet_x0_5 \
  --tracker-name deepsort_yolo11n_osnet_x0_5_cos030 \
  --max-cosine-distance 0.30 \
  --nn-budget 100
```

### 8. Generate overlay videos

The final notebook generates overlay videos for all six sequences:

- 6 videos for the original DeepSORT baseline
- 6 videos for the final best modified DeepSORT version

The generated videos are saved to:

```text
outputs/videos/
outputs/videos_web/
```

The `outputs/` directory is ignored by Git because it contains generated files.

## Saved artifacts

Small generated artifacts such as tracking `.txt` files and detector `.txt` files can be saved in the `artifacts/` directory.

Large generated files are not committed to GitHub. A local or Drive backup archive may be provided separately:

```text
final_outputs_with_videos_*.zip
```

The archive with videos contains:

- 12 tracking `.txt` files
- 6 detection `.txt` files
- 6 ReID `.npy` files
- 12 web-compatible overlay `.mp4` videos
- report files

## Report

Detailed report:

```text
report/report.md
```

Experiment tables:

```text
report/*.csv
```

## Notes

- Generated outputs, model weights, TrackEval files, videos and final ZIP archives are not committed to GitHub.
- The notebook regenerates outputs in a clean Colab runtime.
- The final best configuration is reproducible using the commands above.

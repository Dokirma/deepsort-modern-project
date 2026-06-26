# DeepSORT Modernization Project

## Project goal

This project modifies the original DeepSORT implementation by adding modern person detection and ReID models, evaluating them on MOT-style videos, and comparing the modified system with the original DeepSORT baseline.

The original DeepSORT repository was used as the basis, and its commit history was preserved during the merge.

## Dataset

The required MOT-style videos are provided through Google Drive instead of the MOTChallenge website.

Google Drive dataset link:

https://drive.google.com/file/d/1ujjjDlQZ6eEfdfWqJx-L_pgbJkSqRkU8/view?usp=sharing

The archive contains six sequences:

- KITTI-17
- MOT16-09
- MOT16-11
- PETS09-S2L1
- TUD-Campus
- TUD-Stadtmitte

Each sequence contains:

- img1/
- det/det.txt
- gt/gt.txt
- seqinfo.ini

## Main results

| Method | Detector | ReID | Combined HOTA | Combined IDF1 | Combined MOTA |
|---|---|---|---:|---:|---:|
| Original DeepSORT | provided det.txt | pseudo/original-style features | 31.257 | 39.665 | 41.949 |
| DeepSORT + YOLO11n | YOLO11n | pseudo features | 42.369 | 49.166 | 53.116 |
| DeepSORT + YOLO11n + OSNet x0.25 | YOLO11n | OSNet x0.25 | 43.816 | 52.271 | 53.414 |
| DeepSORT + YOLO11n + OSNet x0.5 | YOLO11n | OSNet x0.5 | 45.111 | 54.055 | 53.371 |

Best final configuration:

- detector: YOLO11n
- confidence threshold: 0.40
- detector IoU threshold: 0.60
- ReID model: OSNet x0.5
- DeepSORT max_cosine_distance: 0.30
- DeepSORT nn_budget: 100

This configuration improves HOTA on every required sequence compared with the original DeepSORT baseline.

## Supported detection models

The detector can be selected with the --model argument in src/detectors/yolo_detector.py.

Tested detection models:

- YOLO11n
- YOLOv8n
- YOLOv5nu

## Supported ReID models

The ReID model can be selected with the --model argument in src/reid/osnet_extractor.py.

Tested ReID models:

- OSNet x0.25
- OSNet x0.5
- MobileNetV2 x1.0

## Colab execution instructions

### 1. Mount Google Drive

Run in Colab:

    from google.colab import drive
    drive.mount('/content/drive')

### 2. Go to the project folder

    %cd /content/drive/MyDrive/deepsort-modern-project

### 3. Install dependencies

    !pip install -q -r requirements.txt
    !pip install -q ultralytics torchreid

### 4. Download and extract dataset

    !python src/data/download_data.py

After this step, the dataset should be located in:

    data/mot/videos/

### 5. Download TrackEval

TrackEval is not committed to the repository. Download it in Colab:

    !mkdir -p external
    !test -d external/TrackEval || git clone https://github.com/JonathonLuiten/TrackEval.git external/TrackEval

### 6. Run original DeepSORT baseline

    !python src/tracking/run_original_deepsort_all.py

Evaluate original baseline:

    !python src/evaluation/prepare_trackeval_tracker.py --tracker-name original_deepsort --tracks-dir outputs/tracks/original_deepsort --seqmap-file DLCV-train-original.txt --sequences KITTI-17 MOT16-09 MOT16-11 PETS09-S2L1 TUD-Campus TUD-Stadtmitte

    !python src/evaluation/run_trackeval.py --tracker-name original_deepsort --seqmap-file DLCV-train-original.txt

### 7. Run YOLO11n detections

    !python src/detectors/run_yolo_all.py --model yolo11n.pt --conf 0.40 --iou 0.60 --imgsz 640 --name yolo11n_conf040

### 8. Extract OSNet x0.5 ReID features

    !python src/reid/run_osnet_features_all.py --det-dir outputs/detections/yolo11n_conf040 --output-dir outputs/detections_reid/yolo11n_osnet_x0_5 --model osnet_x0_5 --batch-size 32

### 9. Run final best DeepSORT version

    !python src/tracking/run_deepsort_from_reid_all.py --reid-dir outputs/detections_reid/yolo11n_osnet_x0_5 --tracker-name deepsort_yolo11n_osnet_x0_5_cos030 --max-cosine-distance 0.30 --nn-budget 100

### 10. Evaluate final best version

    !python src/evaluation/prepare_trackeval_tracker.py --tracker-name deepsort_yolo11n_osnet_x0_5_cos030 --tracks-dir outputs/tracks/deepsort_yolo11n_osnet_x0_5_cos030 --seqmap-file DLCV-train-best.txt --sequences KITTI-17 MOT16-09 MOT16-11 PETS09-S2L1 TUD-Campus TUD-Stadtmitte

    !python src/evaluation/run_trackeval.py --tracker-name deepsort_yolo11n_osnet_x0_5_cos030 --seqmap-file DLCV-train-best.txt

### 11. Generate overlay videos

Original DeepSORT overlay:

    !python src/visualization/make_overlay.py --sequence-dir data/mot/videos/TUD-Campus --tracks-file outputs/tracks/original_deepsort/TUD-Campus.txt --output-video outputs/videos/TUD-Campus_original_deepsort.mp4

Best modified version overlay:

    !python src/visualization/make_overlay.py --sequence-dir data/mot/videos/TUD-Campus --tracks-file outputs/tracks/deepsort_yolo11n_osnet_x0_5_cos030/TUD-Campus.txt --output-video outputs/videos/TUD-Campus_best_yolo11n_osnet_x0_5.mp4

Generated videos are stored in:

    outputs/videos/

The outputs/ directory is ignored by Git because it contains generated files.

## Report

Detailed report:

    report/report.md

Experiment tables:

    report/*.csv

## Notes

- The project is designed for Google Colab.
- Generated outputs, model weights, TrackEval and videos are not committed to Git.
- All important scripts and numerical results are committed.
- The final best configuration is reproducible using the commands above.

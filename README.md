# DeepSORT Modernization Project

## Project goal

This project modifies the original DeepSORT implementation by adding modern person detection and ReID models, evaluating them on MOT-style videos, and comparing the modified system with the original DeepSORT baseline.

The original DeepSORT repository was used as the basis, and its commit history was preserved during the merge.

## Dataset

The required MOT-style videos are provided through Google Drive instead of the MOTChallenge website.

Dataset archive:

`dlcv-final-project-videos.tar.gz`

Google Drive source:

https://drive.google.com/file/d/1ujjjDlQZ6eEfdfWqJx-L_pgbJkSqRkU8/view?usp=sharing

The archive contains six sequences:

- KITTI-17
- MOT16-09
- MOT16-11
- PETS09-S2L1
- TUD-Campus
- TUD-Stadtmitte

Each sequence follows MOT-style structure:

```text
sequence/
  img1/
  det/det.txt
  gt/gt.txt
  seqinfo.ini

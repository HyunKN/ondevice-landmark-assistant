# Training and experiment design

This directory contains the actual model-training source used in the project. It covers data preparation, baseline classification, Sprint 2 multi-task training, evaluation, hard-negative review and mobile export.

## Code map

| File | Role |
|---|---|
| [`src/landmark_candidate/split_data.py`](src/landmark_candidate/split_data.py) | Builds the locked test split and five folds with a fixed seed |
| [`src/landmark_candidate/train.py`](src/landmark_candidate/train.py) | Baseline classification training and checkpoint evaluation |
| [`src/landmark_candidate/multitask_model.py`](src/landmark_candidate/multitask_model.py) | MobileCLIP wrapper, partial unfreezing and LoRA insertion |
| [`src/landmark_candidate/losses.py`](src/landmark_candidate/losses.py) | CE/ArcFace, multi-positive image-text contrastive and hard-negative losses |
| [`src/landmark_candidate/train_multitask.py`](src/landmark_candidate/train_multitask.py) | DDP training, image/text evaluation, prototype construction and checkpoint selection |
| [`scripts/mine_hard_negative_candidates.py`](scripts/mine_hard_negative_candidates.py) | Ranks confusion pairs and low-margin examples for review |
| [`scripts/export_mobile_onnx.py`](scripts/export_mobile_onnx.py) | Exports separate image/text FP32 and mixed-FP16 ONNX artifacts |
| [`scripts/validate_mobile_artifacts.py`](scripts/validate_mobile_artifacts.py) | Checks real-image parity across exported precision folders |

## Why the training design changed

Sprint 1 prioritized a working demo and on-device feasibility. When the dataset expanded, the training configuration added warmup plus cosine decay, a less aggressive crop, reduced RandAug magnitude, square-root class weighting, and per-class/confusion reporting.

The next comparison needed a stricter reset. An earlier S3 configuration named `server_full` still used partial image/text unfreeze ratios, so it could not isolate backbone choice from fine-tuning strategy. Sprint 2 therefore controlled the two dimensions independently:

```text
MobileCLIP2-S3 or MobileCLIP2-S4
    × full CE / partial CE / partial ArcFace / LoRA CE
    × five folds
    = 40 main runs
```

The multi-task objective combines classification, image-text contrastive learning and a hard-negative logit margin. Candidate selection uses mean validation metrics across folds. The locked test split is report-only and is not used to choose a checkpoint.

## Configs

[`configs/main_matrix/`](configs/main_matrix/) contains the eight fold-0 configs used to define the controlled matrix. Only the model/data/loss/training/optimizer/evaluation candidate config is retained; W&B runtime and raw history are omitted.

## Supporting records

- [Training config v2 decision](https://landmark-assistant-sprint1.vercel.app/decisions/ADR-0005-training-config-v2.html)
- [Sprint 2 experiment matrix](https://landmark-assistant-sprint1.vercel.app/experiments/sprint2-paper-experiment-matrix-2026-06-12.html)
- [Training design sequence](https://landmark-assistant-sprint1.vercel.app/learning/model-training-design-sequence.html)

## Reproduction boundary

The source and configs are real project files, but the dataset, split manifests, checkpoints and generated ONNX artifacts are not included. Paths and commands must therefore be supplied for an authorized local dataset before training or export can run.

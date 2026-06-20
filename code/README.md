# Code walkthrough

This is a curated snapshot of the project code, organized around the path from experiment to on-device integration.

## 1. Train and compare models

**Problem.** Compare backbone and fine-tuning choices without changing the data split or evaluation rules between runs.

**Actual code.** [`training/`](training/) contains the dataset/split pipeline, baseline and multi-task trainers, loss functions, hard-negative mining, eight sanitized experiment configs, and mobile ONNX export/validation scripts.

**Evidence.** The main matrix covers MobileCLIP2-S3/S4 with full CE, partial CE, partial ArcFace and LoRA CE across five folds. Reviewed aggregate metrics remain in [`../data/metrics.json`](../data/metrics.json).

**Limitations.** Training data, checkpoints and raw run exports are not public, so this is not a one-command reproduction package.

## 2. Validate the experience with the Sprint 1 prototype

**Problem.** Make image recognition, natural-language search and confidence behavior inspectable before the final mobile app existed.

**Actual code.** [`sprint1_prototype/`](sprint1_prototype/) is the tracked Streamlit prototype, including PyTorch/ONNX inference adapters, asset loading, fusion search and confidence policy.

**Evidence.** The prototype is the design predecessor of the final Android experience. The archived project site records its decisions and experiments.

**Limitations.** Model and image assets are omitted; the source can be inspected, but the complete UI cannot run from this snapshot alone.

## 3. Hand off a validated model contract

**Problem.** Keep model artifacts, preprocessing, class order, thresholds and app expectations consistent across Python and Android.

**Actual code.** [`model_integration/`](model_integration/) contains real contract metadata, contract validation, semantic text-artifact generation and the exact Android asset-cache patch.

**Evidence.** [`model_integration/ANDROID_INTEGRATION.md`](model_integration/ANDROID_INTEGRATION.md) links the final app's team implementation and the later integration commits without copying team-owned Flutter/Android files into this portfolio.

**Limitations.** ONNX binaries and the complete Flutter app are intentionally excluded.

See [`SOURCES.md`](SOURCES.md) for provenance and [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) for the role boundary.

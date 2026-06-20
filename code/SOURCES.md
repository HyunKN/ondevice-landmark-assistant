# Source provenance

This directory is a curated snapshot of code used in the project. It is not a newly written reconstruction.

## Training

- Original repository: [HyunKN/landmark-assistant-model-ver2](https://github.com/HyunKN/landmark-assistant-model-ver2)
- Tracked source commit: [`9e0824608dbd4185ff92574f3a1172a141b72e10`](https://github.com/HyunKN/landmark-assistant-model-ver2/commit/9e0824608dbd4185ff92574f3a1172a141b72e10)
- `src/landmark_candidate/` and `scripts/mine_hard_negative_candidates.py` were extracted from that commit.
- `scripts/export_mobile_onnx.py` is a later local project snapshot, SHA-256 `8B4B1333A151C75D5B6601813C858EC2B7701677FCAFF7941F19A6E0C4567300`.
- `scripts/validate_mobile_artifacts.py` is a later local project snapshot, SHA-256 `0A224CAA6A955BA5731EFC5BCCCD99059301769DCE009B82BCB5E621CD339894`.
- `configs/main_matrix/` contains only the `candidate` section from eight fold-0 W&B config exports. Runtime/tracker fields and raw run history were removed. The source run names are recorded in `SOURCE_MAP.json`.

## Sprint 1 prototype

- Original repository: [HyunKN/landmark-demo-app](https://github.com/HyunKN/landmark-demo-app)
- Snapshot commit: [`cc7abb9f2d6cb0c1337860f1b8cd533378d851ee`](https://github.com/HyunKN/landmark-demo-app/tree/cc7abb9f2d6cb0c1337860f1b8cd533378d851ee)
- Every file in `sprint1_prototype/`, except this portfolio README, was extracted from that tracked commit.

## Model integration

- Team app repository: [lpcvc-2026-CNU/App](https://github.com/lpcvc-2026-CNU/App)
- Reference state: [`2e4349b2a6b960dfff7ab812c5c1b39f5e27c148`](https://github.com/lpcvc-2026-CNU/App/commit/2e4349b2a6b960dfff7ab812c5c1b39f5e27c148)
- Contract JSON files were copied from the FP16 artifact metadata at that state. `manifest.example.json` removes checkpoint identifiers and explicitly records that binaries are omitted.
- `scripts/check_model_contract.py` source SHA-256: `24519C54263CF5014A892CA7F88E1D1AD276BC1EB17251397AD09209DD4CC50D`.
- `scripts/generate_semantic_text_artifacts.py` source SHA-256: `A3D2EE0D9CD2B008700D82B3FD1590FB626BB357847C69E26A02372D99CAFE44`.
- `patches/2e4349b-android-asset-cache-fix.patch` is the exact `MainActivity.kt` diff from commit `2e4349b`.

## Deliberately omitted

Datasets, checkpoints, ONNX/external-data binaries, tokenizer/index bundles, raw W&B exports, logs, private paths, and unrelated final-app features are not included.

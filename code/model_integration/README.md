# Model integration contract

This directory shows the actual handoff boundary between the trained model and the final app. It keeps class order, preprocessing, model shape, confidence thresholds and semantic-search behavior reviewable without publishing the model binaries.

## Contract contents

| File | Contract |
|---|---|
| [`contracts/classes.json`](contracts/classes.json) | Ordered 23-class output mapping |
| [`contracts/preprocessing.json`](contracts/preprocessing.json) | 224×224 input, normalization, layout and dtype |
| [`contracts/manifest.example.json`](contracts/manifest.example.json) | Model identity, 512-dimensional embedding, encoder I/O and ONNX versions |
| [`contracts/confidence_policy.json`](contracts/confidence_policy.json) | Image score, rejection and margin thresholds |
| [`contracts/text_search_policy.json`](contracts/text_search_policy.json) | Semantic/keyword fusion and ambiguity thresholds |

## Actual integration source

- [`scripts/check_model_contract.py`](scripts/check_model_contract.py) checks required files, class alignment, dimensions, preprocessing and non-portable paths.
- [`scripts/generate_semantic_text_artifacts.py`](scripts/generate_semantic_text_artifacts.py) builds multilingual text rows, embeddings, tokenizer fixtures and regression reports from the exported text encoder.
- [`patches/2e4349b-android-asset-cache-fix.patch`](patches/2e4349b-android-asset-cache-fix.patch) is the exact Android change that fixed cache invalidation for compressed large assets where `openFd` could not provide a length.

## Handoff flow

```text
training checkpoint
  → image/text ONNX export
  → metadata + class/prototype/text artifacts
  → Python contract validation
  → Flutter asset declaration
  → Android asset cache
  → ONNX Runtime inference
  → confidence or semantic-search policy
```

[`ANDROID_INTEGRATION.md`](ANDROID_INTEGRATION.md) connects this contract to the final team app and separates team implementation from later integration changes.

## Validation boundary

`tests/test_contract_metadata.py` validates the public metadata-only snapshot. It does not claim to replace the original full integration test, which required the Flutter app and complete ONNX artifact bundle.

The filenames in `manifest.example.json` document the serving contract; the referenced binaries are not included.

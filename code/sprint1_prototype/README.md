# Sprint 1 prototype

This is the actual tracked source snapshot used to demonstrate the project before the final Android app. It exposed image recognition, Korean/English text search, Top-3 results and confidence states through a Streamlit UI.

## Runtime flow

```text
run.py
  → app.py
  → data.py loads metadata, prototypes and text indexes
  → inference.py encodes image or text input
  → search.py fuses scores and applies the confidence policy
  → app.py renders the result and detail view
```

Key files:

- [`src/landmark_demo/inference.py`](src/landmark_demo/inference.py): PyTorch and ONNX image/text adapters plus input-quality checks
- [`src/landmark_demo/search.py`](src/landmark_demo/search.py): cosine scoring, keyword fusion, confidence states and Top-3 results
- [`src/landmark_demo/data.py`](src/landmark_demo/data.py): prototype, text-index and landmark metadata loading
- [`src/landmark_demo/app.py`](src/landmark_demo/app.py): prototype interaction and result presentation
- [`tests/test_search_policy.py`](tests/test_search_policy.py): confidence/search regression cases

`config.toml`, `config.onnx.toml` and `config.int8.toml` select the checkpoint or exported artifact path used by the same prototype flow.

## Relationship to the final app

The landmark-recognition and search experience was designed and tested here first. The final Flutter/Android app was then implemented collaboratively from this prototype and its handoff requirements. This snapshot is not presented as the final mobile implementation.

## Run boundary

The original prototype requires a checkpoint or ONNX bundle plus generated prototype/text/image assets. Those files are intentionally omitted from this portfolio, so the full Streamlit app does not run from this directory alone. The pure search-policy tests can still be inspected and run when their Python dependencies are available.

- [Original source repository](https://github.com/HyunKN/landmark-demo-app/tree/cc7abb9f2d6cb0c1337860f1b8cd533378d851ee)
- [Published project records](https://landmark-assistant-sprint1.vercel.app/)

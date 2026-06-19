# On-device Landmark Assistant README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a visual, factual GitHub README for the completed On-device Landmark Assistant project using verified metrics, real app captures, and GitHub-compatible diagrams.

**Architecture:** Keep the portfolio repository independent from the team source repository. Store only compact evidence data, generated SVG figures, selected app captures, and the README; link to the team source and public documentation rather than copying code or model binaries. Generate charts reproducibly from a small JSON evidence file and validate public content for forbidden local paths, unsupported claims, and missing assets.

**Tech Stack:** GitHub Markdown, Mermaid, SVG, Python 3 standard library, Android `adb` capture, Git

---

## File structure

- Create: `.gitignore` — excludes brainstorming state, model binaries, local captures, and secrets.
- Create: `README.md` — visual-first project portfolio front page.
- Create: `data/metrics.json` — reviewed source values used in charts and README checks.
- Create: `scripts/generate_visuals.py` — generates the hero and evidence SVG files from reviewed data.
- Create: `tests/test_portfolio_assets.py` — validates metrics, generated assets, README links, wording, and privacy constraints.
- Create: `assets/hero.svg` — restrained title banner.
- Create: `assets/experiment-comparison.svg` — eight-configuration validation chart.
- Create: `assets/deployment-flow.svg` — image/text ONNX to Android flow.
- Create: `assets/npu-evidence.svg` — latency-versus-accuracy caveat visual.
- Create: `assets/app-home.png` — actual home/input screen capture.
- Create: `assets/app-image-result.png` — actual Top-3 image result capture.
- Create: `assets/app-text-search.png` — actual natural-language search capture.

### Task 1: Repository safety and evidence data

**Files:**
- Create: `.gitignore`
- Create: `data/metrics.json`
- Test: `tests/test_portfolio_assets.py`

- [ ] **Step 1: Add repository exclusions**

Create `.gitignore` with:

```gitignore
.superpowers/
*.onnx
*.onnx.data
*.pt
*.pth
*.ckpt
.env
*service-account*.json
google-services.json
captures-raw/
__pycache__/
```

- [ ] **Step 2: Add reviewed metrics**

Create `data/metrics.json`:

```json
{
  "class_count": 23,
  "main_run_count": 40,
  "fold_count": 5,
  "configurations": [
    {"name": "S4 full CE", "val_top1": 99.05, "test_top1": 98.67, "macro_f1": 97.11, "low_margin": 6},
    {"name": "S3 full CE", "val_top1": 98.85, "test_top1": 98.73, "macro_f1": 97.41, "low_margin": 4},
    {"name": "S3 partial CE", "val_top1": 98.74, "test_top1": 98.63, "macro_f1": 97.16, "low_margin": 8},
    {"name": "S3 partial ArcFace", "val_top1": 98.55, "test_top1": 98.56, "macro_f1": 97.67, "low_margin": 8},
    {"name": "S4 partial ArcFace", "val_top1": 98.17, "test_top1": 98.09, "macro_f1": 96.41, "low_margin": 18},
    {"name": "S4 partial CE", "val_top1": 97.84, "test_top1": 97.61, "macro_f1": 94.70, "low_margin": 34},
    {"name": "S4 LoRA", "val_top1": 96.67, "test_top1": 96.59, "macro_f1": 93.21, "low_margin": 49},
    {"name": "S3 LoRA", "val_top1": 95.81, "test_top1": 95.92, "macro_f1": 91.36, "low_margin": 68}
  ],
  "dynamic_int8": {"fp32_cosine_mean": 0.99941, "cpu_warm_median_ms": 314},
  "npu_latency_ms": {"snapdragon_8_gen_2": 517.91, "snapdragon_8_gen_3": 389.67, "snapdragon_8_elite": 374.10},
  "npu_accuracy_status": "collapsed",
  "contract_tests": {"python": 10, "flutter": 4, "verified_on": "2026-06-20"}
}
```

- [ ] **Step 3: Write initial validation tests**

Create `tests/test_portfolio_assets.py` with `unittest` tests that load `data/metrics.json`, assert 8 configurations and 40 runs, assert the primary value `99.05`, reject model binaries, reject absolute Windows paths and credential terms in public files, and require all README image paths to exist.

- [ ] **Step 4: Run the tests and confirm the expected initial failure**

Run:

```powershell
python -m unittest tests.test_portfolio_assets -v
```

Expected: metrics tests pass; README and generated-asset checks fail because those files do not exist yet.

- [ ] **Step 5: Commit safety and evidence data**

```powershell
git add .gitignore data/metrics.json tests/test_portfolio_assets.py
git commit -m "chore: add reviewed portfolio evidence"
```

### Task 2: Reproducible visual assets

**Files:**
- Create: `scripts/generate_visuals.py`
- Create: `assets/hero.svg`
- Create: `assets/experiment-comparison.svg`
- Create: `assets/deployment-flow.svg`
- Create: `assets/npu-evidence.svg`
- Modify: `tests/test_portfolio_assets.py`

- [ ] **Step 1: Implement an SVG generator with no external dependencies**

Create `scripts/generate_visuals.py`. It must:

1. Load `data/metrics.json`.
2. Escape all displayed strings with `html.escape`.
3. Generate a 1200x360 navy/teal `hero.svg` containing only the project title, `MobileCLIP2 · ONNX Runtime · Flutter`, and the factual subtitle `Model experiments to Android on-device inference`.
4. Generate a horizontal bar chart of all eight `val_top1` values with a visible 94-100% axis and exact numeric labels.
5. Generate a deployment flow with these nodes: `Image/Text Input`, `MobileCLIP2 Encoders`, `Prototype/Text Index`, `Confidence Policy`, `Flutter UI`, `Local/Server Logs`.
6. Generate an NPU evidence panel listing the three measured latencies and a separate amber warning box stating `Quantized accuracy collapsed — latency is feasibility evidence only`.
7. Write UTF-8 SVG files under `assets/`.

- [ ] **Step 2: Generate the SVG files**

Run:

```powershell
python scripts/generate_visuals.py
```

Expected: four SVG paths are printed and each file exists under `assets/`.

- [ ] **Step 3: Extend asset tests**

Add assertions that every SVG contains a `<svg` root, a `viewBox`, no `<script>`, no local path, and the expected project-specific label.

- [ ] **Step 4: Run tests**

```powershell
python -m unittest tests.test_portfolio_assets -v
```

Expected: evidence and SVG tests pass; README and app-capture checks remain failing.

- [ ] **Step 5: Commit generated visuals**

```powershell
git add scripts/generate_visuals.py assets/*.svg tests/test_portfolio_assets.py
git commit -m "feat: add reproducible portfolio visuals"
```

### Task 3: Capture actual application screens

**Files:**
- Create: `assets/app-home.png`
- Create: `assets/app-image-result.png`
- Create: `assets/app-text-search.png`

- [ ] **Step 1: Start a clean Android target**

Use the Android emulator QA workflow. Confirm exactly one test device with:

```powershell
adb devices
```

Expected: one `device` entry. Do not use an account containing personal data; use the app's local/mock path.

- [ ] **Step 2: Launch the verified app checkout**

From `D:\app-test\fixmodel-artifact-integration\App`, run the existing app without changing source:

```powershell
flutter run --dart-define=BACKEND_URL=http://10.0.2.2:8000
```

Expected: the application reaches its home screen and loads the local model assets without a model-spec warning.

- [ ] **Step 3: Capture the home screen**

```powershell
adb shell screencap -p /sdcard/app-home.png
adb pull /sdcard/app-home.png C:\Users\Ltp\Downloads\projects\ondevice-landmark-assistant\assets\app-home.png
```

- [ ] **Step 4: Capture image recognition**

Use a non-personal project sample image, run image search, confirm the Top-3 and confidence state are visible, then capture:

```powershell
adb shell screencap -p /sdcard/app-image-result.png
adb pull /sdcard/app-image-result.png C:\Users\Ltp\Downloads\projects\ondevice-landmark-assistant\assets\app-image-result.png
```

- [ ] **Step 5: Capture text search**

Run the query `성곽길과 도시 전망이 보이는 공원`, confirm semantic/keyword search results appear, then capture:

```powershell
adb shell screencap -p /sdcard/app-text-search.png
adb pull /sdcard/app-text-search.png C:\Users\Ltp\Downloads\projects\ondevice-landmark-assistant\assets\app-text-search.png
```

- [ ] **Step 6: Inspect and crop the captures**

Open each PNG, verify it contains no account, notification, email, device identifier, or unrelated desktop content. Crop only device chrome if needed; do not alter application results.

- [ ] **Step 7: Run asset tests and commit**

```powershell
python -m unittest tests.test_portfolio_assets -v
git add assets/app-home.png assets/app-image-result.png assets/app-text-search.png
git commit -m "docs: add verified app captures"
```

### Task 4: Compose the GitHub README

**Files:**
- Create: `README.md`
- Modify: `tests/test_portfolio_assets.py`

- [ ] **Step 1: Build the hero and evidence links**

Start `README.md` with `assets/hero.svg`, a single factual description, and three links: app documentation, team source, and experiment results. Do not add marketing slogans or motivational language.

- [ ] **Step 2: Add the app captures near the top**

Use a GitHub-compatible HTML table with three cells for home, image result, and text search screenshots. Give each image descriptive alt text.

- [ ] **Step 3: Add compact project facts**

Add a four-column Markdown table for `23 classes`, `40 main runs`, `FP16 image/text ONNX`, and `4 confidence states`. Explain that 99.05% is the primary validation mean for S4 full CE, not an open-world accuracy claim.

- [ ] **Step 4: Add the architecture**

Embed a Mermaid flowchart matching `assets/deployment-flow.svg`, followed by two short paragraphs describing image search and text search. Keep the SVG as the visual fallback.

- [ ] **Step 5: Add experiment evidence**

Embed `assets/experiment-comparison.svg`. State the selection rule: validation mean is primary, macro F1 and low-margin counts are supporting signals, and test values do not reverse the selection rule.

- [ ] **Step 6: Add deployment and failure analysis**

Embed `assets/npu-evidence.svg`. State that dynamic INT8 retained FP32-like embeddings on the tested CPU path, while the tested NPU quantized artifact had collapsed accuracy; therefore the NPU number is not presented as a successful optimized model result.

- [ ] **Step 7: Add ownership and limitations**

Use a two-column table:

- User-led: project direction, model training and experiment design, artifact/serving contract, documentation, Sprint 1 demo, final Android architecture.
- Team-owned: final Android code implementation, Auth, notification, and suggestion features.

List these limitations: small closed-set dataset, no open-world generalization claim, final FP16 NPU accuracy not validated, large model bundle, and no claim of real-time performance.

- [ ] **Step 8: Add evidence and reproduction section**

Link the public docs hub, exact experiment-results page, model-serving contract, mobile benchmark, and team repository. Include the verified local contract-test count but do not include local filesystem paths.

- [ ] **Step 9: Extend README tests**

Require all section headings, image alt text, public URLs, ownership wording, and limitation wording. Reject `혁신`, `여정`, `압도적`, `game-changing`, `cutting-edge`, `production-ready`, `fully optimized`, and local absolute paths.

- [ ] **Step 10: Run tests and commit**

```powershell
python -m unittest tests.test_portfolio_assets -v
git add README.md tests/test_portfolio_assets.py
git commit -m "docs: add visual project README"
```

Expected: all tests pass.

### Task 5: GitHub rendering and final audit

**Files:**
- Modify only if validation finds a concrete defect: `README.md`, `assets/*.svg`, `scripts/generate_visuals.py`, `tests/test_portfolio_assets.py`

- [ ] **Step 1: Regenerate assets and confirm a clean diff**

```powershell
python scripts/generate_visuals.py
git diff --exit-code -- assets
```

Expected: no diff, proving deterministic output.

- [ ] **Step 2: Run all portfolio tests**

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Scan for secrets, local paths, and oversized files**

```powershell
rg -n "C:\\\\Users|D:\\\\|service-account|PRIVATE KEY|api[_-]?key|token" README.md assets data scripts
Get-ChildItem -Recurse -File | Where-Object Length -gt 10MB
```

Expected: no secret/local-path match and no file larger than 10 MB.

- [ ] **Step 4: Preview GitHub-compatible rendering**

Open the local README preview or GitHub after an explicitly authorized push. Verify the Mermaid diagram, SVGs, screenshot sizing, narrow-screen stacking, and all links.

- [ ] **Step 5: Review the final diff and commit any audit fixes**

```powershell
git diff --check
git status --short
```

Expected: only intended portfolio files are tracked and the working tree is clean after the final audit commit.

- [ ] **Step 6: Stop before external publication**

Do not push until the user explicitly approves the final local README and assets.

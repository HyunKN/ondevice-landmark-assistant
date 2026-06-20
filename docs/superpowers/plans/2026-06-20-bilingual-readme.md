# Bilingual README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete Korean README and bidirectional language links at the top of both README files.

**Architecture:** Keep `README.md` as the English entry point and add `README.ko.md` as a section-for-section Korean translation. Both documents reuse the same local assets and public evidence links; repository tests enforce cross-links and factual parity.

**Tech Stack:** GitHub Markdown, Python `unittest`

---

### Task 1: Define bilingual README checks

**Files:**
- Modify: `tests/test_portfolio_assets.py`

- [ ] Add `README_KO_PATH = ROOT / "README.ko.md"`.
- [ ] Add a test that requires `README.md` to link to `README.ko.md` and `README.ko.md` to link to `README.md`.
- [ ] Require the Korean document to contain `프로젝트 개요`, `앱 동작`, `시스템 설계`, `실험 근거`, `배포 결과`, `담당 범위`, `한계`, and `근거 자료`.
- [ ] Require both documents to contain `99.05%`, `98.67%`, `0.99941`, `314 ms`, and the same seven asset paths.
- [ ] Run `python -m unittest tests.test_portfolio_assets -v` and confirm failure because `README.ko.md` does not exist.

### Task 2: Add language navigation and Korean translation

**Files:**
- Modify: `README.md`
- Create: `README.ko.md`

- [ ] Add this selector before the hero in `README.md`: `<p align="right"><strong>English</strong> | <a href="README.ko.md">한국어</a></p>`.
- [ ] Add the inverse selector before the hero in `README.ko.md`: `<p align="right"><a href="README.md">English</a> | <strong>한국어</strong></p>`.
- [ ] Translate every English section into natural Korean while preserving all metrics, URLs, image paths, contribution boundaries, and limitations.
- [ ] Keep technical terms in English where clearer and avoid promotional wording.
- [ ] Run `python -m unittest discover -s tests -v` and require all tests to pass.
- [ ] Run `git diff --check`, review both documents, and commit the change.

# Bilingual README Design

## Goal

Add a Korean version of the portfolio README and allow readers to switch languages from the top of either document.

## Files

- `README.md`: English document.
- `README.ko.md`: Korean document with the same structure, evidence, images, metrics, ownership boundary, and limitations.

## Language switch

Place a right-aligned language selector before the hero image.

- `README.md`: `English` is bold; `한국어` links to `README.ko.md`.
- `README.ko.md`: `English` links to `README.md`; `한국어` is bold.
- Use plain repository-relative links. Do not use external badge services.

## Translation rules

- Translate prose and headings into natural Korean.
- Keep technical terms such as MobileCLIP2, ONNX Runtime, Flutter, FP16, INT8, Top-1, macro F1, closed-set, embedding, and NPU in English when clearer.
- Preserve all verified numbers, public URLs, asset paths, caveats, and contribution boundaries exactly.
- Do not add promotional or motivational wording.

## Validation

- Both README files exist and link to each other.
- Both documents reference existing image assets.
- The Korean document contains every required section and the same verified metrics as the English document.
- Existing privacy, forbidden-word, and missing-asset checks continue to pass.

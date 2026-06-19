# On-device Landmark Assistant Portfolio Design

Date: 2026-06-20  
Status: Draft for user review

## Objective

Create a personal GitHub repository that presents the completed Landmark Assistant work as an engineering project, not as a case-study-only repository. The front page must let a reviewer understand the problem, the user's ownership, the model and deployment decisions, the measured results, and the remaining limitations without opening several documents. The README will be visual-first while remaining compatible with GitHub Markdown.

## Chosen approach

Use `HyunKN/ondevice-landmark-assistant` as a standalone portfolio repository. Do not duplicate the full team source tree or large ONNX artifacts. Link to the public documentation hub and the original team application repository as evidence.

This approach preserves the team repository, makes personal ownership explicit, and avoids presenting team-owned Android implementation as individual work.

## Deliverables

### `README.md`

The README will use this order:

1. Project title and one-sentence outcome.
2. Four compact highlights: 23 landmark classes, 40 main experiment runs, image/text ONNX deployment, and four-way confidence handling.
3. A Mermaid architecture diagram covering image/text input, MobileCLIP2 encoders, prototype/text indexes, score policy, Flutter UI, and logging.
4. Problem and constraints.
5. Personal ownership versus team-owned implementation.
6. Model experiment design and selection evidence.
7. On-device integration and model-serving contract.
8. Failed optimization evidence and the engineering decision it caused.
9. Results, limitations, technology stack, and evidence links.

The README will prioritize evidence over a chronological changelog. It will not reproduce authentication, notification, or suggestion-system details because they are not central to the project thesis and were not the user's ideas.

### Visual assets

The repository will use project evidence as visual material:

- A restrained hero banner containing only the project name and core stack.
- An actual app screenshot or short demo GIF near the top of the README.
- A GitHub-rendered Mermaid system architecture diagram.
- A chart generated from the 40-run experiment summary.
- A compact deployment diagram for the image/text ONNX bundle and Android asset flow.
- A failure-analysis figure that separates NPU latency evidence from model accuracy.

The README may link to a longer demo video if one is later recorded, but a presentation script or narrated video is not a required deliverable. Generated imagery may be used only as restrained decoration. App behavior and experiment claims must use actual screenshots and project data.

## Evidence to include

Only verified project evidence will be used:

- 23 supported model classes.
- 8 configurations x 5 folds, 40/40 main runs completed.
- Primary validation Top-1 mean of 99.05% for S4 full CE + hard negative.
- Corresponding test Top-1 mean of 98.67% and test macro F1 of 97.11%.
- Sprint 1 dynamic INT8 versus FP32 embedding cosine mean of 0.99941 and CPU warm median of 314 ms.
- NPU warm latency of 374-518 ms treated only as feasibility evidence because the evaluated quantized model's accuracy collapsed.
- Separate image/text FP16 mixed ONNX encoders with manifest-driven loading.
- Semantic and keyword score fusion plus `matched`, `ambiguous`, `out_of_scope`, and `low_quality` states.
- Contract tests verified in the local checkout: 10 Python tests and 4 Flutter tests passed on 2026-06-20.

Metrics will be accompanied by their scope and caveats. The README will not imply that the final FP16 bundle achieved validated production NPU accuracy or that the high test score proves open-world generalization.

## Ownership statement

The public wording will state:

- The user led project direction, model training and experiment design, artifact and serving-contract decisions, documentation, Sprint 1 demo design and implementation, and the architecture of the final Android application.
- A teammate implemented the final Android application code.
- Authentication, notification, and suggestion features were team contributions and were not proposed by the user.

Any narrower code-level ownership claims must remain consistent with this boundary.

## Public links

- Documentation hub: `https://landmark-assistant-sprint1.vercel.app/`
- Team application repository: `https://github.com/lpcvc-2026-CNU/App`
- Portfolio repository: `https://github.com/HyunKN/ondevice-landmark-assistant`

Local filesystem paths, private credentials, service-account files, personal contact information, and private operational details must not appear in the public repository.

## Presentation constraints

- Korean-first prose with English technical terms where clearer.
- One project thesis: model selection through deployable on-device integration.
- Use GitHub-compatible Markdown, Mermaid, SVG/PNG/GIF, badges, and simple HTML tables only. Do not depend on custom CSS or JavaScript.
- Prefer visual evidence, tables, and diagrams over long prose.
- Keep the README scannable without forcing readers to open the documentation hub.
- Describe the work as a team project with a clearly bounded personal contribution.
- Avoid unsupported terms such as `production-ready`, `fully optimized`, or `real-time`.
- Preserve the failed quantization result because it demonstrates honest engineering judgment.
- Use plain, factual language. Avoid exaggerated or sentimental wording such as `innovative`, `journey`, `game-changing`, `cutting-edge`, or motivational closing statements.

## Validation

Before considering the portfolio ready:

1. Check every metric against the public experiment documents.
2. Verify every public link.
3. Render the Markdown and inspect the Mermaid diagram, tables, images, and narrow-screen behavior.
4. Scan for local absolute paths, email addresses, credentials, tokens, and personal data.
5. Confirm that no claim assigns final Android implementation or Auth/notification/suggestion ideation to the user.
6. Confirm that the README distinguishes command/test success from actual Android-device demonstration.
7. Keep the repository free of model binaries and copied team source code.

## Out of scope

- Modifying the original team repository.
- Copying the complete Flutter/FastAPI codebase.
- Committing ONNX model binaries or external-data files.
- Re-running the physical Android demonstration.
- Recording or publishing a video unless the user later supplies or requests the required footage.
- Pushing this repository before explicit user approval.

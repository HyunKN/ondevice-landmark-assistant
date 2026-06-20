# Android integration trace

The final app is a team repository. This portfolio links the implementation rather than copying teammates' Flutter/Android source and presenting it as individual work.

## Implementation lineage

| Evidence | Scope | Attribution used here |
|---|---|---|
| [`5512601`](https://github.com/lpcvc-2026-CNU/App/commit/551260129527781adba79811a250612f2d99a295) | Initial Flutter/Android app-side integration structure | Team implementation |
| [`onnx_inference_service.dart`](https://github.com/lpcvc-2026-CNU/App/blob/551260129527781adba79811a250612f2d99a295/lib/services/onnx_inference_service.dart) | Flutter ONNX session and inference service | Team implementation |
| [`local_api_client_impl.dart`](https://github.com/lpcvc-2026-CNU/App/blob/551260129527781adba79811a250612f2d99a295/lib/api/local_api_client_impl.dart) | App-facing local inference orchestration | Team implementation |
| [`MainActivity.kt`](https://github.com/lpcvc-2026-CNU/App/blob/551260129527781adba79811a250612f2d99a295/android/app/src/main/kotlin/com/example/landmark_demo_app/MainActivity.kt) | Android asset handoff | Team implementation |
| [`e21e423`](https://github.com/lpcvc-2026-CNU/App/commit/e21e423d421f66774a8f1ab44e87012fcf37f500) | Mobile artifact integration corrections | My integration change |
| [`ab0f7b2`](https://github.com/lpcvc-2026-CNU/App/commit/ab0f7b2c62be70e695b8fb352adcca88070cbacc) | Model integration contract tests | My validation work |
| [`9bfac5b`](https://github.com/lpcvc-2026-CNU/App/commit/9bfac5bb00fbaf2a18c77c38e7afd3d653cc9813) | Semantic text-search artifacts and integration | My integration change |
| [`2e4349b`](https://github.com/lpcvc-2026-CNU/App/commit/2e4349b2a6b960dfff7ab812c5c1b39f5e27c148) | Large-asset cache invalidation fix | My Android fix; exact patch included here |

## Design boundary

I completed the on-device model integration, including the artifact/serving contract, validation, semantic artifacts and integration debugging. The team handled the initial app-side structure, final Flutter/Android app implementation, and app UI design and implementation. Their Flutter widgets, services and Kotlin source are not claimed as my individual work.

Auth, notification, suggestion-management and unrelated backend features are outside this portfolio scope.

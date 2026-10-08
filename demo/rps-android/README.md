# Rock-paper-scissors on Android: the same `model.pte` on a phone

A small native Android app that plays rock-paper-scissors with the **same ExecuTorch `model.pte`** that runs on the IQ‑9075's CPU. The phone's own camera is the input. The model runs on the phone's CPU through ExecuTorch's Android runtime, and the screen shows which model file is running (a hash) and how long each inference takes.

This is the "runs anywhere" half of the repository: nothing about the model changes between a Linux board, a laptop and a phone. Only the app around it does.

```text
IQ-9075:  USB camera   → Python preprocess → executorch (Python)         → XNNPACK → FastAPI game + web UI
Android:  phone camera → Kotlin preprocess → executorch-android (AAR)     → XNNPACK → Compose game UI
                                 same model.pte, same preprocessing, same game rules
```

## Prerequisites

- **Android Studio** (Ladybug or newer). It brings its own JDK and Gradle.
- **An Android phone** on Android 8.0 (API 26) or newer with a 64-bit Arm CPU (`arm64-v8a`, which is nearly every phone from the last several years), with **USB debugging** on.
- **A `model.pte` and its `labels.txt`.** Use the CPU program, not `model_qnn.pte`. Either:
  - download both from this repository's GitHub release (the model from the verified training run), or
  - take them from your own `deploy.zip`, or from the learning block's output if you ran it locally.
- **Optional, for the parity check:** Python with `executorch`, `opencv-python` and `numpy`.

## Quick start

1. **Add the model.** Copy `model.pte` and `labels.txt` into `app/src/main/assets/`, replacing the placeholder `labels.txt`. The order of `labels.txt` must match the model's outputs. For an Edge Impulse model that is alphabetical: `Paper, Rock, Scissors, unknown`.
2. **Open this folder (`demo/rps-android`) in Android Studio** and let it sync. Android Studio supplies Gradle, so no wrapper is checked in.
   If sync fails with *"The selected Kotlin version does not support JDK versions newer than N"*, Gradle is running on a very new JDK. Point it at JDK 17 or 21 under Settings → Build, Execution, Deployment → Build Tools → Gradle → **Gradle JDK** (the bundled `jbr-21` is fine). The Kotlin plugin is already at 2.1.10, which fixes the most common case.
3. **Plug in the phone and press Run.** The run configuration dropdown should say **app**. If it says "Add Configuration…", run File → Sync Project with Gradle Files first.
4. **Grant the camera permission.** The first screen shows `model.pte · sha256 … · ExecuTorch …`. If it shows a red message instead, it says what is wrong: a missing model, labels that don't match the model, or a runtime/version problem.
5. **Hold a hand inside the box and press PLAY.**

To swap models without rebuilding:

```bash
adb push model.pte labels.txt /sdcard/Android/data/com.demo.rpspte/files/
```

The app prefers that folder over the bundled files and says which one it used.

**Same file everywhere.** The 8-character hash on screen is the start of the file's SHA-256. Compare it with `sha256sum model.pte` on any other machine running it.

## Parity check: same scores as Python

A model can "work" but report the wrong class because of preprocessing or label order. One photo catches that.

1. Pick an upright photo of a known gesture, about 1600 px or less on the long side.
2. **Python side** (any computer, including the IQ‑9075):
   ```bash
   pip install executorch==1.5.1 torch==2.14.0 opencv-python numpy
   python3 tools/reference_scores.py --model model.pte --labels labels.txt --image rock.jpg
   ```
   Paths are relative to where you run it; point `--model` and `--labels` at the same files the phone has.
3. **Android side:** save the same photo as `app/src/main/assets/selftest.jpg` (or `adb push` it next to the model) and tap **Self-test**.

Both print a line such as `paper 0.012   rock 0.973   …`. They should agree to about two decimals. If they don't, the cause is the preprocessing or `labels.txt`, not the model. Both sides read the photo without applying its EXIF rotation, so use an image that is already upright.

## What is the same as on the IQ‑9075, and what is not

| | IQ‑9075 (CPU path) | Android |
|---|---|---|
| Model file | `model.pte` | **the same file** |
| Runtime | `executorch` Python package | `executorch-android` AAR (the C++ runtime behind a Kotlin API) |
| Backend | XNNPACK, CPU | XNNPACK, CPU |
| Preprocessing | centre-crop → `cv2.resize` → RGB → ÷255 → NCHW | the same, in `core/Preprocess.kt` |
| Input size | read from the model | probed from the model |
| Camera | USB camera, 640×480 | phone camera, ~640×480 analysis stream |
| Game | Python game loop | `core/GameEngine.kt`, ported from it |
| Output | softmax built into the CPU `.pte` | the same; logits are softmaxed if a model returns them |

The game rules come from the first standalone IQ‑9075 game script (`rps.py`, referenced in code comments; its detector lives on as `demo/rps-amazing-hand/app/pte_perception.py`): a 3-2-1 countdown, SHOOT, a 0.2 s settle, 8 sampled frames, a frame needs confidence ≥ 0.40 and a gesture class (not `unknown`) to vote, the phone picks at random, and the result is held for 3 s.

## Design choices

All of these are constants or one-line changes.

- **Self-contained.** The phone uses its own camera and screen; the latency on screen is the comparison with other devices.
- **Native Kotlin, not a WebView.** The IQ‑9075's web UI needs a server behind it; a phone app doesn't.
- **Portrait, front camera by default**, so the player sees themselves. **Flip camera** switches to the rear one. The model input is *not* mirrored, matching the IQ‑9075's USB camera; only the on-screen preview is.
- **The classified region is the centre 60 %** of the frame's short side, drawn as a box (`RpsViewModel.ROI_PERCENT`).
- **A round needs at least 3 votes and no tie** for the top gesture, otherwise it is "no gesture" rather than a guess (`GameConfig.minVotes`).
- **Thresholds are the proven ones** from the IQ‑9075 script (0.40, 8 frames). The stricter gates in the IQ‑9075 web app (0.75 confidence, 0.35 margin) are not ported.
- **The screen shows the model hash, size, ExecuTorch version and backend**, so "the same file" is visible.

## Versions

A `.pte` loads on the ExecuTorch release that exported it, and in practice on the next releases too. `gradle.properties` pins `executorchVersion=1.3.1`, the release the deployment block uses to build `model.pte`. The model from this repository's release also loads and runs on ExecuTorch 1.5.1 in Python.

If Gradle cannot resolve `org.pytorch:executorch-android` at that version, pick the nearest one listed at <https://mvnrepository.com/artifact/org.pytorch/executorch-android> and run the parity check. A version mismatch shows up as a red "could not load the model" message that mentions the program or its version, not as wrong predictions.

## Tests

The `core` module has no Android classes, so its tests run on any JVM. Run them from Android Studio (right-click `core/src/test` → Run Tests), or with a Gradle 8.9+ install: `gradle :core:test`.

- `PreprocessGoldenTest` compares the Kotlin output with tensors produced by `cv2`, in `core/src/test/resources/golden/` (made by `tools/make_golden.py`; tolerance 1/255).
- `GameEngineTest` covers the countdown, voting, no-gesture, timeout, hold and scoring.

To check the Python side of the maths without Android: `python3 tools/reference_preprocess.py`.

## Layout

```text
core/                       pure Kotlin: Preprocess, GameEngine, Scores (+ tests and golden data)
app/                        the Android app
  .../PteClassifier.kt        Module.load / forward, input-size probe, timing
  .../RpsViewModel.kt         model loading, frame → engine, UI state, self-test
  .../CameraPreview.kt        CameraX preview + RGBA analysis stream
  .../RpsScreen.kt            Compose UI
  .../ModelFiles.kt           asset / adb model lookup, SHA-256 fingerprint
tools/                      reference_preprocess.py, make_golden.py, reference_scores.py
```

## Known gaps and next steps

- **The phone's NPU.** Everything here runs on the CPU. `model_qnn.pte` is compiled for the IQ‑9075 and will not load on a phone. A phone NPU needs a `.pte` lowered for that phone's chip and an Android runtime built with that backend: for a Snapdragon phone, the same QNN backend with a different `--soc-model`; for an Exynos phone, Samsung's own ExecuTorch backend. See [retargeting](../../docs/retargeting.md).
- **No `Module.destroy()`** on teardown. It only matters if the model is reloaded in the same process.
- **Backgrounding the app** during a round makes the round time out to "no gesture".
- **Android 16 and 16 KB memory pages** have not been checked against the AAR's native libraries.
# AANetra: on-device steel surface defect inspection

AANetra is an Android app for quality control on a steel line. The operator takes a photo of a steel
surface. The app finds and classifies the defect, rates its severity, picks a line action
(PASS / ACCEPT / REWORK / REJECT / HOLD / RETAKE) with fixed rules, and then a small language model
running on the phone explains the result in plain words. Everything runs on the phone. After a
one-time model download the app needs no network.

Built for the **Bharat AI-SoC Challenge 2026-27, Problem Statement 5 (PS5)**:
<https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html>

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Decision rules and thresholds](#decision-rules-and-thresholds)
- [Dataset](#dataset)
- [Results on the test set](#results-on-the-test-set)
- [On-device performance](#on-device-performance)
- [App features](#app-features)
- [Repository layout](#repository-layout)
- [How to train](#how-to-train)
- [How to build the app](#how-to-build-the-app)
- [How to install](#how-to-install)
- [Limitations and future work](#limitations-and-future-work)
- [Licences and credits](#licences-and-credits)
- [Team](#team)

---

## Overview

| | |
|---|---|
| Input | One photo of a steel surface (camera or gallery) |
| Defect classes | crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches (NEU-DET) |
| Extra class | not_steel: the photo is not a bare steel surface, so the app asks for a retake |
| Output | Defect class, boxes, severity (Low / Medium / High / Critical / Review), line action, short explanation |
| Runs on | Android 9+ (minSdk 28), CPU only, fully offline after setup |

The design keeps two jobs apart:

- **Pipeline A (vision) decides.** Two small YOLOv8n models, a fusion rule, a severity rule and a
  fixed decision table give the line action. The result is deterministic and testable.
- **Pipeline B (language model) explains.** Llama 3.2 1B Instruct turns the vision result into
  a short explanation and suggested action for the operator. It never changes the decision.

## Architecture

```
 photo ──► image checks ──► Pipeline A (vision)                               ──► decision ──► Pipeline B (Llama)
          size, brightness,   ┌─ YOLOv8n detector (TFLite, 320 px) ─ boxes ─┐     PASS        explanation text
          blur, colour        │                                             ├─ fusion ─► severity ─► ACCEPT ...   (explains only)
          (fail → RETAKE)     └─ YOLOv8n-cls classifier (TFLite, 224 px) ───┘
                                 7 classes: 6 NEU defects + not_steel
```

### Pipeline A: vision

1. **Image checks** (`ImageCheck.kt`): rejects images that are too small, too dark, too bright,
   flat or heavily blurred, or too colourful to be bare steel. A failed check gives RETAKE.
2. **Detector**: YOLOv8n, exported to TFLite, input 320 x 320, 2,100 anchors, 6 NEU classes.
   Decoded on the phone with per-class confidence thresholds and NMS (IoU 0.45).
3. **Classifier**: YOLOv8n-cls, exported to TFLite, input 224 x 224, 7 outputs
   (the 6 NEU classes + `not_steel`).
4. **Fusion** (`phase1-vision/fusion.py`, mirrored in `FusionSeverity.kt`): combines the
   classifier top-1 and the kept detector boxes into one final class and an alert flag.
5. **Severity** (`phase1-vision/severity.py`): area of the boxes of the final class relative to
   the image, mapped to Low / Medium / High / Critical, with a count escalation.
6. **Decision** (`QualityDecision.kt`): fixed table from severity to line action.

The Python and Kotlin versions of fusion and severity share the same thresholds, and unit tests on
both sides pin the values.

### Pipeline B: explanation

- **Model:** Llama 3.2 1B Instruct.
- **Runtime:** ExecuTorch 1.5.1 with the XNNPACK backend (CPU).
- **Quantisation:** 8da4w (8-bit dynamic activations, 4-bit grouped weights).
- **Export config:** `phase2-reasoning/export/llama_config.yaml`.
- **Role:** explains only. The prompt contains the vision result and the decision that was already
  made. Generation stops after the ACTION line to keep answers short, and answers are cached per
  result.

## Decision rules and thresholds

### Fusion

| Step | Rule |
|---|---|
| 0. not_steel gate | classifier top-1 = `not_steel` with confidence >= **0.33** → **RETAKE** (boxes ignored) |
| 1. Kept boxes | detector boxes above the per-class threshold: crazing **0.10**, inclusion **0.50**, patches **0.55**, pitted_surface **0.15**, rolled-in_scale **0.15**, scratches **0.30** |
| 2. Final class | classifier class if its confidence >= **0.64**, else the class of the best kept box, else "no defect found" |
| 3. Alert | any kept box, or classifier confidence >= 0.64 |
| 4. Severity | from the kept boxes of the final class; alert but no matching box → **Review** |

All fusion thresholds were chosen on the validation split, never on the test split.

### Severity

Weighted area ratio = box area / image area, using the largest box of the final class.

| Area ratio | Level |
|---|---|
| < 0.082 | Low |
| 0.082 – 0.184 | Medium |
| 0.184 – 0.359 | High |
| > 0.359 | Critical |

The cut-offs are the 40th / 70th / 90th percentiles of the ground-truth box areas in the validation
split. With 3 or more kept boxes of the final class, the level goes up one step (capped at Critical).

### Line action

| Vision result | Action |
|---|---|
| Image check failed, or not_steel | **RETAKE** (no part decision) |
| Unsure: classifier did not fire and best box < 0.50, or boxes of 3+ different classes | **HOLD** (manual check) |
| No defect | **PASS** |
| Low | **ACCEPT** |
| Medium | **REWORK** |
| High | **REJECT** |
| Critical | **REJECT + STOP LINE** |
| Review / unknown | **HOLD** |

## Dataset

**NEU-DET** (Northeastern University surface defect database): 1,800 grayscale 200 x 200 images of
hot-rolled steel strip, 300 per class, with bounding boxes.

- Split **70 / 15 / 15** (train / val / test), stratified per class, fixed seed 42
  (`phase1-vision/scripts/prepare_data.py`). The same split is used for the detector and the
  classifier (`build_cls_dataset.py` checks that the splits do not overlap).
- Test set: 270 steel images.

**not_steel**: 2,000 images that are not bare steel, used to train the 7th classifier class
(`phase1-vision/scripts/build_not_steel.py`):

- **COCO** (everyday scenes and objects),
- **DTD** (Describable Textures Dataset, non-steel textures),
- **Places365** (fields, ground, roads, buildings).

Images were downloaded from the official dataset URLs. The manifest with the chosen files is in
`phase1-vision/results/phase1_2/not_steel_manifest.json`. Test set: 300 not_steel images.

Datasets are not included in this repository (see [Licences and credits](#licences-and-credits)).

## Results on the test set

The test split was used once, after all thresholds were fixed on validation.

| Metric | Result |
|---|---|
| Steel images classified correctly (7-class classifier) | **269 / 270** |
| not_steel images rejected (RETAKE) | **298 / 300** |
| Classifier overall | **567 / 570** (99.5 %) |
| Detector mAP50 | **0.741** |
| Detector mAP50-95 | **0.413** |
| Detector precision / recall | **0.67 / 0.74** |

Raw numbers, confusion matrices and curves are in `phase1-vision/results/`
(`phase1_1/` for the detector and the 6-class classifier, `phase1_2/` for the 7-class classifier
and the not_steel threshold).

## On-device performance

Tested on:

- **Samsung Galaxy A16 5G** (6 GB RAM)
- **Samsung Galaxy A12**

| Stage | Time |
|---|---|
| Vision (image checks + detector + classifier + fusion) | ~0.2 – 0.5 s per image |
| Llama 3.2 1B explanation | ~11 tokens/s |

The decision is shown as soon as vision finishes. The explanation streams in after it.
`phase2-reasoning/android/benchmarks/benchmark_template.csv` is the sheet used to record runs.

## App features

- **Camera or gallery** input.
- **Image checks** before inference (size, brightness, blur, colour).
- **RETAKE** for photos that are not a bare steel surface; **HOLD** when the vision result is unsure.
- Boxes drawn on the image, defect class, severity and line action.
- **Explanation** from Llama on the phone, streamed, with a suggested action.
- **Inspection history** with timestamps and thumbnails, and **CSV export**.
- **Offline**: the vision models are inside the APK. Llama (`llama3_2.pte`, `tokenizer.model`) is
  downloaded once on first use from
  <https://github.com/ajaybahuguna17/aanetra-models/releases/tag/v1.0-llama>
  and stored on the phone. After that no network is used.

## Repository layout

```
.
├── run_demo.bat                     desktop demo (Python, Gradio) of Pipeline A
├── phase1-vision/                   Pipeline A: data, training, export, rules
│   ├── fusion.py                    fusion rule (detector + classifier)
│   ├── severity.py                  severity rule
│   ├── notebooks/                   Colab training notebooks
│   │   ├── train_colab.ipynb            Phase 1: detector
│   │   ├── train_colab_phase1_1.ipynb   Phase 1.1: 70/15/15 split, detector + 6-class classifier
│   │   └── train_colab_phase1_2.ipynb   Phase 1.2: 7-class classifier with not_steel
│   ├── scripts/                     data prep, TFLite export, evaluation, threshold tuning
│   ├── demo/                        desktop demo (app.py, predict.py, pipeline.py, examples/)
│   ├── tests/                       Python unit tests
│   ├── data/                        dataset folders (images not in git)
│   ├── models/exported/             exported TFLite models go here (not in git)
│   └── results/                     metrics, confusion matrices, chosen thresholds
└── phase2-reasoning/
    ├── export/llama_config.yaml     ExecuTorch export config for Llama 3.2 1B (8da4w, XNNPACK)
    └── android/                     Android app (Gradle project)
        ├── core/                    contracts and the decision table
        ├── vision/                  TFLite vision pipeline, image checks, fusion, severity
        ├── advisor/                 prompt building and parsing for Llama
        ├── app/                     UI, camera, model download, history
        ├── benchmarks/, samples/    templates for on-device measurements and reviews
        └── EXECUTORCH_LICENSE
```

## How to train

Training runs on Google Colab (a T4 GPU is enough).

1. Open the notebooks in `phase1-vision/notebooks/` in order:
   `train_colab.ipynb` → `train_colab_phase1_1.ipynb` → `train_colab_phase1_2.ipynb`.
2. Each notebook downloads its data, builds the split, trains, exports TFLite and writes
   `model_io.json`, `labels.txt` and the metrics.
3. Copy the exported files to `phase1-vision/models/exported/`:

```
phase1-vision/models/exported/
├── phase1_1/detector/     *.tflite, labels.txt, model_io.json
├── phase1_1/classifier/   6-class classifier (fallback)
└── phase1_2/classifier/   7-class classifier with not_steel (used when present)
```

Run the Python unit tests:

```bash
cd phase1-vision
pip install -r requirements.txt
python -m unittest discover -s tests
```

Desktop demo (optional): `pip install -r phase1-vision/requirements-demo.txt`, then `run_demo.bat`.

The Llama model is exported with ExecuTorch 1.5.1 using `phase2-reasoning/export/llama_config.yaml`.
The ready-made files are on the
[model release page](https://github.com/ajaybahuguna17/aanetra-models/releases/tag/v1.0-llama).

## How to build the app

Requirements:

- **JDK 21** (set `JAVA_HOME` to it)
- **Android SDK 35** (set `sdk.dir` in `phase2-reasoning/android/local.properties`, or `ANDROID_HOME`)
- The vision models in `phase1-vision/models/exported/` (see above). The build copies them into the APK.

```bat
cd phase2-reasoning\android
gradlew.bat :app:assembleDebug
```

The APK is written to `phase2-reasoning/android/app/build/outputs/apk/debug/`.

Unit tests:

```bat
gradlew.bat :core:testDebugUnitTest :vision:testDebugUnitTest :advisor:testDebugUnitTest
```

## How to install

1. Enable USB debugging on the phone and connect it.
2. `adb install -r app\build\outputs\apk\debug\app-debug.apk`
3. Open AANetra. On first use of the explanation, allow the one-time Llama download
   (about 800 MB, Wi-Fi recommended). After that the app works offline.

## Limitations and future work

- **NEU domain gap.** NEU-DET images are 200 x 200 grayscale crops taken under lab conditions.
  Real phone photos on a line differ in lighting, scale, angle and texture. Severity cut-offs were
  tuned on NEU crops and must be re-tuned on real images.
- **No defect-free images.** NEU-DET has no "good" steel class, so PASS cannot be measured on real
  clean steel yet. The not_steel class covers wrong photos, not clean steel.
- **Live camera mode.** Today the app works on single photos. Next step: continuous inspection on
  the camera stream.
- **Real factory data.** Collect and label images from a real line, retrain, and validate the rules
  with plant quality staff.

## Licences and credits

**Built with Llama.**

- **Llama 3.2** is used under the
  [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/).
  "Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc.
  All Rights Reserved." The model files are not stored in this repository.
- **NEU-DET.** Please cite:
  - K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for
    hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858–864, 2013.
  - Y. He, K. Song, Q. Meng and Y. Yan, "An End-to-end Steel Surface Defect Detection Approach via
    Fusing Multiple Hierarchical Features," *IEEE Transactions on Instrumentation and Measurement*,
    vol. 69, no. 4, pp. 1493–1504, 2020.
- **DTD** (Describable Textures Dataset), **Places365** and **COCO** images were used for the
  not_steel class under their research-use terms (COCO annotations CC BY 4.0, images under their
  Flickr terms; DTD and Places365 for non-commercial research and education). No images from these
  datasets are redistributed here.
- **ExecuTorch LlamaDemo.** The Android app is based on the ExecuTorch Llama demo app, BSD licence.
  The original licence is kept in `phase2-reasoning/android/EXECUTORCH_LICENSE`.
- **Ultralytics YOLOv8** is used for training and export under AGPL-3.0. Models trained with it and
  code that uses it are subject to the AGPL-3.0 terms.
- **LiteRT / TensorFlow Lite** runs the vision models on the phone (Apache-2.0).

## Team

| Role | Name |
|---|---|
| Team member | [Student 1] |
| Team member | [Student 2] |
| Team member | [Student 3] |
| Institution | [College] |
| Faculty mentor | [Faculty mentor] |

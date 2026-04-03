# Heart CV-gnal

**Pure Computer Vision — Real-time Affection / Interest Analyzer**

A mock dating simulator that quantifies non-verbal engagement signals in real time using only a webcam.  
No audio, no NLP — MediaPipe + OpenCV only. Optional Claude Vision API cross-validation.

---

## Project Overview

Heart CV-gnal measures perceived affection and interest through body language alone. During a short demo session, the system extracts distance-invariant facial and postural features from a live webcam feed, evaluates them against five psychologically grounded behavioral rules, and produces a continuous affection score (0–100). An optional Vision-Language Model (VLM) cross-validator blends a Claude Vision API assessment into the final score every 30 seconds, providing a second opinion without blocking the main pipeline.

---

## Pipeline

```
Webcam Frame (1280×720)
        │
        ▼
MediaPipe Holistic
  ├─ Face Mesh  (468 landmarks)
  └─ Pose       (33 landmarks)
        │
        ▼
FeatureExtractor                      stateless, distance-invariant
  mouth_ratio   eye_ratio   shoulder_ratio
  roll_angle_deg   yaw_deg   pitch_deg   wrists_crossed
        │
        ▼
AffectionEngine                       stateful, 5-second windows
  3 s calibration → personal baseline locked
  batch_evaluate() → score delta + event log
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼  (optional, async)
  score [0–100]                               VLMAnalyzer
  mood label                          Claude Vision API (every 30 s)
  active signals                      blended: 0.65 × rule + 0.35 × VLM
        │
        ▼
Flask Web UI  /  OpenCV Desktop App
  real-time gauge · event banners · signal indicators
```

### Feature Extraction

All features are normalized relative to face/shoulder scale so that moving closer to or farther from the camera does not affect the score.

| Feature | Description |
|---|---|
| `mouth_ratio` | mouth width / cheek width |
| `eye_ratio` | avg eye-opening height / cheek width |
| `shoulder_ratio` | inter-shoulder distance / face width |
| `roll_angle_deg` | head tilt angle from horizontal eye line |
| `yaw_deg` / `pitch_deg` | normalized nose-offset ratios from face center |
| `wrists_crossed` | boolean — both wrists overlap in the torso region |

### Scoring Rules

Score starts at **50**, clamped to **[0, 100]**. Rules are evaluated over a 5-second sliding window after a 3-second personal calibration phase.

| Signal | Condition | Δ Score | Frame Threshold |
|---|---|---|---|
| **Duchenne Smile** | mouth_ratio ≥ +12% AND eye_ratio ≤ −10% vs. baseline | **+5** | ≥ 25% of frames |
| **Leaning In** | shoulder_ratio ≥ 1.15× baseline | **+4** | ≥ 50% of frames |
| **Head Tilt** | roll angle 5°–20° | **+2** | ≥ 40% of frames |
| **Looking Away** | yaw > ±40° or pitch > ±30° | **−3** | mean exceeds threshold |
| **Barrier Signal** | both wrists crossed | **−5** | ≥ 50% of frames |

---

## Output (Evaluation)

At the end of the 3-minute session the system displays a final affection score with a mood label and a verdict message.

| Score Range | Mood Label | Final Message |
|---|---|---|
| 88–100 | Head Over Heels | It's a Match! ♥ |
| 75–87 | Smitten | It's a Match! ♥ |
| 60–74 | Interested | Definitely Something There... |
| 40–59 | Warming Up | Definitely Something There... |
| 20–39 | Mildly Curious | Let's Just Be Friends... |
| 0–19 | Not Feeling It | Let's Just Be Friends... |

When VLM cross-validation is enabled, the final score also includes the VLM's reasoning and confidence level displayed in the UI panel.

---

## Repository Structure

```
.
├── app.py                            # Flask web server — MJPEG stream + /status JSON
├── apps/
│   └── run_heart_cvgnal.py           # Desktop (OpenCV) entry point
├── src/heart_cvgnal/
│   ├── app/
│   │   └── runner.py                 # Webcam loop, calibration/eval phases, OpenCV UI
│   └── pipelines/vision/
│       ├── feature_extractor.py      # Stateless — landmarks → normalized features
│       ├── affection_engine.py       # Stateful — calibration, batch evaluation, event log
│       └── vlm_analyzer.py           # Optional — Claude Vision API, daemon thread, score blend
├── templates/
│   ├── index.html                    # Web UI (pastel Heart Signal theme)
│   └── heartsignal.png               # Logo asset
├── tests/
│   └── conftest.py                   # Pytest fixtures, sys.path config
├── requirements.txt
└── run_heart_cvgnal.command          # macOS double-click launcher
```

---

## Installation

```bash
conda create -n dslcv2 python=3.11 -y
conda activate dslcv2
pip install -r requirements.txt
pip install "mediapipe==0.10.14"
```

### Enable VLM Cross-Validation (optional)

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Running

```bash
# Web app (recommended)
conda activate dslcv2
PYTHONPATH=src python app.py
# → http://localhost:5001

# Desktop (OpenCV window)
PYTHONPATH=src python apps/run_heart_cvgnal.py

# macOS double-click
# run_heart_cvgnal.command  (right-click → Open on first run)
```

---

## References

- [MediaPipe Holistic](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker) — real-time face mesh and pose estimation
- [Duchenne Marker](https://en.wikipedia.org/wiki/Duchenne_smile) — AU6 + AU12 as genuine smile indicator
- [Proxemics & Leaning](https://en.wikipedia.org/wiki/Proxemics) — interpersonal distance as interest signal (Hall, 1966)
- [Head Tilt as Engagement Cue](https://psycnet.apa.org/record/1967-08867-001) — head tilt as attentiveness signal
- [Arm-Crossing as Barrier Behavior](https://en.wikipedia.org/wiki/Body_language) — defensive/closed posture literature
- [Anthropic Claude Vision API](https://docs.anthropic.com/en/api/messages) — VLM cross-validator (claude-haiku-4-5)

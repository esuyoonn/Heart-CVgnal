# Heart CV-gnal

**Pure Computer Vision — Real-time Affection / Interest Analyzer**

A mock dating simulator that analyzes non-verbal engagement signals in real time using only a webcam.
No audio or NLP — MediaPipe + OpenCV only. Optional Claude Vision API cross-validation.

---

## Signals

| Signal | Condition | Score |
|---|---|---|
| **Duchenne Smile** | Mouth corners ≥ 12% wider AND eyes ≤ 10% narrower vs. baseline (≥ 25% of frames in window) | **+5 pts** |
| **Leaning In** | Shoulder width ≥ 1.15× baseline (≥ 50% of frames) | **+4 pts** |
| **Head Tilt** | 5–20° roll (≥ 40% of frames) | **+2 pts** |
| **Barrier Signal** | Both wrists crossed (≥ 50% of frames) | **-5 pts** |
| **Looking Away** | Yaw > ±40° or Pitch > ±30° (≥ 50% of frames) | **-3 pts** |

Score starts at **50**, clamped to **[0, 100]**. Session duration: **3 minutes**.

---

## Running

### Web app (recommended)
```bash
conda activate dslcv2
PYTHONPATH=src python app.py
# → http://localhost:5001
```

### Terminal (desktop OpenCV)
```bash
conda activate dslcv2
PYTHONPATH=src python apps/run_heart_cvgnal.py
```

### macOS double-click
Double-click `run_heart_cvgnal.command`
(First run: right-click → Open if macOS blocks it)

---

## Installation (first time)

```bash
conda create -n dslcv2 python=3.11 -y
conda activate dslcv2
pip install -r requirements.txt
pip install "mediapipe==0.10.14"
```

### Enable VLM cross-validation (optional)
Claude Vision API cross-checks the rule-based score every 30 seconds.
```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Architecture

```
app.py                            ← Flask web server (recommended)
apps/run_heart_cvgnal.py          ← Desktop entry point
src/heart_cvgnal/
  pipelines/vision/
    feature_extractor.py          ← Distance-invariant ratio/angle extraction (stateless)
    affection_engine.py           ← 5-rule scoring engine (stateful)
    vlm_analyzer.py               ← Claude Vision API cross-validator (optional)
  app/runner.py                   ← Webcam loop + OpenCV UI
```

```
Webcam Frame
    ↓ MediaPipe Holistic (Face 468pt + Pose 33pt)
FeatureExtractor  →  mouth_ratio, eye_ratio, shoulder_ratio, roll_angle_deg, yaw_deg, pitch_deg, wrists_crossed
    ↓
AffectionEngine   →  score [0-100], events, active signals
    ↓ (optional) VLMAnalyzer  →  Claude Vision cross-check (30s interval, 35% blend)
    ↓
Flask / OpenCV    →  real-time UI (gauge, banners, signal indicators)
```

---

## Controls (desktop)

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |

---

## Troubleshooting

- **Camera not detected** → System Settings → Privacy & Security → Camera → allow Terminal
- **`mediapipe has no attribute 'solutions'`** → `pip install "mediapipe==0.10.14"`
- **Slow frames** → Close Zoom/FaceTime, lower resolution in `runner.py`
- **VLM offline** → `pip install anthropic` and set the `ANTHROPIC_API_KEY` environment variable

# Heart CV-gnal

**Pure Computer Vision — Real-time Affection / Interest Analyzer**

웹캠 영상만으로 상대방의 비언어적 관심 신호를 실시간 분석하는 Mock Dating Simulator.
오디오·NLP 없이 MediaPipe + OpenCV만 사용. Claude Vision API로 크로스 검증 (선택).

---

## 측정 신호

| 신호 | 조건 | 점수 |
|---|---|---|
| **Duchenne Smile** | 입꼬리 ≥ 12% ↑ AND 눈 ≤ 10% ↓ (베이스라인 대비, 윈도우 내 25% 이상 프레임) | **+5 pts** |
| **Leaning In** | 어깨 너비 ≥ 1.15× 베이스라인 (50% 이상 프레임) | **+4 pts** |
| **Head Tilt** | 5–20° 기울임 (40% 이상 프레임) | **+2 pts** |
| **Barrier Signal** | 양 손목 교차 (50% 이상 프레임) | **-5 pts** |
| **Looking Away** | Yaw > ±40° 또는 Pitch > ±30° (50% 이상 프레임) | **-3 pts** |

점수는 **50**에서 시작, **[0, 100]** 범위. 측정 시간 **3분**.

---

## 실행 방법

### 웹 앱 (권장)
```bash
conda activate dslcv2
PYTHONPATH=src python app.py
# → http://localhost:5001
```

### 터미널 (데스크탑)
```bash
conda activate dslcv2
PYTHONPATH=src python apps/run_heart_cvgnal.py
```

### macOS 더블클릭
`run_heart_cvgnal.command` 더블클릭
(최초 실행 시: 우클릭 → Open)

---

## 설치 (최초 1회)

```bash
conda create -n dslcv2 python=3.11 -y
conda activate dslcv2
pip install -r requirements.txt
pip install "mediapipe==0.10.14"
```

### VLM 크로스 검증 활성화 (선택)
Claude Vision API가 30초마다 룰 기반 점수를 교차 검증합니다.
```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 아키텍처

```
app.py                            ← Flask 웹 서버 (권장)
apps/run_heart_cvgnal.py          ← 데스크탑 진입점
src/heart_cvgnal/
  pipelines/vision/
    feature_extractor.py          ← 거리 불변 비율/각도 추출 (stateless)
    affection_engine.py           ← 5규칙 점수 엔진 (stateful)
    vlm_analyzer.py               ← Claude Vision API 크로스 검증 (optional)
  app/runner.py                   ← 웹캠 루프 + OpenCV UI
```

```
Webcam Frame
    ↓ MediaPipe Holistic (Face 468pt + Pose 33pt)
FeatureExtractor  →  mouth_ratio, eye_ratio, shoulder_ratio, roll_angle_deg, yaw_deg, pitch_deg, wrists_crossed
    ↓
AffectionEngine   →  score [0-100], events, active signals
    ↓ (선택) VLMAnalyzer  →  Claude Vision 크로스 검증 (30s 간격, 35% 블렌딩)
    ↓
Flask / OpenCV    →  실시간 UI (게이지, 배너, 신호 지표)
```

---

## 조작키 (데스크탑)

| 키 | 동작 |
|---|---|
| `Q` / `ESC` | 종료 |

---

## 문제해결

- **카메라 안 잡힘** → 시스템 설정 → 개인 정보 → 카메라 → 터미널/IDE 허용
- **`mediapipe has no attribute 'solutions'`** → `pip install "mediapipe==0.10.14"`
- **프레임 느림** → Zoom/FaceTime 종료, `runner.py`에서 해상도 낮추기
- **VLM offline** → `pip install anthropic` 후 `ANTHROPIC_API_KEY` 환경변수 설정

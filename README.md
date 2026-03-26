# Heart CV-gnal

**Pure Computer Vision — Real-time Affection / Interest Analyzer**

웹캠 영상만으로 상대방의 비언어적 관심 신호를 실시간 분석하는 Mock Dating Simulator.
오디오·NLP 없이 MediaPipe + OpenCV만 사용.

---

## 측정 신호 (4가지)

| 신호 | 조건 | 점수 |
|---|---|---|
| **Duchenne Smile** | 입꼬리 ≥ 12% ↑ AND 눈 ≤ 10% ↓ (베이스라인 대비) | **+5 pts** |
| **Leaning In** | 어깨 너비 ≥ 1.15× 베이스라인 (카메라 방향 기울기) | **+2 pts/s** |
| **Synchrony** | 고개 5–25° 기울임 2초 지속 OR 끄덕임 감지 | **+1.5 pts/s** |
| **Barrier Signal** | 양 손목이 반대쪽 어깨 교차 3초 이상 | **-3 pts/s** |

점수는 **50**에서 시작, **[0, 100]** 범위.

---

## 실행 방법

### 터미널
```bash
conda activate dslcv2
PYTHONPATH=src python apps/run_heart_cvgnal.py
# Windows Powershell
$env:PYTHONPATH="src"; python apps/run_heart_cvgnal.py
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

---

## 아키텍처

```
apps/run_heart_cvgnal.py          ← 진입점
src/heart_cvgnal/
  pipelines/vision/
    feature_extractor.py          ← 거리 불변 비율/각도 추출 (stateless)
    affection_engine.py           ← 4규칙 점수 엔진 (stateful)
  app/runner.py                   ← 웹캠 루프 + UI 렌더링
```

```
Webcam Frame
    ↓ MediaPipe Holistic (Face 468pt + Pose 33pt)
FeatureExtractor  →  mouth_ratio, eye_ratio, shoulder_ratio, roll_angle_deg, wrists_crossed
    ↓
AffectionEngine   →  score [0-100], events, active signals
    ↓
HeartCVgnalApp    →  실시간 오버레이 (게이지, 배너, 지표)
```

---

## 조작키

| 키 | 동작 |
|---|---|
| `Q` / `ESC` | 종료 |

---

## 문제해결

- **카메라 안 잡힘** → 시스템 설정 → 개인 정보 → 카메라 → 터미널/IDE 허용
- **`mediapipe has no attribute 'solutions'`** → `pip install "mediapipe==0.10.14"`
- **프레임 느림** → Zoom/FaceTime 종료, `runner.py`에서 해상도 낮추기

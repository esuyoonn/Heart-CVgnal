"""
Heart CV-gnal — Main Application Runner  (Demo Edition)
=========================================================
Pink romantic theme · 5-second evaluation windows · 5-minute demo timer.

Controls
--------
Q / ESC      — quit at any time
Any key      — close the final result screen

Run via::

    PYTHONPATH=src python apps/run_heart_cvgnal.py
"""

from __future__ import annotations

import time
from typing import List, Optional

import threading

import cv2
import mediapipe as mp
import numpy as np

from ..pipelines.vision.affection_engine import AffectionEngine, AffectionOutput
from ..pipelines.vision.feature_extractor import FeatureExtractor, FrameFeatures
from ..pipelines.vision.vlm_analyzer import VLMAnalyzer

# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------
_DEMO_DURATION    = 300.0   # 5 minutes (seconds)
_EVAL_WINDOW      = 5.0     # scoring window size (seconds)
_EVENT_DISPLAY    = 4.0     # how long the popup banner stays on screen

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
_PANEL_W   = 260   # right-side score panel width (px)
_PANEL_PAD = 10    # outer margin

# ---------------------------------------------------------------------------
# Pink Romantic Colour Palette  (OpenCV BGR)
# ---------------------------------------------------------------------------
_PINK_LIGHT  = (203, 192, 255)   # soft pastel pink
_PINK_HOT    = (147,  20, 255)   # deep magenta / hot pink
_PINK_MED    = (180, 105, 255)   # medium lavender-pink
_PINK_PANEL  = ( 40,  15,  50)   # very dark aubergine  (panel bg)
_PINK_BORDER = (160,  80, 200)   # medium purple-pink   (borders)
_WHITE       = (255, 255, 255)
_GRAY_DIM    = ( 80,  70,  90)

# ---------------------------------------------------------------------------
# Mood labels
# ---------------------------------------------------------------------------
_MOOD_LABELS = [
    (0,  "Not Feeling It"),
    (20, "Mildly Curious"),
    (40, "Warming Up"),
    (60, "Interested"),
    (75, "Smitten"),
    (88, "Head Over Heels"),
]


def _mood_label(score: float) -> str:
    label = _MOOD_LABELS[0][1]
    for threshold, text in _MOOD_LABELS:
        if score >= threshold:
            label = text
    return label


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class HeartCVgnalApp:
    """
    Top-level application.  Instantiate once; call ``run()`` to start.

    Architecture
    ------------
    * Calibration phase  (first ~3 s): ``engine.update()`` called every frame.
    * Evaluation phase   (remaining demo time):
        - Features are buffered in ``_feature_buf``.
        - Every ``_EVAL_WINDOW`` seconds → ``engine.batch_evaluate()`` is called
          exactly ONCE, score is updated, popup fires.
        - The cached ``_last_output`` is used for display between evaluations
          (prevents per-frame flicker).
    """

    def __init__(self, camera_index: int = 0) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {camera_index}.  "
                "Check System Settings → Privacy & Security → Camera."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        self._extractor = FeatureExtractor()
        self._engine    = AffectionEngine()

        self._mp_holistic = mp.solutions.holistic
        self._mp_drawing  = mp.solutions.drawing_utils

        # ── Demo / windowing state (all wall-clock via time.time()) ─────
        self._demo_start:     Optional[float]          = None
        self._window_start:   Optional[float]          = None
        self._feature_buf:    list[FrameFeatures]       = []
        self._last_output:    Optional[AffectionOutput] = None
        self._was_calibrated: bool                      = False
        self._last_frame:     Optional[np.ndarray]      = None

        # ── VLM cross-validator (optional — degrades silently if unavailable) ─
        self._vlm             = VLMAnalyzer()
        self._blended_score: float = 50.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("=" * 58)
        print("  Heart CV-gnal  ♥  Press Q or ESC to quit")
        print("=" * 58)

        holistic_cfg = dict(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with self._mp_holistic.Holistic(**holistic_cfg) as holistic:
            while self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    print("[WARN] Frame read failed — retrying …")
                    continue

                # Mirror for natural selfie view
                frame = cv2.flip(frame, 1)
                now = time.time()
                ts  = time.monotonic()   # monotonic for engine internals

                # ── MediaPipe inference ───────────────────────────────────
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                # ── Feature extraction ────────────────────────────────────
                h_fr, w_fr = frame.shape[:2]
                features = self._extractor.extract(results, w_fr, h_fr)

                # ── Score / calibration logic ─────────────────────────────
                if not self._was_calibrated:
                    # --- Calibration phase: update every frame ---
                    output = self._engine.update(features, ts)
                    self._last_output = output

                    if output.calibrated:
                        # Calibration just completed → start demo timers
                        self._was_calibrated = True
                        self._demo_start   = now
                        self._window_start = now
                        print("[INFO] Calibration done. Demo timer started.")
                else:
                    # --- Evaluation phase: buffer → batch every 5 s ---
                    self._feature_buf.append(features)
                    window_elapsed = now - self._window_start  # type: ignore[operator]

                    if window_elapsed >= _EVAL_WINDOW:
                        output = self._engine.batch_evaluate(
                            self._feature_buf, window_elapsed, ts
                        )
                        self._last_output  = output
                        self._feature_buf.clear()
                        self._window_start = now
                    else:
                        # Use the cached output between evaluations (no flicker)
                        output = self._last_output  # type: ignore[assignment]

                # ── VLM: async trigger + blend score for display ─────────
                if self._was_calibrated:
                    self._vlm.maybe_trigger(frame, now)
                    self._blended_score = self._vlm.blend(output.score)
                else:
                    self._blended_score = output.score

                # ── Demo timer check ──────────────────────────────────────
                if self._demo_start is not None:
                    remaining = _DEMO_DURATION - (now - self._demo_start)
                    if remaining <= 0:
                        self._last_frame = frame.copy()
                        break
                else:
                    remaining = _DEMO_DURATION   # timer not yet running

                # ── Pink face bounding box ────────────────────────────────
                if results.face_landmarks:
                    h_f, w_f = frame.shape[:2]
                    xs = [lm.x * w_f for lm in results.face_landmarks.landmark]
                    ys = [lm.y * h_f for lm in results.face_landmarks.landmark]
                    x1 = max(0, int(min(xs)) - 8)
                    y1 = max(0, int(min(ys)) - 8)
                    x2 = min(w_f, int(max(xs)) + 8)
                    y2 = min(h_f, int(max(ys)) + 8)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), _PINK_HOT, 2)

                # ── Pose skeleton (pink tones) ────────────────────────────
                if results.pose_landmarks:
                    self._mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self._mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=self._mp_drawing.DrawingSpec(
                            color=_PINK_MED, thickness=1, circle_radius=2
                        ),
                        connection_drawing_spec=self._mp_drawing.DrawingSpec(
                            color=_PINK_LIGHT, thickness=1
                        ),
                    )

                # ── Render overlay ────────────────────────────────────────
                if not output.calibrated:
                    frame = self._render_calibrating(frame, output)
                else:
                    frame = self._render_score_panel(frame, features, output)
                    frame = self._render_status_dots(frame, features)
                    frame = self._render_active_signals(frame, output)
                    frame = self._render_timer(frame, remaining, now)
                    # Event banner rendered last so it sits on top of everything
                    frame = self._render_event_banner(frame, output, ts)

                self._draw_fps(frame, ts)

                cv2.imshow("Heart CV-gnal", frame)

                key = cv2.waitKey(5) & 0xFF
                if key in (ord('q'), 27):
                    self._cleanup()
                    return

        # ── 5-minute timer expired → show final static result screen ────
        final_score = self._last_output.score if self._last_output else 50.0
        self._show_final_screen(self._last_frame, final_score)
        self._cleanup()

    # ------------------------------------------------------------------
    # Final Result Screen
    # ------------------------------------------------------------------

    def _show_final_screen(
        self, last_frame: Optional[np.ndarray], score: float
    ) -> None:
        """
        Render a static "Time's Up" result screen and block until any key.
        """
        if last_frame is None:
            # Fallback: solid dark-pink canvas
            result = np.full((720, 1280, 3), (50, 18, 65), dtype=np.uint8)
        else:
            # Heavily blur the last camera frame
            result = cv2.GaussianBlur(last_frame, (61, 61), 25)

        h, w = result.shape[:2]
        cx, cy = w // 2, h // 2

        # Pink semi-transparent wash over the blurred frame
        wash = np.full_like(result, (55, 18, 70))
        cv2.addWeighted(wash, 0.60, result, 0.40, 0, result)

        # Hot-pink accent bars (top & bottom)
        cv2.rectangle(result, (0, 0),     (w, 10),  _PINK_HOT, -1)
        cv2.rectangle(result, (0, h - 10),(w, h),   _PINK_HOT, -1)

        # ── "Time's Up!" ──────────────────────────────────────────────
        line1 = "Time's Up!"
        (tw1, th1), _ = cv2.getTextSize(
            line1, cv2.FONT_HERSHEY_DUPLEX, 2.2, 3
        )
        cv2.putText(
            result, line1,
            (cx - tw1 // 2, cy - 110),
            cv2.FONT_HERSHEY_DUPLEX, 2.2, _PINK_LIGHT, 3, cv2.LINE_AA,
        )

        # ── "Final Affection Score: X / 100" ─────────────────────────
        line2 = f"Final Affection Score:  {int(score)} / 100"
        (tw2, th2), _ = cv2.getTextSize(
            line2, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2
        )
        cv2.putText(
            result, line2,
            (cx - tw2 // 2, cy - 10),
            cv2.FONT_HERSHEY_DUPLEX, 1.4, _WHITE, 2, cv2.LINE_AA,
        )
        # Decorative underline
        cv2.line(
            result,
            (cx - tw2 // 2, cy + 6),
            (cx + tw2 // 2, cy + 6),
            _PINK_HOT, 2,
        )

        # ── Conditional sub-text ──────────────────────────────────────
        if score > 80:
            sub       = "It's a Match!  <3"
            sub_color = _PINK_LIGHT
        elif score >= 50:
            sub       = "Definitely Something There..."
            sub_color = (210, 200, 255)
        else:
            sub       = "Let's Just Be Friends..."
            sub_color = (160, 150, 200)

        (tw3, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)
        cv2.putText(
            result, sub,
            (cx - tw3 // 2, cy + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.05, sub_color, 2, cv2.LINE_AA,
        )

        # ── Heart gauge bar ───────────────────────────────────────────
        gw, gh = 520, 26
        gx = cx - gw // 2
        gy = cy + 100
        fill = int(gw * score / 100.0)
        # Track
        cv2.rectangle(result, (gx, gy), (gx + gw, gy + gh), (30, 10, 40), -1)
        # Fill (hot pink)
        if fill > 0:
            cv2.rectangle(result, (gx, gy), (gx + fill, gy + gh), _PINK_HOT, -1)
        # White border
        cv2.rectangle(result, (gx, gy), (gx + gw, gy + gh), _WHITE, 2)
        # Score % label inside bar
        pct = f"{int(score)}%"
        (ptw, _), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            result, pct,
            (gx + gw // 2 - ptw // 2, gy + gh - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA,
        )

        # ── "Press any key to exit" hint ──────────────────────────────
        hint = "Press any key to exit"
        (twh, _), _ = cv2.getTextSize(
            hint, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        )
        cv2.putText(
            result, hint,
            (cx - twh // 2, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, _GRAY_DIM, 1, cv2.LINE_AA,
        )

        cv2.imshow("Heart CV-gnal", result)
        cv2.waitKey(0)   # hold until presenter presses any key

    # ------------------------------------------------------------------
    # Render: calibration screen  (pink theme)
    # ------------------------------------------------------------------

    def _render_calibrating(
        self, frame: np.ndarray, output: AffectionOutput
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Dark pink vignette
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (25, 8, 35), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        cx, cy = w // 2, h // 2

        cv2.putText(
            frame, "HEART  CV-GNAL",
            (cx - 220, cy - 95),
            cv2.FONT_HERSHEY_DUPLEX, 1.7, _PINK_LIGHT, 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "Affection / Interest Analyzer",
            (cx - 180, cy - 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, _PINK_MED, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "Look straight at the camera to calibrate your baseline.",
            (cx - 255, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 175, 220), 1, cv2.LINE_AA,
        )

        # Pink progress bar with white border
        bar_w, bar_h = 440, 24
        bx = cx - bar_w // 2
        by = cy + 22
        fill = int(bar_w * output.calib_progress)

        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (25, 8, 35), -1)
        if fill > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), _PINK_MED, -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), _WHITE, 1)

        pct = f"{int(output.calib_progress * 100)}%"
        (ptw, _), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(
            frame, pct,
            (cx - ptw // 2, by + bar_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, _WHITE, 1, cv2.LINE_AA,
        )

        cv2.putText(
            frame, "Q / ESC to quit",
            (cx - 62, cy + 74),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GRAY_DIM, 1, cv2.LINE_AA,
        )
        return frame

    # ------------------------------------------------------------------
    # Render: right-side score panel  (pink theme)
    # ------------------------------------------------------------------

    def _render_score_panel(
        self,
        frame: np.ndarray,
        features: FrameFeatures,
        output: AffectionOutput,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        pw = _PANEL_W
        px = w - pw - _PANEL_PAD
        py = _PANEL_PAD
        ph = h - _PANEL_PAD * 2

        # Dark aubergine panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 5, py), (px + pw, py + ph), _PINK_PANEL, -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        cv2.rectangle(frame, (px - 5, py), (px + pw, py + ph), _PINK_BORDER, 1)

        score = self._blended_score   # blend of rule-based + VLM (falls back to rule-only)

        # ── Header ────────────────────────────────────────────────────
        cv2.putText(
            frame, "HEART  GAUGE",
            (px + 4, py + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, _PINK_MED, 1, cv2.LINE_AA,
        )
        # Thin separator under header
        cv2.line(frame, (px, py + 30), (px + pw, py + 30), _PINK_BORDER, 1)

        # ── Big score number ───────────────────────────────────────────
        score_str = f"{int(score):3d}"
        cv2.putText(
            frame, score_str,
            (px + 8, py + 95),
            cv2.FONT_HERSHEY_DUPLEX, 2.9, _PINK_LIGHT, 3, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "/ 100",
            (px + 162, py + 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, _PINK_MED, 1, cv2.LINE_AA,
        )

        # ── Mood label ────────────────────────────────────────────────
        cv2.putText(
            frame, _mood_label(score),
            (px + 4, py + 114),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _PINK_LIGHT, 1, cv2.LINE_AA,
        )

        # ── Heart Gauge bar: filled hot-pink rectangle + white border ─
        gx   = px + 4
        gy   = py + 126
        gw   = pw - 8
        gh   = 24
        fill = int(gw * score / 100.0)

        # Track (very dark)
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (22, 8, 30), -1)
        # Fill (hot pink — solid, no gradient)
        if fill > 0:
            cv2.rectangle(frame, (gx, gy), (gx + fill, gy + gh), _PINK_HOT, -1)
        # White border (aesthetic)
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), _WHITE, 1)

        # ── Divider ───────────────────────────────────────────────────
        dy0 = py + 162
        cv2.line(frame, (px, dy0), (px + pw, dy0), _PINK_BORDER, 1)

        # ── Metric readouts ───────────────────────────────────────────
        def _row(
            label: str, val_str: str, sub: str, y: int,
            val_color: tuple = _PINK_LIGHT,
        ) -> None:
            cv2.putText(frame, label,   (px + 4,   y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, _PINK_MED,  1, cv2.LINE_AA)
            cv2.putText(frame, val_str, (px + 82,  y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, val_color,   1, cv2.LINE_AA)
            if sub:
                cv2.putText(frame, sub, (px + 130, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, _GRAY_DIM, 1, cv2.LINE_AA)

        if features.face_detected:
            _row("Mouth ratio",
                 f"{features.mouth_ratio:.3f}",
                 f"(b {output.baseline_mouth:.3f})",
                 py + 180)
            _row("Eye ratio",
                 f"{features.eye_ratio:.3f}",
                 f"(b {output.baseline_eye:.3f})",
                 py + 198)
            # Head pose angles (from solvePnP) — penalty triggers above ±20°/±15°
            yaw_color   = _PINK_HOT if abs(features.yaw_deg)   > 20 else _PINK_LIGHT
            pitch_color = _PINK_HOT if abs(features.pitch_deg) > 15 else _PINK_LIGHT
            _row("Head Yaw",
                 f"{features.yaw_deg:+.1f}°", "",
                 py + 216, val_color=yaw_color)
            _row("Head Pitch",
                 f"{features.pitch_deg:+.1f}°", "",
                 py + 234, val_color=pitch_color)
        else:
            cv2.putText(frame, "No face detected", (px + 4, py + 198),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GRAY_DIM, 1, cv2.LINE_AA)

        if features.pose_detected:
            _row("Shoulder",
                 f"{features.shoulder_ratio:.3f}",
                 f"(b {output.baseline_shoulder:.3f})",
                 py + 254)
            arm_val   = "CROSSED" if features.wrists_crossed else "free"
            arm_color = _PINK_HOT if features.wrists_crossed else (90, 200, 120)
            _row("Arms", arm_val, "", py + 272, val_color=arm_color)
        else:
            cv2.putText(frame, "No pose detected", (px + 4, py + 262),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GRAY_DIM, 1, cv2.LINE_AA)

        # ── Divider ───────────────────────────────────────────────────
        dy1 = py + 290
        cv2.line(frame, (px, dy1), (px + pw, dy1), _PINK_BORDER, 1)

        # ── Event log (last 4 events) ─────────────────────────────────
        cv2.putText(frame, "Recent events", (px + 4, dy1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, _PINK_MED, 1, cv2.LINE_AA)
        for i, evt in enumerate(output.event_log[:4]):
            alpha = max(0.35, 1.0 - i * 0.22)
            col   = tuple(int(c * alpha) for c in _PINK_LIGHT)  # type: ignore[misc]
            cv2.putText(frame, evt, (px + 4, dy1 + 34 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

        # ── VLM cross-validator panel ──────────────────────────────────
        dy2 = dy1 + 106   # fixed offset — enough room for 4 event-log rows
        cv2.line(frame, (px, dy2), (px + pw, dy2), _PINK_BORDER, 1)

        ts_now = time.monotonic()
        vlm    = self._vlm.get_result()

        if not self._vlm.available:
            # Module disabled or anthropic not installed — one-liner hint
            cv2.putText(frame, "VLM  offline", (px + 4, dy2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, _GRAY_DIM, 1, cv2.LINE_AA)

        elif vlm is None:
            # Available but waiting for first response
            cv2.putText(frame, "VLM  analyzing...", (px + 4, dy2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, _PINK_MED, 1, cv2.LINE_AA)

        else:
            age_s = int(ts_now - vlm.timestamp)

            # Row 1: section title + age
            cv2.putText(frame, "VLM OPINION", (px + 4, dy2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, _PINK_MED, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{age_s}s ago", (px + 170, dy2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, _GRAY_DIM, 1, cv2.LINE_AA)

            # Row 2: mini score bar
            vgx, vgy = px + 4, dy2 + 20
            vgw, vgh = pw - 8, 10
            vfill    = int(vgw * vlm.score / 100.0)
            cv2.rectangle(frame, (vgx, vgy), (vgx + vgw, vgy + vgh), (22, 8, 30), -1)
            if vfill > 0:
                cv2.rectangle(frame, (vgx, vgy), (vgx + vfill, vgy + vgh), _PINK_MED, -1)
            cv2.rectangle(frame, (vgx, vgy), (vgx + vgw, vgy + vgh), _WHITE, 1)

            # Row 3: numeric score + confidence
            conf_col = (90, 200, 120) if vlm.confidence == "high" \
                       else _PINK_MED if vlm.confidence == "medium" \
                       else _GRAY_DIM
            cv2.putText(frame, f"{int(vlm.score):3d}", (px + 4, dy2 + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, _PINK_LIGHT, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{vlm.confidence} conf.", (px + 48, dy2 + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, conf_col, 1, cv2.LINE_AA)

            # Row 4: agreement indicator (compare raw rule score vs VLM)
            delta      = abs(output.score - vlm.score)
            agree_text = "Systems Agree  v" if delta < 15 else "Conflicting  !"
            agree_col  = (90, 200, 120) if delta < 15 else _PINK_HOT
            cv2.putText(frame, agree_text, (px + 4, dy2 + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, agree_col, 1, cv2.LINE_AA)

            # Row 5: reasoning text (truncated to fit panel width)
            cv2.putText(frame, f'"{vlm.reasoning[:30]}"', (px + 4, dy2 + 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.29, (180, 165, 200), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Render: event popup banner  (top-centre, fades over _EVENT_DISPLAY)
    # ------------------------------------------------------------------

    def _render_event_banner(
        self,
        frame: np.ndarray,
        output: AffectionOutput,
        ts: float,
    ) -> np.ndarray:
        age = ts - output.last_event_time
        if age > _EVENT_DISPLAY or not output.last_event_text:
            return frame

        text = output.last_event_text
        if text.startswith("+"):
            bg = _PINK_HOT          # hot pink → positive points
            fg = _WHITE
        elif text.startswith("-"):
            bg = (80, 20, 100)      # dark purple → penalty
            fg = _PINK_LIGHT
        else:
            bg = (55, 35, 70)
            fg = (220, 200, 240)

        h, w = frame.shape[:2]
        font_scale = 0.95
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
        bx = (w - tw) // 2 - 22
        by = 16
        bw = tw + 44
        bh = th + 24

        # Pill background with fade-out alpha
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), bg, -1)
        alpha_val = max(0.0, 1.0 - (age / _EVENT_DISPLAY) ** 1.5)
        cv2.addWeighted(overlay, alpha_val * 0.92, frame, 1 - alpha_val * 0.92, 0, frame)

        # White border
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), _WHITE, 1)

        # Text
        cv2.putText(
            frame, text,
            ((w - tw) // 2, by + th + 12),
            cv2.FONT_HERSHEY_DUPLEX, font_scale, fg, 2, cv2.LINE_AA,
        )
        return frame

    # ------------------------------------------------------------------
    # Render: demo countdown timer  (top-left)
    # ------------------------------------------------------------------

    def _render_timer(
        self, frame: np.ndarray, remaining: float, now: float
    ) -> np.ndarray:
        mins = max(0, int(remaining)) // 60
        secs = max(0, int(remaining)) % 60
        timer_text = f"DEMO  {mins}:{secs:02d}"

        # Urgent colour when < 60 s remain
        color = _PINK_HOT if remaining < 60 else _PINK_MED

        # Pill background for legibility
        (tw, th), _ = cv2.getTextSize(
            timer_text, cv2.FONT_HERSHEY_DUPLEX, 0.68, 1
        )
        tx, ty = _PANEL_PAD, 58
        pad = 5
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (tx - pad, ty - th - pad),
            (tx + tw + pad, ty + pad),
            _PINK_PANEL, -1,
        )
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(
            frame,
            (tx - pad, ty - th - pad),
            (tx + tw + pad, ty + pad),
            color, 1,
        )
        cv2.putText(
            frame, timer_text,
            (tx, ty),
            cv2.FONT_HERSHEY_DUPLEX, 0.68, color, 1, cv2.LINE_AA,
        )

        # Thin window-progress bar just below the timer pill
        # (shows how long until the next 5-second evaluation fires)
        if self._window_start is not None:
            elapsed = now - self._window_start
            ratio = min(1.0, elapsed / _EVAL_WINDOW)
            bar_x = tx - pad
            bar_y = ty + pad + 3
            bar_w = tw + pad * 2
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 4),
                          (25, 8, 35), -1)
            if ratio > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + int(bar_w * ratio), bar_y + 4),
                    _PINK_MED, -1,
                )

        return frame

    # ------------------------------------------------------------------
    # Render: detection status dots  (top-left)
    # ------------------------------------------------------------------

    def _render_status_dots(
        self, frame: np.ndarray, features: FrameFeatures
    ) -> np.ndarray:
        items = [("Face", features.face_detected), ("Pose", features.pose_detected)]
        for i, (label, ok) in enumerate(items):
            color = (80, 200, 120) if ok else _PINK_HOT
            x = _PANEL_PAD + i * 90
            y = _PANEL_PAD + 14
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.putText(frame, label, (x + 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, _PINK_MED, 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    # Render: active-signal badges  (left column)
    # ------------------------------------------------------------------

    def _render_active_signals(
        self, frame: np.ndarray, output: AffectionOutput
    ) -> np.ndarray:
        signals: list[tuple[str, tuple]] = []
        if output.is_leaning:
            signals.append(("LEAN IN",    (90, 210, 130)))
        if output.is_looking_away:
            signals.append(("LOOK AWAY",  _PINK_HOT))
        if output.is_barrier:
            signals.append(("BARRIER",    _PINK_HOT))

        for i, (sig, col) in enumerate(signals):
            cv2.putText(
                frame, f">> {sig}",
                (_PANEL_PAD, _PANEL_PAD + 40 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1, cv2.LINE_AA,
            )
        return frame

    # ------------------------------------------------------------------
    # FPS counter  (bottom-left, subtle)
    # ------------------------------------------------------------------

    _prev_ts: float = 0.0

    def _draw_fps(self, frame: np.ndarray, ts: float) -> None:
        dt = ts - self._prev_ts
        if dt > 0:
            fps = 1.0 / dt
            h   = frame.shape[0]
            cv2.putText(
                frame, f"{fps:.0f} fps",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GRAY_DIM, 1, cv2.LINE_AA,
            )
        self._prev_ts = ts

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        cap = getattr(self, "_cap", None)
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

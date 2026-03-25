"""
Heart CV-gnal — Affection Score Engine
========================================
Rule-based evaluation of four psychologically-grounded signals.

  Signal              | Direction | Trigger
  --------------------|-----------|--------------------------------------------
  Duchenne Smile      | +5 pts    | mouth_ratio spikes AND eye_ratio drops
  Leaning In          | +4 pts    | shoulder_ratio > 1.15 × baseline (≥50% of window)
  Synchrony           | +3 pts    | sustained head tilt (5-25°, ≥30% of window)
  Nodding             | +2 pts    | high nose-y variance across window
  Barrier Signal      | -5 pts    | both wrists crossed ≥50% of window

Scoring runs in TWO phases:
  1. Per-frame  — used ONLY during the 3-second calibration phase (``update()``)
  2. Per-window — called by the runner every 5 s after calibration (``batch_evaluate()``)

Global score initialises at 50 and is clamped to [0, 100].

Usage
-----
    engine = AffectionEngine()
    # calibration (call every frame until output.calibrated is True)
    output = engine.update(frame_features, ts)

    # evaluation (call every EVAL_WINDOW seconds with the buffered list)
    output = engine.batch_evaluate(buffered_features, window_elapsed_secs, ts)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .feature_extractor import FrameFeatures


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class AffectionOutput:
    """Snapshot of the engine's state returned after each update."""

    score: float
    """Current affection score, always in [0, 100]."""

    # ── Calibration ──────────────────────────────────────────────────────
    calibrated: bool
    calib_progress: float
    """Fraction of the calibration window elapsed, 0.0 → 1.0."""

    # ── Baselines (published so the UI can show deltas) ──────────────
    baseline_mouth:    float
    baseline_eye:      float
    baseline_shoulder: float

    # ── Active-signal flags (updated every batch window) ─────────────
    is_leaning:  bool
    is_barrier:  bool
    is_syncing:  bool

    # ── Event notification (drives the popup banner) ──────────────────
    last_event_text: str
    last_event_time: float

    # ── Rolling event log ─────────────────────────────────────────────
    event_log: List[str]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AffectionEngine:
    """
    Stateful affection scorer.

    * Call ``update()`` every frame **during calibration only**.
    * Call ``batch_evaluate()`` every EVAL_WINDOW seconds **after calibration**.
    """

    # ── Calibration ──────────────────────────────────────────────────────
    CALIB_SECS: float = 3.0

    # ── Rule 1: Duchenne Smile ────────────────────────────────────────────
    MOUTH_SPIKE_FACTOR: float = 1.12
    EYE_SQUINT_FACTOR:  float = 0.90
    DUCHENNE_DELTA:     float = 5.0
    DUCHENNE_COOLDOWN:  float = 2.5   # seconds between awards
    DUCHENNE_THRESHOLD: float = 0.25  # min fraction of face-frames showing smile

    # ── Rule 2: Leaning In ───────────────────────────────────────────────
    LEAN_FACTOR:      float = 1.15
    LEAN_PTS:         float = 4.0    # awarded when ≥50 % of pose-frames lean
    LEAN_THRESHOLD:   float = 0.50

    # ── Rule 3: Synchrony ────────────────────────────────────────────────
    SYNC_TILT_MIN_DEG:  float = 5.0
    SYNC_TILT_MAX_DEG:  float = 25.0
    SYNC_PTS:           float = 3.0   # awarded when ≥30 % of face-frames tilt
    SYNC_THRESHOLD:     float = 0.30
    NOD_PTS:            float = 2.0   # awarded for detected nodding
    NODDING_STD:        float = 0.007

    # ── Rule 4: Barrier Signal ───────────────────────────────────────────
    BARRIER_THRESHOLD:  float = 0.50   # min fraction of pose-frames crossed
    BARRIER_PTS:        float = 5.0    # deducted when threshold met

    # ── Score bounds ─────────────────────────────────────────────────────
    SCORE_MIN: float = 0.0
    SCORE_MAX: float = 100.0

    # ── Misc ─────────────────────────────────────────────────────────────
    EVENT_LOG_SIZE: int = 8

    def __init__(self) -> None:
        # ── Calibration state ────────────────────────────────────────────
        self._start_time:  Optional[float] = None
        self._calib_buf:   list[FrameFeatures] = []

        self._calibrated:         bool  = False
        self._baseline_mouth:     float = 0.40
        self._baseline_eye:       float = 0.06
        self._baseline_shoulder:  float = 0.30

        # ── Score ────────────────────────────────────────────────────────
        self._score: float = 50.0

        # ── Per-batch signal state (updated by batch_evaluate) ───────────
        self._duchenne_last: float = 0.0
        self._was_leaning:   bool  = False
        self._is_syncing:    bool  = False
        self._is_barrier:    bool  = False

        # ── Events ───────────────────────────────────────────────────────
        self._last_event_text: str   = ""
        self._last_event_time: float = 0.0
        self._event_log: deque[str]  = deque(maxlen=self.EVENT_LOG_SIZE)

    # ------------------------------------------------------------------
    # Public API — calibration phase (call every frame)
    # ------------------------------------------------------------------

    def update(self, features: FrameFeatures, ts: float) -> AffectionOutput:
        """
        Process one frame during the calibration phase.
        After ``output.calibrated`` is True the runner should switch to
        ``batch_evaluate()`` and stop calling this method.
        """
        if self._start_time is None:
            self._start_time = ts

        if not self._calibrated:
            self._calibrate(features, ts)

        return self._build_output(ts)

    # ------------------------------------------------------------------
    # Public API — evaluation phase (call every EVAL_WINDOW seconds)
    # ------------------------------------------------------------------

    def batch_evaluate(
        self,
        features_list: list[FrameFeatures],
        window_elapsed: float,
        ts: float,
    ) -> AffectionOutput:
        """
        One-shot evaluation of features buffered over a time window.

        Args:
            features_list   : FrameFeatures accumulated since the last call.
            window_elapsed  : Actual duration of the window in seconds.
            ts              : Current ``time.monotonic()`` timestamp.

        Returns:
            AffectionOutput snapshot with the updated score.
        """
        if not self._calibrated or not features_list:
            return self._build_output(ts)

        face_frames = [f for f in features_list if f.face_detected]
        pose_frames = [f for f in features_list if f.pose_detected]
        n_face = len(face_frames)
        n_pose = len(pose_frames)

        # Collect (delta, banner_text) for each triggered rule
        pos_events: list[tuple[float, str]] = []
        neg_events: list[tuple[float, str]] = []

        # ── Rule 1: Duchenne Smile ─────────────────────────────────────
        if n_face > 0 and (ts - self._duchenne_last) >= self.DUCHENNE_COOLDOWN:
            duchenne_n = sum(
                1 for f in face_frames
                if (f.mouth_ratio > self._baseline_mouth * self.MOUTH_SPIKE_FACTOR
                    and f.eye_ratio  < self._baseline_eye  * self.EYE_SQUINT_FACTOR)
            )
            if duchenne_n / n_face >= self.DUCHENNE_THRESHOLD:
                d = self.DUCHENNE_DELTA
                self._score += d
                self._duchenne_last = ts
                pos_events.append((d, f"+{int(d)} pts! Genuine Smile!"))

        # ── Rule 2: Leaning In ────────────────────────────────────────
        if n_pose > 0:
            lean_n = sum(
                1 for f in pose_frames
                if f.shoulder_ratio > self._baseline_shoulder * self.LEAN_FACTOR
            )
            lean_ratio = lean_n / n_pose
            self._was_leaning = lean_ratio >= self.LEAN_THRESHOLD
            if self._was_leaning:
                d = self.LEAN_PTS
                self._score += d
                pos_events.append((d, f"+{int(d)} pts! Leaning In!"))
        else:
            self._was_leaning = False

        # ── Rule 3: Synchrony (head tilt OR nodding) ──────────────────
        if n_face > 0:
            tilt_n = sum(
                1 for f in face_frames
                if self.SYNC_TILT_MIN_DEG <= abs(f.roll_angle_deg) <= self.SYNC_TILT_MAX_DEG
            )
            tilt_ratio = tilt_n / n_face
            nose_ys    = [f.nose_y for f in face_frames]
            nodding    = (
                len(nose_ys) > 5
                and float(np.std(nose_ys)) > self.NODDING_STD
            )
            self._is_syncing = tilt_ratio >= self.SYNC_THRESHOLD or nodding

            if tilt_ratio >= self.SYNC_THRESHOLD:
                d = self.SYNC_PTS
                self._score += d
                pos_events.append((d, f"+{int(d)} pts! Great Eye Contact!"))
            elif nodding:
                d = self.NOD_PTS
                self._score += d
                pos_events.append((d, f"+{int(d)} pts! Nodding Along!"))
        else:
            self._is_syncing = False

        # ── Rule 4: Barrier Signal ────────────────────────────────────
        if n_pose > 0:
            barrier_n     = sum(1 for f in pose_frames if f.wrists_crossed)
            barrier_ratio = barrier_n / n_pose
            self._is_barrier = barrier_ratio >= self.BARRIER_THRESHOLD
            if self._is_barrier:
                d = self.BARRIER_PTS
                self._score -= d
                neg_events.append((d, f"-{int(d)} pts! Barrier Signal!"))
        else:
            self._is_barrier = False

        # ── Clamp score ───────────────────────────────────────────────
        self._score = max(self.SCORE_MIN, min(self.SCORE_MAX, self._score))

        # ── Log all events; set banner to most impactful positive ─────
        all_text = [t for _, t in pos_events] + [t for _, t in neg_events]
        for text in all_text:
            self._event_log.appendleft(text)

        if pos_events:
            best_text = max(pos_events, key=lambda x: x[0])[1]
            self._last_event_text = best_text
            self._last_event_time = ts
        elif neg_events:
            self._last_event_text = neg_events[0][1]
            self._last_event_time = ts

        return self._build_output(ts)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibrate(self, features: FrameFeatures, ts: float) -> None:
        if features.face_detected and features.pose_detected:
            self._calib_buf.append(features)

        elapsed = ts - self._start_time  # type: ignore[operator]
        if elapsed >= self.CALIB_SECS:
            buf = self._calib_buf
            if buf:
                self._baseline_mouth    = float(np.mean([f.mouth_ratio    for f in buf]))
                self._baseline_eye      = float(np.mean([f.eye_ratio      for f in buf]))
                self._baseline_shoulder = float(np.mean([f.shoulder_ratio for f in buf]))
            self._calibrated = True
            self._calib_buf.clear()
            self._last_event_text = "Baseline locked — go!"
            self._last_event_time = ts
            self._event_log.appendleft("Baseline locked — go!")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_output(self, ts: float) -> AffectionOutput:
        calib_progress = 0.0
        if not self._calibrated and self._start_time is not None:
            calib_progress = min(1.0, (ts - self._start_time) / self.CALIB_SECS)

        return AffectionOutput(
            score              = self._score,
            calibrated         = self._calibrated,
            calib_progress     = calib_progress,
            baseline_mouth     = self._baseline_mouth,
            baseline_eye       = self._baseline_eye,
            baseline_shoulder  = self._baseline_shoulder,
            is_leaning         = self._was_leaning,
            is_barrier         = self._is_barrier,
            is_syncing         = self._is_syncing,
            last_event_text    = self._last_event_text,
            last_event_time    = self._last_event_time,
            event_log          = list(self._event_log),
        )

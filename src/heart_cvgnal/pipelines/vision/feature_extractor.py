"""
Heart CV-gnal — Feature Extractor
==================================
Converts raw MediaPipe Holistic results into distance-invariant ratios and
angles.  All face metrics are normalised by the inter-cheek distance so they
remain stable across different camera distances.

Extracted features
------------------
mouth_ratio      : mouth_width  / cheek_width
eye_ratio        : avg_eye_height / cheek_width
shoulder_ratio   : normalised shoulder-to-shoulder distance (proxy for lean-in)
roll_angle_deg   : arctan(Δy / Δx) of the eye-corner line  (head tilt)
nose_y           : normalised nose-tip y — used externally as a head-pitch proxy
wrists_crossed   : True when each wrist passes the opposite shoulder x-position
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# FaceMesh landmark indices  (MediaPipe 468-point canonical model)
# All indices are from the PERSON's anatomical perspective.
# Because the app processes the horizontally-flipped webcam frame, what
# MediaPipe calls "left" appears on the LEFT side of the displayed image.
# ---------------------------------------------------------------------------
_CHEEK_L   = 234   # face-oval left cheek outer edge
_CHEEK_R   = 454   # face-oval right cheek outer edge
_MOUTH_L   = 61    # left mouth corner
_MOUTH_R   = 291   # right mouth corner
_L_EYE_UP  = 159   # left eye upper eyelid
_L_EYE_LO  = 145   # left eye lower eyelid
_R_EYE_UP  = 386   # right eye upper eyelid
_R_EYE_LO  = 374   # right eye lower eyelid
_EYE_L_OUT = 33    # left eye outer corner  (roll anchor A)
_EYE_R_OUT = 263   # right eye outer corner (roll anchor B)
_NOSE_TIP  = 4     # nose tip (head-pitch / nodding proxy)

# ---------------------------------------------------------------------------
# Pose landmark indices  (MediaPipe 33-point model)
# ---------------------------------------------------------------------------
_SHOULDER_L = 11
_SHOULDER_R = 12
_WRIST_L    = 15
_WRIST_R    = 16

_VIS_THRESH = 0.40   # minimum MediaPipe visibility score to trust a landmark


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class FrameFeatures:
    """
    Distance-invariant ratios and angles extracted per video frame.
    All values are in normalised units unless the field name ends in ``_deg``.
    """

    # ── Face signals (from FaceMesh via Holistic) ──────────────────────────
    mouth_ratio: float = 0.0
    """mouth_width / cheek_width  — increases with smiling"""

    eye_ratio: float = 0.0
    """avg_eye_height / cheek_width  — decreases with Duchenne squinting"""

    roll_angle_deg: float = 0.0
    """Eye-line tilt: arctan2(Δy, Δx) of the outer eye-corner vector.
    Positive = head tilted toward person's right; negative = toward left."""

    nose_y: float = 0.5
    """Normalised nose-tip y coordinate [0 = top, 1 = bottom].
    Used externally to detect head-nodding via variance over time."""

    # ── Pose signals (from Pose landmarks via Holistic) ────────────────────
    shoulder_ratio: float = 0.0
    """Normalised shoulder-to-shoulder distance.
    Increases when the subject physically leans closer to the camera."""

    wrists_crossed: bool = False
    """True when each wrist has passed the opposite shoulder's x-coordinate,
    indicating a defensive arm-cross (barrier) posture."""

    # ── Validity flags ─────────────────────────────────────────────────────
    face_detected: bool = False
    pose_detected: bool = False


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Stateless transformer: MediaPipe Holistic results → FrameFeatures.

    Usage::

        extractor = FeatureExtractor()
        features  = extractor.extract(holistic_results)
    """

    def extract(self, holistic_results) -> FrameFeatures:
        """
        Args:
            holistic_results: The object returned by
                ``mp.solutions.holistic.Holistic.process(image_rgb)``.
        Returns:
            FrameFeatures populated with as many values as the landmarks allow.
        """
        feats = FrameFeatures()
        feats.face_detected = self._extract_face(holistic_results.face_landmarks, feats)
        feats.pose_detected = self._extract_pose(holistic_results.pose_landmarks, feats)
        return feats

    # ------------------------------------------------------------------
    # Private — face extraction
    # ------------------------------------------------------------------

    def _extract_face(self, face_lm, out: FrameFeatures) -> bool:
        if face_lm is None or len(face_lm.landmark) < 468:
            return False

        lm = face_lm.landmark

        # Reference distance: inter-cheek width
        cheek_w = _dist2d(lm[_CHEEK_L], lm[_CHEEK_R])
        if cheek_w < 1e-6:
            return False

        # ── Mouth Ratio ──────────────────────────────────────────────────
        out.mouth_ratio = _dist2d(lm[_MOUTH_L], lm[_MOUTH_R]) / cheek_w

        # ── Eye Ratio ────────────────────────────────────────────────────
        l_h = abs(lm[_L_EYE_UP].y - lm[_L_EYE_LO].y)
        r_h = abs(lm[_R_EYE_UP].y - lm[_R_EYE_LO].y)
        out.eye_ratio = ((l_h + r_h) / 2.0) / cheek_w

        # ── Roll Angle ───────────────────────────────────────────────────
        # Vector from left eye outer-corner → right eye outer-corner.
        # In the mirrored, processed frame: _EYE_L_OUT (33) is on the LEFT
        # and _EYE_R_OUT (263) is on the RIGHT, so dx is typically positive.
        dx = lm[_EYE_R_OUT].x - lm[_EYE_L_OUT].x
        dy = lm[_EYE_R_OUT].y - lm[_EYE_L_OUT].y
        out.roll_angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        # ── Nose y (head-pitch proxy) ────────────────────────────────────
        out.nose_y = float(lm[_NOSE_TIP].y)

        return True

    # ------------------------------------------------------------------
    # Private — pose extraction
    # ------------------------------------------------------------------

    def _extract_pose(self, pose_lm, out: FrameFeatures) -> bool:
        if pose_lm is None:
            return False

        lm = pose_lm.landmark

        # Require both shoulders to be reasonably visible
        if (lm[_SHOULDER_L].visibility < _VIS_THRESH or
                lm[_SHOULDER_R].visibility < _VIS_THRESH):
            return False

        # ── Shoulder Ratio ────────────────────────────────────────────────
        # Normalised coords [0,1] → ratio represents relative screen span.
        # Leaning toward the camera → subject appears larger → ratio grows.
        out.shoulder_ratio = _dist2d(lm[_SHOULDER_L], lm[_SHOULDER_R])

        # ── Wrists Crossed ───────────────────────────────────────────────
        # 왼손목 x > 오른손목 x 이면 손목이 엇갈린 것 → 팔짱 낀 상태.
        # 가시성 미달 시 반대편으로 spurious trigger 방지.
        ls_x = lm[_SHOULDER_L].x
        rs_x = lm[_SHOULDER_R].x

        lw_x = (lm[_WRIST_L].x if lm[_WRIST_L].visibility > _VIS_THRESH
                else ls_x)
        rw_x = (lm[_WRIST_R].x if lm[_WRIST_R].visibility > _VIS_THRESH
                else rs_x)

        out.wrists_crossed = lw_x < rw_x

        return True


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _dist2d(a, b) -> float:
    """Euclidean distance between two MediaPipe NormalizedLandmark objects."""
    return float(np.hypot(a.x - b.x, a.y - b.y))

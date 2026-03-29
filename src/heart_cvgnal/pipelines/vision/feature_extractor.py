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
nose_y           : normalised nose-tip y — kept for legacy compatibility
yaw_deg          : head yaw angle in degrees from solvePnP (left/right turn)
pitch_deg        : head pitch angle in degrees from solvePnP (up/down tilt)
wrists_crossed   : True when each wrist passes the opposite shoulder x-position
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
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
# Head-pose solvePnP landmark indices  (6-point subset of FaceMesh)
# ---------------------------------------------------------------------------
_HP_NOSE_TIP  = 1    # nose bridge tip  (slightly above _NOSE_TIP for stability)
_HP_CHIN      = 152  # chin bottom
_HP_EYE_L_OUT = 33   # left  eye outer corner
_HP_EYE_R_OUT = 263  # right eye outer corner
_HP_MOUTH_L   = 61   # left  mouth corner
_HP_MOUTH_R   = 291  # right mouth corner

# Generic 3-D face model in mm (standard academic reference geometry)
# Origin = nose tip; face looks toward -Z.
_MODEL_3D = np.array([
    [  0.0,    0.0,    0.0],   # nose tip
    [  0.0, -330.0,  -65.0],   # chin
    [-225.0,  170.0, -135.0],   # left  eye outer corner
    [ 225.0,  170.0, -135.0],   # right eye outer corner
    [-150.0, -150.0, -125.0],   # left  mouth corner
    [ 150.0, -150.0, -125.0],   # right mouth corner
], dtype=np.float64)

_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)   # assume no lens distortion

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
    """Normalised nose-tip y coordinate [0 = top, 1 = bottom]."""

    yaw_deg: float = 0.0
    """Head yaw angle in degrees from solvePnP.
    Positive = face turned to its right; negative = face turned to its left.
    Absolute value > ~20° means the subject is looking away horizontally."""

    pitch_deg: float = 0.0
    """Head pitch angle in degrees from solvePnP.
    Positive = face tilted up; negative = face tilted down.
    Absolute value > ~15° means the subject is looking away vertically."""

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

    def extract(
        self,
        holistic_results,
        frame_w: int = 1280,
        frame_h: int = 720,
    ) -> FrameFeatures:
        """
        Args:
            holistic_results: The object returned by
                ``mp.solutions.holistic.Holistic.process(image_rgb)``.
            frame_w: Frame pixel width  (used for solvePnP camera matrix).
            frame_h: Frame pixel height (used for solvePnP camera matrix).
        Returns:
            FrameFeatures populated with as many values as the landmarks allow.
        """
        feats = FrameFeatures()
        feats.face_detected = self._extract_face(
            holistic_results.face_landmarks, feats, frame_w, frame_h
        )
        feats.pose_detected = self._extract_pose(holistic_results.pose_landmarks, feats)
        return feats

    # ------------------------------------------------------------------
    # Private — face extraction
    # ------------------------------------------------------------------

    def _extract_face(
        self,
        face_lm,
        out: FrameFeatures,
        frame_w: int,
        frame_h: int,
    ) -> bool:
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
        dx = lm[_EYE_R_OUT].x - lm[_EYE_L_OUT].x
        dy = lm[_EYE_R_OUT].y - lm[_EYE_L_OUT].y
        out.roll_angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        # ── Nose y ───────────────────────────────────────────────────────
        out.nose_y = float(lm[_NOSE_TIP].y)

        # ── Head Pose: Yaw & Pitch via solvePnP ─────────────────────────
        self._compute_head_pose(lm, out, frame_w, frame_h)

        return True

    # ------------------------------------------------------------------
    # Private — stable head pose via solvePnP + RQDecomp3x3
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_head_pose(
        lm,
        out: FrameFeatures,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """
        Estimate Yaw and Pitch from 6 FaceMesh landmarks using solvePnP.

        The focal length is approximated as frame_w (a reasonable heuristic
        for a typical webcam with a ~60° horizontal FOV).  No distortion is
        assumed.  Euler angles are extracted via RQDecomp3x3 which decomposes
        the rotation matrix stably into pitch (X), yaw (Y), roll (Z).
        """
        focal = float(frame_w)
        cx    = frame_w / 2.0
        cy    = frame_h / 2.0
        cam_matrix = np.array(
            [[focal, 0.0,   cx ],
             [0.0,   focal, cy ],
             [0.0,   0.0,   1.0]],
            dtype=np.float64,
        )

        pts_2d = np.array([
            [lm[_HP_NOSE_TIP ].x * frame_w, lm[_HP_NOSE_TIP ].y * frame_h],
            [lm[_HP_CHIN     ].x * frame_w, lm[_HP_CHIN     ].y * frame_h],
            [lm[_HP_EYE_L_OUT].x * frame_w, lm[_HP_EYE_L_OUT].y * frame_h],
            [lm[_HP_EYE_R_OUT].x * frame_w, lm[_HP_EYE_R_OUT].y * frame_h],
            [lm[_HP_MOUTH_L  ].x * frame_w, lm[_HP_MOUTH_L  ].y * frame_h],
            [lm[_HP_MOUTH_R  ].x * frame_w, lm[_HP_MOUTH_R  ].y * frame_h],
        ], dtype=np.float64)

        success, rvec, _tvec = cv2.solvePnP(
            _MODEL_3D, pts_2d, cam_matrix, _DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return

        rmat, _ = cv2.Rodrigues(rvec)
        # RQDecomp3x3 returns angles in degrees: [pitch(X), yaw(Y), roll(Z)]
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        out.pitch_deg = float(angles[0])
        out.yaw_deg   = float(angles[1])

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
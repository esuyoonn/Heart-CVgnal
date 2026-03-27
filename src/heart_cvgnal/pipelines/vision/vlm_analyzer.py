"""
Heart CV-gnal — VLM Analyzer
==============================
Optional Claude Vision API module that cross-validates the rule-based affection
score with a zero-shot Vision-Language Model judgment.

Architecture
------------
* Runs in a **daemon thread** — never blocks the camera loop.
* Called via ``maybe_trigger(frame, now)`` every frame; internally rate-limited
  to ``INTERVAL`` seconds so the API is not hammered.
* Result is read back via ``get_result()`` (thread-safe, returns ``None`` until
  the first response arrives).

Fallback / kill-switch behaviour
---------------------------------
Failure mode                              → Result
``anthropic`` package not installed      → ``available = False``; all calls no-op
``ANTHROPIC_API_KEY`` not set / invalid  → ``available = False``; all calls no-op
Set ``VLMAnalyzer.ENABLED = False``      → same as above, instant
API error or timeout                     → warning logged; **old result kept**
Malformed JSON from model                → warning logged; **old result kept**

In every failure case the main pipeline continues unaffected.

Usage
-----
    vlm = VLMAnalyzer()                     # once, at app start-up

    # inside the frame loop (non-blocking):
    vlm.maybe_trigger(frame, time.time())

    result = vlm.get_result()               # Optional[VLMResult]
    if result:
        blended = 0.65 * rule_score + 0.35 * result.score
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt — kept tight so the model returns parseable JSON reliably
# ---------------------------------------------------------------------------
_PROMPT = """\
You are a body-language analysis system measuring social engagement and interest.

Look at this person's facial expression, posture, and visible body language.
Respond with ONLY valid JSON — no markdown, no explanation:

{
  "score": <integer 0-100, where 0 = completely disengaged, 100 = extremely interested>,
  "dominant_signal": "<one of: genuine_smile | neutral | disengaged | barrier | leaning_in | looking_away>",
  "confidence": "<one of: low | medium | high>",
  "reasoning": "<8 words max describing what you observed>"
}"""


# ---------------------------------------------------------------------------
# Result data model
# ---------------------------------------------------------------------------

@dataclass
class VLMResult:
    """A single snapshot judgment returned by the Vision API."""

    score: float
    """Engagement score 0–100, directly blendable with the rule-based score."""

    dominant_signal: str
    """Qualitative label for the strongest observed signal."""

    confidence: str
    """Model's self-reported confidence: 'low' | 'medium' | 'high'."""

    reasoning: str
    """Short human-readable phrase, shown in the UI event panel."""

    timestamp: float
    """``time.monotonic()`` timestamp when this result was written."""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class VLMAnalyzer:
    """
    Non-blocking Vision-Language Model caller.

    Class-level constants can be overridden before instantiation::

        VLMAnalyzer.ENABLED  = False   # instant kill-switch
        VLMAnalyzer.INTERVAL = 20.0    # call every 20 s instead of 30
    """

    # ── Kill switch ────────────────────────────────────────────────────
    ENABLED: bool = True

    # ── Config ─────────────────────────────────────────────────────────
    MODEL:      str   = "claude-haiku-4-5-20251001"
    INTERVAL:   float = 30.0   # minimum seconds between API calls
    JPEG_Q:     int   = 75     # JPEG compression quality for frame encoding
    MAX_TOKENS: int   = 150    # upper-bound on response length

    # ── Blending weight (used by callers, not internally) ──────────────
    BLEND_WEIGHT: float = 0.35  # share of VLM score in the final blend

    def __init__(self) -> None:
        self._result:    Optional[VLMResult]       = None
        self._lock:      threading.Lock            = threading.Lock()
        self._thread:    Optional[threading.Thread] = None
        self._last_call: float                     = 0.0
        self._client                               = None
        self._available: bool                      = False

        if not self.ENABLED:
            log.info("[VLM] Disabled via ENABLED flag.")
            return

        try:
            import anthropic  # optional dependency
            self._client    = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
            self._available = True
            log.info("[VLM] Anthropic client ready — model: %s  interval: %.0fs",
                     self.MODEL, self.INTERVAL)
        except ImportError:
            log.warning("[VLM] 'anthropic' package not installed — VLM disabled. "
                        "Run: pip install anthropic")
        except Exception as exc:
            log.warning("[VLM] Client init failed (%s) — VLM disabled.", exc)

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True only when the Anthropic client is ready and ENABLED is set."""
        return self._available

    def maybe_trigger(self, frame: np.ndarray, now: float) -> None:
        """
        Call every frame after calibration completes.

        Starts a background API call when ``INTERVAL`` seconds have elapsed
        since the last call.  Never blocks; never raises.
        """
        if not self._available:
            return
        if self._thread and self._thread.is_alive():
            return                          # previous call still in-flight
        if (now - self._last_call) < self.INTERVAL:
            return

        self._last_call = now
        snapshot = frame.copy()            # capture before the frame is mutated
        self._thread = threading.Thread(
            target=self._call, args=(snapshot,), daemon=True
        )
        self._thread.start()

    def get_result(self) -> Optional[VLMResult]:
        """Thread-safe read of the latest VLM result (``None`` until first response)."""
        with self._lock:
            return self._result

    def seconds_since_result(self, now: float) -> Optional[float]:
        """Seconds elapsed since the last successful API response, or ``None``."""
        with self._lock:
            if self._result is None:
                return None
            return now - self._result.timestamp

    def blend(self, rule_score: float) -> float:
        """
        Return a weighted blend of *rule_score* and the latest VLM score.
        Falls back to *rule_score* unchanged when no VLM result is available.
        """
        with self._lock:
            if self._result is None:
                return rule_score
            w = self.BLEND_WEIGHT
            return (1.0 - w) * rule_score + w * self._result.score

    # ── Private — background thread ──────────────────────────────────────

    def _call(self, frame: np.ndarray) -> None:
        """Runs in a daemon thread.  Writes ``_result`` on success; silent on error."""
        ts = time.monotonic()
        try:
            img_b64 = self._encode_frame(frame)
            raw     = self._request(img_b64)
            result  = self._parse(raw, ts)

            with self._lock:
                self._result = result

            log.info(
                "[VLM] score=%.0f  %-14s  conf=%-6s  | %s",
                result.score, result.dominant_signal,
                result.confidence, result.reasoning,
            )

        except Exception as exc:
            log.warning("[VLM] call failed: %s", exc)
            # Intentionally swallowed — old result is kept, main loop unaffected

    def _encode_frame(self, frame: np.ndarray) -> str:
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_Q]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode()

    def _request(self, img_b64: str) -> str:
        response = self._client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       img_b64,
                        },
                    },
                    {"type": "text", "text": _PROMPT},
                ],
            }],
        )
        return response.content[0].text

    @staticmethod
    def _parse(raw: str, ts: float) -> VLMResult:
        # Strip markdown code fences that some model versions add
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            clean = clean.rstrip("`").strip()

        data = json.loads(clean)

        return VLMResult(
            score=float(max(0.0, min(100.0, data["score"]))),
            dominant_signal=str(data.get("dominant_signal", "neutral")),
            confidence=str(data.get("confidence", "low")),
            reasoning=str(data.get("reasoning", ""))[:60],
            timestamp=ts,
        )

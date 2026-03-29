#!/usr/bin/env bash
# run_heart_cvgnal.command  — One-click macOS launcher
# Double-click this file in Finder.
# On first run: right-click → Open if macOS shows "unidentified developer".
#
# macOS permissions required (one-time):
#   System Settings → Privacy & Security → Camera → allow Terminal

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "ERROR: cannot cd to $SCRIPT_DIR"; exit 1; }

# ── Activate conda env ───────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || \
  source "$CONDA_BASE/bin/activate" 2>/dev/null || true
conda activate dslcv2 2>/dev/null || true

PYTHON="$CONDA_BASE/envs/dslcv2/bin/python"
[ -f "$PYTHON" ] || PYTHON=python3

# ── Guard: mediapipe solutions API ───────────────────────────────────────────
"$PYTHON" - <<'PYEOF' 2>/dev/null
import mediapipe as mp
assert hasattr(mp, "solutions"), "solutions API missing"
PYEOF
if [ $? -ne 0 ]; then
  echo "Fixing mediapipe …"
  "$PYTHON" -m pip install "mediapipe==0.10.14" --quiet
fi

# ── Guard: flask installed ────────────────────────────────────────────────────
"$PYTHON" -c "import flask" 2>/dev/null || {
  echo "Installing flask …"
  "$PYTHON" -m pip install "flask>=3.0,<4.0" --quiet
}

echo ""
echo "================================================="
echo "  Heart CV-gnal  ♥  Web UI Edition"
echo "  Opening → http://localhost:5001"
echo "  Press Ctrl+C here to quit."
echo "================================================="
echo ""

# Open browser after a short delay (server needs a moment to start)
sleep 1.5 && open "http://localhost:5001" &

PYTHONPATH="$SCRIPT_DIR/src" "$PYTHON" "$SCRIPT_DIR/app.py"
EXIT_CODE=$?

echo ""
[ $EXIT_CODE -eq 0 ] && echo "Closed cleanly." || echo "Exited with code $EXIT_CODE."
echo ""
echo "Press any key to close …"
read -r -n 1

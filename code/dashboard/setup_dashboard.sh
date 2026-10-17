#!/usr/bin/env bash
# End-to-end: clone + patch the LLMxIO dashboard, export our runs, serve it.
#
# Usage:
#   code/dashboard/setup_dashboard.sh <dashboard_dir> <traj:runnum> [<traj:runnum> ...]
#
# Example:
#   code/dashboard/setup_dashboard.sh /tmp/LLMxIO_Dashboard \
#       output/ppo_v2_goal/live_goal_judge/viz/trajectory.json:9 \
#       output/ppo_v2_goal/random_viz/trajectory.json:8
#
# <dashboard_dir> is cloned at the pinned commit if it does not exist.
# Each <traj:runnum> is exported into <dashboard_dir>/dashboard/public/data
# as run <runnum> (run 9 also writes the persona CSVs).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIN="9f913eed47de3001b8e6c7953d7d09415a8cb99c"
DASHBOARD_REMOTE="https://github.com/SIGNALS-Lab/LLMxIO_Dashboard.git"

if [[ $# -lt 2 ]]; then
  sed -n '2,14p' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

DASH_DIR="$1"; shift
shift_count=0
declare -a TRAJS=()
for spec in "$@"; do
  TRAJS+=("$spec")
  shift_count=$((shift_count + 1))
done

# ---- clone + patch -------------------------------------------------------
if [[ ! -d "$DASH_DIR/.git" ]]; then
  echo "[*] cloning dashboard -> $DASH_DIR"
  git clone "$DASHBOARD_REMOTE" "$DASH_DIR"
fi
pushd "$DASH_DIR" >/dev/null
git checkout -- . 2>/dev/null || true
git checkout "$PIN" 2>/dev/null || echo "[!] pinned commit unavailable; using current HEAD"
if git apply --check "$REPO_DIR/code/dashboard/llmxio_integration.patch" 2>/dev/null; then
  git apply "$REPO_DIR/code/dashboard/llmxio_integration.patch"
  echo "[*] applied llmxio_integration.patch"
else
  echo "[*] patch already applied or failed to apply (checking...)"
  git apply --reverse --check "$REPO_DIR/code/dashboard/llmxio_integration.patch" 2>/dev/null \
    && echo "[*] patch already applied" \
    || echo "[!] WARNING: patch did not apply cleanly"
fi
popd >/dev/null

DATA_DIR="$DASH_DIR/dashboard/public/data"

# ---- export each run -----------------------------------------------------
for spec in "${TRAJS[@]}"; do
  traj="${spec%%:*}"; runnum="${spec##*:}"
  if [[ ! -f "$REPO_DIR/$traj" && ! -f "$traj" ]]; then
    echo "[!] trajectory not found: $traj" >&2
    exit 1
  fi
  traj_path="$traj"
  [[ -f "$traj_path" ]] || traj_path="$REPO_DIR/$traj"
  personas_flag="--no-personas"
  # run 9 (or the first exported run) owns the global persona CSVs
  if [[ "$runnum" == "9" ]]; then personas_flag=""; fi
  echo "[*] exporting run $runnum  <-  $traj_path"
  ( cd "$REPO_DIR" && PYTHONPATH=. .venv/bin/python -m code.export_dashboard \
      --trajectory "$traj_path" --data-dir "$DATA_DIR" \
      --run-file-num "$runnum" --label "social-agents run $runnum" $personas_flag )
done

# ---- install + run -------------------------------------------------------
pushd "$DASH_DIR/dashboard" >/dev/null
if [[ ! -d node_modules ]]; then
  echo "[*] npm ci"
  if [[ -f package-lock.json ]]; then npm ci; else npm install; fi
fi
echo "[*] starting dev server on http://localhost:3000  (Ctrl-C to stop)"
npm run dev

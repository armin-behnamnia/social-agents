#!/usr/bin/env bash
# Campaign 7 & 8 — RL persuader on the colleague's two mechanisms.
#
#   C7 cascade : stance-ful within-period BFS   (matched to campaign 6,
#                judge arm flags: --goal-delta --align judge --judge bt)
#   C8 exposure: stance-less persistent queue   (+ --mech-beta-init —
#                without it the queue provably stalls after 1 hop —
#                and --reward-adoption-bonus 1.0 since Δs ≡ 0 there;
#                reward = bonus + accept shaping + judge λ·a(m))
#
# BOTH mechanisms need --mech-beta-init: their source rule is
# "adopted AND stance <= 0", so with our all-positive crowd the adopters
# (~+0.32 after one pull) never qualify as sources and diffusion dies
# after the persuaders' direct neighbours. Cascade needs it for exactly
# the same reason exposure does (verified by test).
#
# LLM backend note: the configured model is a reasoning model — the
# generation/inference token budgets in code/config.py are sized for it
# (see LLM_GENERATION_MAX_TOKENS / LLM_INFERENCE_MAX_TOKENS overrides);
# too-small budgets make it return empty content.
#
# Matched protocol (identical to campaign 6): v2 strategy space, 50
# agents, 5 persuaders, erdos_renyi, 10 periods, --obs-mode pair,
# --target-selection random_neighbor, seed 0, 300 iters × 32 episodes,
# paired eval on 50 shared seeds every 20 iters, workers 160.
#
# The shared chat endpoint occasionally loses its model registration
# (404 "model not found"); this script waits for it (10 min poll, 24 h
# cap) instead of starting a doomed 17 h run.
#
# Usage:  bash code/run_mech_campaigns.sh            # both, serially
#         bash code/run_mech_campaigns.sh cascade    # only C7
#         bash code/run_mech_campaigns.sh exposure   # only C8
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
export PYTHONPATH=.
export PERSUASION_STRATEGY_SPACE=v2

echo "[*] waiting for the chat endpoint model (${LLM_MODEL:-from .env})..."
set -a; [ -f .env ] && source .env; set +a
deadline=$(( $(date +%s) + 86400 ))
while ! PYTHONPATH=. $PY -c "
from code.train_persuader import ping_llm, check_llm
import sys
sys.exit(0 if (check_llm() and ping_llm()) else 1)
" >/dev/null 2>&1; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "[!] chat endpoint still down after 24h; aborting" >&2
        exit 2
    fi
    echo "    model not available yet; retrying in 10 min ($(date +%H:%M))"
    sleep 600
done
echo "[+] chat endpoint live."

COMMON="--mode live --iterations 300 --episodes-per-iter 32 \
  --rollout-workers 160 --n-periods 10 --obs-mode pair \
  --target-selection random_neighbor --judge bt \
  --paired-eval-every 20 --paired-eval-episodes 50"

run_cascade() {
    echo "[*] C7 cascade -> output/ppo_mech_cascade/live"
    $PY -u -m code.train_persuader $COMMON \
        --mechanism cascade --mech-beta-init \
        --goal-delta --align judge \
        --output-dir output/ppo_mech_cascade/live
}

run_exposure() {
    echo "[*] C8 exposure -> output/ppo_mech_exposure/live"
    PERSUASION_GOAL_LAMBDA=0.2 $PY -u -m code.train_persuader $COMMON \
        --mechanism exposure --mech-beta-init \
        --reward-adoption-bonus 1.0 --align judge \
        --output-dir output/ppo_mech_exposure/live
}

case "${1:-both}" in
    cascade)  run_cascade ;;
    exposure) run_exposure ;;
    both)     run_cascade && run_exposure ;;
    *) echo "usage: $0 [cascade|exposure|both]" >&2; exit 1 ;;
esac
echo "[+] mechanism campaigns finished."

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
# too-small budgets make it return empty content. .env sets
# LLM_REASONING_EFFORT=none, which keeps it in plain instruct mode:
# without that, ~10x the latency and frequent empty replies (the hidden
# reasoning eats the whole budget).
#
# Reward model: both arms reuse campaign 6's Bradley-Terry table
# (--bt-table). It is identity-indexed, so the new decoder's message
# text does not matter, and reusing it keeps the reward model exactly
# comparable to campaign 6 instead of rebuilding 31k comparisons with a
# different model.
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
#         bash code/run_mech_campaigns.sh resume-both  # resume C7 + C8
#         bash code/run_mech_campaigns.sh resume-cascade-parallel
#                                    # resume C7 + start C8, concurrently
#         bash code/run_mech_campaigns.sh resume-parallel
#                                    # resume both, concurrently
#
# Both arms run with --trace, so code/live_monitor.py can show every
# immediate variable (per-decision persuasiveness, per-period population
# state, per-hop diffusion events) with exact values and charts while the
# campaign is still running.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
export PYTHONPATH=.
export PERSUASION_STRATEGY_SPACE=v2

# Per-arm rollout concurrency. When both arms run together they put 2x
# this many threads on the shared gateway; 96 keeps the pair (~192)
# inside the measured envelope (RESULTS.md §E.2). A 320-wide launch
# already earned a sustained 503 burst from the gateway and killed the
# campaign ~2 minutes in, so the default is deliberately conservative.
# Override with MECH_WORKERS=160 to go back to the old setting.
WORKERS="${MECH_WORKERS:-96}"

# Transient-failure policy for a multi-hour campaign. The gateway answers
# 503 while it is saturated, in bursts longer than the library default
# (3 attempts / ~10 s), and losing a run that is hours in is far worse
# than waiting: 6 attempts, 2 s base, 1.5x-1x jittered, capped at 30 s
# (see code/config.py).
export LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-6}"
export LLM_RETRY_BACKOFF_SECONDS="${LLM_RETRY_BACKOFF_SECONDS:-2}"

# A crashed arm is retried (resuming from its checkpoint) up to this many
# times before the run is declared failed.
MAX_ARM_ATTEMPTS="${MECH_MAX_ATTEMPTS:-8}"

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
  --rollout-workers $WORKERS --n-periods 10 --obs-mode pair \
  --target-selection random_neighbor --judge bt \
  --bt-table output/ppo_v2_bt/bt_judge.json \
  --agent-num 50 --n-persuaders 5 --topology erdos_renyi \
  --paired-eval-every 20 --paired-eval-episodes 50"

# Immediate-variable tracing -> <output-dir>/trace/{steps,periods,events,
# episodes}.jsonl, consumed by code/live_monitor.py (see METRICS.md for
# every recorded variable). Purely observational: rewards, dynamics and
# RNG draws are untouched (pinned by code/tests/test_run_trace.py::
# TestEpisodeTracing::test_tracing_is_pure), so a traced run stays
# comparable to campaign 6. Add --trace-episodes N to cap the volume.
TRACE="--trace"

run_cascade() {
    local extra="${1:-}"
    echo "[*] C7 cascade -> output/ppo_mech_cascade/live"
    $PY -u -m code.train_persuader $COMMON $TRACE $extra \
        --mechanism cascade --mech-beta-init \
        --goal-delta --align judge \
        --output-dir output/ppo_mech_cascade/live
}

run_exposure() {
    local extra="${1:-}"
    echo "[*] C8 exposure -> output/ppo_mech_exposure/live"
    PERSUASION_GOAL_LAMBDA=0.2 $PY -u -m code.train_persuader $COMMON \
        $TRACE $extra \
        --mechanism exposure --mech-beta-init \
        --reward-adoption-bonus 1.0 --align judge \
        --output-dir output/ppo_mech_exposure/live
}

# Resuming keeps the existing curve (--resume-log re-attaches the
# per-iteration records up to the last checkpoint) and appends to the
# existing trace, so an interrupted campaign can be re-attached to
# tracing without losing its results.
C7_RESUME="--resume --resume-log output/ppo_mech_cascade/live/training_log.json"
C8_RESUME="--resume --resume-log output/ppo_mech_exposure/live/training_log.json"

# ---------------------------------------------------------------------
# Crash recovery. Both arms checkpoint every 10 iterations and the
# trainer can re-attach its curve (--resume + --resume-log), so when an
# arm exits non-zero the cheapest correct move is to restart it from its
# checkpoint rather than lose a campaign that is hours in. This wrapper
# does exactly that, bounded, so a genuine bug still surfaces instead of
# looping forever.
# ---------------------------------------------------------------------
run_arm() {
    local label="$1" fn="$2" resume_args="$3" ckpt="$4"
    local attempt extra
    for attempt in $(seq 1 "$MAX_ARM_ATTEMPTS"); do
        extra=""
        [ -f "$ckpt" ] && extra="$resume_args"
        if "$fn" "$extra"; then
            echo "[+] $label finished (attempt $attempt)"
            return 0
        fi
        if [ -n "$extra" ]; then
            echo "[!] $label exited non-zero (attempt" \
                 "$attempt/$MAX_ARM_ATTEMPTS): resuming from checkpoint" >&2
        else
            echo "[!] $label exited non-zero (attempt" \
                 "$attempt/$MAX_ARM_ATTEMPTS): no checkpoint yet, restarting" >&2
        fi
        sleep 15
    done
    echo "[!] $label FAILED after $MAX_ARM_ATTEMPTS attempts" >&2
    return 1
}

CASCADE_CKPT=output/ppo_mech_cascade/live/policy.pt
EXPOSURE_CKPT=output/ppo_mech_exposure/live/policy.pt

# Parallel modes run the two arms CONCURRENTLY as separate processes.
# Measured (RESULTS.md §E.2): the gateway's throughput scales with
# concurrency (27/s at c=16 → 90/s at c=512) while our client is idle
# (0.4 of 176 cores, 234/234 threads blocked on I/O), so the two arms
# together demand ~2x$WORKERS calls/s and finish in roughly the time ONE
# takes alone. Local cost: 2 processes, ~5 GB RAM, 2x$WORKERS threads.
#
# All three parallel spellings now behave the same way: each arm resumes
# iff it already has a checkpoint. That makes them safe to re-run after
# an interruption without remembering which arm was mid-flight.
run_parallel() {
    local p_cascade p_exposure rc=0
    run_arm "C7 cascade"  run_cascade  "$C7_RESUME" "$CASCADE_CKPT" &
    p_cascade=$!
    run_arm "C8 exposure" run_exposure "$C8_RESUME" "$EXPOSURE_CKPT" &
    p_exposure=$!
    wait "$p_cascade" || rc=1
    wait "$p_exposure" || rc=1
    return "$rc"
}

case "${1:-both}" in
    cascade)         run_arm "C7 cascade"  run_cascade  "$C7_RESUME" "$CASCADE_CKPT" ;;
    exposure)        run_arm "C8 exposure" run_exposure "$C8_RESUME" "$EXPOSURE_CKPT" ;;
    both)
        run_arm "C7 cascade" run_cascade "$C7_RESUME" "$CASCADE_CKPT" \
            && run_arm "C8 exposure" run_exposure "$C8_RESUME" "$EXPOSURE_CKPT" ;;
    resume-cascade)  run_arm "C7 cascade"  run_cascade  "$C7_RESUME" "$CASCADE_CKPT" ;;
    resume-exposure) run_arm "C8 exposure" run_exposure "$C8_RESUME" "$EXPOSURE_CKPT" ;;
    resume-both)
        run_arm "C7 cascade" run_cascade "$C7_RESUME" "$CASCADE_CKPT" \
            && run_arm "C8 exposure" run_exposure "$C8_RESUME" "$EXPOSURE_CKPT" ;;
    parallel|resume-parallel|resume-cascade-parallel) run_parallel ;;
    *)
        echo "usage: $0 [cascade|exposure|both|resume-cascade|resume-exposure|resume-both|parallel|resume-parallel|resume-cascade-parallel]" >&2
        exit 1
        ;;
esac
echo "[+] mechanism campaigns finished."

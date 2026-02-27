#!/bin/bash
# Parallel evaluation script for VisualWebArena with Gemma-3 models
# Usage: bash run_parallel_eval.sh [model_name] [result_dir_suffix]
#
# This script launches multiple run.py processes in parallel,
# each handling a different site/task range. All write to the same result dir.

set -e

# --- Configuration ---
MODEL_NAME="${1:-/home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-12b-it}"
RESULT_SUFFIX="${2:-gemma12b}"
RESULT_DIR="cache/results_${RESULT_SUFFIX}_$(date +%Y%m%d_%H%M%S)"

export DATASET=visualwebarena
export CLASSIFIEDS="http://10.148.0.21:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://10.148.0.21:7770"
export REDDIT="http://10.148.0.21:9999"
export WIKIPEDIA="http://10.148.0.21:8888"
export HOMEPAGE="http://10.148.0.21:4399"
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://localhost:8000/v1"

# Common args
COMMON_ARGS=(
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json
  --provider openai
  --model "$MODEL_NAME"
  --mode chat
  --observation_type image_som
  --action_set_tag som
  --temperature 0.0
  --max_steps 15
  --max_obs_length 3840
  --viewport_width 1280
  --viewport_height 720
  --result_dir "$RESULT_DIR"
)

mkdir -p "$RESULT_DIR"
echo "=== Parallel eval started at $(date) ==="
echo "Model: $MODEL_NAME"
echo "Result dir: $RESULT_DIR"
echo ""

# --- Launch parallel workers ---
PIDS=()

# Worker 1: Classifieds easy tasks (IDs 10-50)
echo "[Worker 1] Classifieds tasks 10-50..."
python run.py \
  --test_config_base_dir config_files/vwa/test_classifieds \
  --test_start_idx 10 \
  --test_end_idx 50 \
  "${COMMON_ARGS[@]}" \
  > "${RESULT_DIR}/worker1_classifieds.log" 2>&1 &
PIDS+=($!)

# Worker 2: Shopping easy tasks (IDs 8-50)
echo "[Worker 2] Shopping tasks 8-50..."
python run.py \
  --test_config_base_dir config_files/vwa/test_shopping \
  --test_start_idx 8 \
  --test_end_idx 50 \
  "${COMMON_ARGS[@]}" \
  > "${RESULT_DIR}/worker2_shopping.log" 2>&1 &
PIDS+=($!)

echo ""
echo "Launched ${#PIDS[@]} parallel workers: PIDs = ${PIDS[*]}"
echo "Logs: ${RESULT_DIR}/worker*.log"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${RESULT_DIR}/worker1_classifieds.log"
echo "  tail -f ${RESULT_DIR}/worker2_shopping.log"
echo ""
echo "Or check results with:"
echo "  ls ${RESULT_DIR}/render_*.html | wc -l"
echo ""

# --- Wait for all workers to finish ---
FAILED=0
for pid in "${PIDS[@]}"; do
  if wait "$pid"; then
    echo "[OK] Worker PID $pid finished successfully"
  else
    echo "[FAIL] Worker PID $pid exited with error"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "=== All workers done at $(date) ==="

# --- Summarize results ---
TOTAL=$(ls "${RESULT_DIR}"/render_*.html 2>/dev/null | wc -l)
echo "Total tasks completed: $TOTAL"

# Extract scores from logs
echo ""
echo "--- Per-worker scores ---"
for log in "${RESULT_DIR}"/worker*.log; do
  worker=$(basename "$log" .log)
  score_line=$(grep "Average score" "$log" 2>/dev/null || echo "No score found")
  pass_count=$(grep -c "(PASS)" "$log" 2>/dev/null || true)
  fail_count=$(grep -c "(FAIL)" "$log" 2>/dev/null || true)
  echo "  $worker: PASS=$pass_count FAIL=$fail_count | $score_line"
done

if [ $FAILED -gt 0 ]; then
  echo ""
  echo "WARNING: $FAILED worker(s) failed. Check logs for details."
  exit 1
fi

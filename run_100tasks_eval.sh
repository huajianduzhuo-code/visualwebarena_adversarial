#!/bin/bash
# ==============================================================================
# VisualWebArena 100-Task Evaluation Script
# 50 easy + 50 medium tasks (no Wikipedia), including gemma-12b's 8 passed tasks
# Sites: Classifieds, Shopping, Reddit
#
# Usage:
#   1. Make sure vLLM server is running in tmux (see below)
#   2. conda activate vwa
#   3. bash run_100tasks_eval.sh
#
# To start vLLM server in tmux (if not already running):
#   tmux new-session -d -s vllm_gemma12b
#   tmux send-keys -t vllm_gemma12b \
#     "conda activate spag && CUDA_VISIBLE_DEVICES=0 vllm serve \
#      /home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-12b-it \
#      --tensor-parallel-size 1 --max-model-len 16384 \
#      --gpu-memory-utilization 0.85 --dtype auto \
#      --limit-mm-per-prompt image=5" C-m
# ==============================================================================

set -e

# --- Environment Variables ---
export DATASET=visualwebarena
export CLASSIFIEDS="http://10.148.0.21:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://10.148.0.21:7770"
export REDDIT="http://10.148.0.21:9999"
export WIKIPEDIA="http://10.148.0.21:8888"
export HOMEPAGE="http://10.148.0.21:4399"
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://localhost:8000/v1"

# --- Model and output config ---
MODEL_NAME="/home/lhaoyu_google_com/spag_ckpt/pretrained/gemma-3-12b-it"
RESULT_DIR="cache/results_gemma12b_100tasks"

# --- Check vLLM server ---
echo "Checking vLLM server..."
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "ERROR: vLLM server is not running at localhost:8000!"
    echo "Start it first in tmux. See instructions at the top of this script."
    exit 1
fi
echo "vLLM server is running."

# --- Prepare result directory ---
rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

# --- Common arguments ---
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

# --- Task IDs (50 easy + 50 medium, carefully selected) ---

# Classifieds: 21 easy + 16 medium = 37 tasks
CLASSIFIEDS_IDS="0,1,2,3,6,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,40,41,42,43,48,49,50,52,53,54,55,56,68,74"

# Shopping: 25 easy + 21 medium = 46 tasks
SHOPPING_IDS="3,4,5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,33,34,35,39,40,41,42,43,59,60,61,62,63,64,65,67,68,69,70,71"

# Reddit: 4 easy + 13 medium = 17 tasks
REDDIT_IDS="0,1,2,3,4,5,9,17,18,34,35,36,37,39,40,41,42"

START_TIME=$(date +%s)
echo ""
echo "================================================================"
echo "Starting 100-task evaluation at $(date)"
echo "  Classifieds: 37 tasks (21 easy + 16 medium)"
echo "  Shopping:    46 tasks (25 easy + 21 medium)"
echo "  Reddit:      17 tasks (4 easy + 13 medium)"
echo "================================================================"
echo ""

# --- Launch 3 parallel workers (BLIP2 on CPU to save GPU for vLLM) ---

echo "=== Launching Worker 1: Classifieds (37 tasks) ==="
CUDA_VISIBLE_DEVICES="" python run.py \
  --test_config_base_dir config_files/vwa/test_classifieds \
  --task_ids "$CLASSIFIEDS_IDS" \
  "${COMMON_ARGS[@]}" \
  > "${RESULT_DIR}/worker1_classifieds.log" 2>&1 &
PID1=$!

echo "=== Launching Worker 2: Shopping (46 tasks) ==="
CUDA_VISIBLE_DEVICES="" python run.py \
  --test_config_base_dir config_files/vwa/test_shopping \
  --task_ids "$SHOPPING_IDS" \
  "${COMMON_ARGS[@]}" \
  > "${RESULT_DIR}/worker2_shopping.log" 2>&1 &
PID2=$!

echo "=== Launching Worker 3: Reddit (17 tasks) ==="
CUDA_VISIBLE_DEVICES="" python run.py \
  --test_config_base_dir config_files/vwa/test_reddit \
  --task_ids "$REDDIT_IDS" \
  "${COMMON_ARGS[@]}" \
  > "${RESULT_DIR}/worker3_reddit.log" 2>&1 &
PID3=$!

echo ""
echo "Worker 1 (Classifieds): PID $PID1"
echo "Worker 2 (Shopping):    PID $PID2"
echo "Worker 3 (Reddit):      PID $PID3"
echo ""
echo "Monitoring progress every 30s... (Ctrl+C to stop monitoring; workers continue in background)"
echo ""

# --- Monitor loop ---
while true; do
  sleep 30

  RUNNING=0
  for pid in $PID1 $PID2 $PID3; do
    if kill -0 "$pid" 2>/dev/null; then
      RUNNING=$((RUNNING + 1))
    fi
  done

  PASS=$(grep -c "(PASS)" "${RESULT_DIR}"/worker*.log 2>/dev/null || true)
  FAIL=$(grep -c "(FAIL)" "${RESULT_DIR}"/worker*.log 2>/dev/null || true)
  ERRORS=$(grep -c "Unhandled Error\|OpenAI Error" "${RESULT_DIR}"/worker*.log 2>/dev/null || true)
  TOTAL=$((PASS + FAIL))
  echo "[$(date +%H:%M:%S)] Workers: $RUNNING/3 | Done: $TOTAL/100 | PASS: $PASS | FAIL: $FAIL | ERR: $ERRORS"

  if [ "$RUNNING" -eq 0 ]; then
    break
  fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "================================================================"
echo "All workers finished at $(date)"
echo "Total wall-clock time: ${MINUTES}m ${SECONDS}s"
echo "================================================================"
echo ""

# --- Final Summary ---
echo "--- Per-worker results ---"
for log in "${RESULT_DIR}"/worker*.log; do
  worker=$(basename "$log" .log)
  score_line=$(grep "Average score" "$log" 2>/dev/null || echo "No score")
  pass_count=$(grep -c "(PASS)" "$log" 2>/dev/null || true)
  fail_count=$(grep -c "(FAIL)" "$log" 2>/dev/null || true)
  errors=$(grep -c "Unhandled Error\|OpenAI Error" "$log" 2>/dev/null || true)
  echo "  $worker: PASS=$pass_count FAIL=$fail_count ERRORS=$errors | $score_line"
done

TOTAL_PASS=$(grep -c "(PASS)" "${RESULT_DIR}"/worker*.log 2>/dev/null || true)
TOTAL_FAIL=$(grep -c "(FAIL)" "${RESULT_DIR}"/worker*.log 2>/dev/null || true)
TOTAL=$((TOTAL_PASS + TOTAL_FAIL))
if [ "$TOTAL" -gt 0 ]; then
  echo ""
  echo "================================================================"
  echo "Overall: $TOTAL_PASS/$TOTAL passed ($(python3 -c "print(f'{$TOTAL_PASS/$TOTAL*100:.1f}%')"))"
  echo "================================================================"
fi

# --- Detailed breakdown by difficulty ---
echo ""
echo "--- Breakdown by difficulty ---"
python3 << 'PYEOF'
import json, re, os
from collections import defaultdict

RESULT_DIR = os.environ.get("RESULT_DIR", "cache/results_gemma12b_100tasks")
task_info = {}
for site in ['classifieds', 'shopping', 'reddit']:
    with open(f'config_files/vwa/test_{site}.raw.json') as f:
        tasks = json.load(f)
    for i, t in enumerate(tasks):
        task_info[(site, i)] = {
            'difficulty': t.get('overall_difficulty', 'unknown'),
            'task_id': t['task_id'],
            'intent': t['intent'][:80],
        }

worker_map = {
    'worker1_classifieds': 'classifieds',
    'worker2_shopping': 'shopping',
    'worker3_reddit': 'reddit',
}

results = []
for worker, site in worker_map.items():
    log_path = f"{RESULT_DIR}/{worker}.log"
    if not os.path.exists(log_path):
        continue
    with open(log_path) as f:
        lines = f.readlines()
    current_idx = None
    for line in lines:
        m = re.search(r'\[Config file\].*?/(\d+)\.json', line)
        if m:
            current_idx = int(m.group(1))
        if current_idx is not None:
            if '(PASS)' in line:
                results.append((site, current_idx, 'pass'))
                current_idx = None
            elif '(FAIL)' in line:
                results.append((site, current_idx, 'fail'))
                current_idx = None
            elif 'Unhandled Error' in line or 'OpenAI Error' in line:
                results.append((site, current_idx, 'error'))
                current_idx = None

stats = defaultdict(lambda: {'pass': 0, 'fail': 0, 'error': 0})
site_diff = defaultdict(lambda: {'pass': 0, 'fail': 0, 'error': 0})

for site, idx, result in results:
    info = task_info.get((site, idx), {})
    diff = info.get('difficulty', 'unknown')
    stats[diff][result] += 1
    site_diff[(site, diff)][result] += 1

print(f"{'Difficulty':<10} {'PASS':>6} {'FAIL':>6} {'ERR':>5} {'Total':>6} {'Rate':>8}")
print("-" * 45)
for diff in ['easy', 'medium']:
    s = stats.get(diff, {'pass':0,'fail':0,'error':0})
    total = s['pass'] + s['fail'] + s['error']
    scored = s['pass'] + s['fail']
    rate = f"{s['pass']/scored*100:.1f}%" if scored > 0 else "N/A"
    print(f"{diff:<10} {s['pass']:>6} {s['fail']:>6} {s['error']:>5} {total:>6} {rate:>8}")

print()
print(f"{'Site':<15} {'Diff':<10} {'PASS':>5} {'FAIL':>5} {'ERR':>4} {'Total':>5} {'Rate':>8}")
print("-" * 55)
for site in ['classifieds', 'shopping', 'reddit']:
    for diff in ['easy', 'medium']:
        s = site_diff.get((site, diff))
        if not s: continue
        total = s['pass'] + s['fail'] + s['error']
        scored = s['pass'] + s['fail']
        rate = f"{s['pass']/scored*100:.1f}%" if scored > 0 else "N/A"
        print(f"{site:<15} {diff:<10} {s['pass']:>5} {s['fail']:>5} {s['error']:>4} {total:>5} {rate:>8}")

print()
print("Passed tasks:")
for site, idx, result in results:
    if result == 'pass':
        info = task_info.get((site, idx), {})
        print(f"  [{site}] #{idx} ({info.get('difficulty','?')}): {info.get('intent','?')}")
PYEOF

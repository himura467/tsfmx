#!/usr/bin/env bash
# Run MM-TSFlib on its own pre-processed Time-MMD data.
# Results are appended to third_party/MM-TSFlib/result_longterm_forecast.
# Compare with: PYTHONPATH=. uv run python scripts/compare_benchmark_results.py
#
# Usage: ./scripts/run_mm_tsflib_benchmark.sh [GPU_ID] [TS_MODEL] [HF_TOKEN]
set -euo pipefail

GPU=${1:-0}
MODEL=${2:-"Autoformer"}
HF_TOKEN=${3:-"NA"}

SEQ_LEN=32   # matches tsfmx context_len
PRED_LEN=32  # matches tsfmx horizon_len
LABEL_LEN=16 # seq_len // 2, consistent with MM-TSFlib convention
TEXT_LEN=4   # selects Final_Search_4; MM-TSFlib data only has Final_Search_2/4/6

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
MM_DIR="$REPO_DIR/third_party/MM-TSFlib"
DATA_ROOT="$MM_DIR/data"
PYTHON="$REPO_DIR/.venv/bin/python"

export CUDA_VISIBLE_DEVICES="$GPU"

if [ ! -f "$PYTHON" ]; then
    echo "Error: uv venv not found at $PYTHON. Run 'uv sync --extra bench' first." >&2
    exit 1
fi

if [ ! -d "$MM_DIR" ]; then
    echo "Error: MM-TSFlib not found at $MM_DIR. Run './scripts/setup_mm_tsflib.sh' first." >&2
    exit 1
fi

# Only the 5 domains tsfmx was trained on.
declare -A DOMAIN_FILES=(
    ["Algriculture"]="US_RetailBroilerComposite_Month.csv"
    ["Economy"]="US_TradeBalance_Month.csv"
    ["Environment"]="NewYork_AQI_Day.csv"
    ["Public_Health"]="US_FLURATIO_Week.csv"
    ["Traffic"]="US_VMT_Month.csv"
)

cd "$MM_DIR"

for domain in "${!DOMAIN_FILES[@]}"; do
    data_file="${DOMAIN_FILES[$domain]}"
    echo "=== $domain ($data_file) ==="
    "$PYTHON" -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id "${domain}_${SEQ_LEN}_${PRED_LEN}_LLAMA3" \
        --model "$MODEL" \
        --data custom \
        --root_path "$DATA_ROOT/$domain" \
        --data_path "$data_file" \
        --features S \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --pred_len $PRED_LEN \
        --llm_model LLAMA3 \
        --text_len $TEXT_LEN \
        --prompt_weight 0.1 \
        --huggingface_token "$HF_TOKEN"
done

echo "Done. Results in $MM_DIR/result_longterm_forecast"

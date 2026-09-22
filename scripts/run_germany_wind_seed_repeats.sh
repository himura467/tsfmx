#!/usr/bin/env bash
# Re-train fixed trial configurations under several seeds on the Germany wind series.
#
# One sweep per condition yields one checkpoint per condition, which confounds the comparison with
# the seed that trained it and with the luck of which configuration topped the ranking. This takes
# the configs export_sweep_trial_config.py writes, trains each once per seed, and evaluates every
# resulting checkpoint on test, so the gap between conditions can be read against their spread.
#
# Usage: ./scripts/run_germany_wind_seed_repeats.sh CONFIG_DIR
#   CONFIG_DIR  Holds one <condition>.yml per condition. A name starting with "adapter" trains in
#               adapter mode, one starting with "fusion" in fusion mode; e.g. adapter.yml,
#               fusion.yml, fusion_rank2.yml.
#   SEEDS       Seeds to train under (default "1 2 3 4 5").
#
# A seed whose test_metrics.json already exists is skipped, so an interrupted run resumes where it
# stopped. Trials run one after another because each mode's trials share outputs/sweeps/<mode>/checkpoints.
set -euo pipefail

CONFIG_DIR="${1:?Usage: $0 CONFIG_DIR}"
SEEDS="${SEEDS:-1 2 3 4 5}"
MODEL_CONFIG="examples/time_mmd/configs/models/chronos.yml"
ENTITIES=(wind_50Hertz wind_Amprion wind_TenneT wind_TransnetBW)
OUT_DIR="outputs/germany_wind_seed_repeats"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$REPO_DIR"
export PYTHONPATH=.

shopt -s nullglob
configs=("$CONFIG_DIR"/*.yml)
if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No <condition>.yml found in $CONFIG_DIR" >&2
    exit 1
fi

for config in "${configs[@]}"; do
    condition="$(basename "$config" .yml)"
    case "$condition" in
        adapter*) sweep_script="scripts/tune_time_mmd_adapter_sweep.py" ;;
        fusion*) sweep_script="scripts/tune_time_mmd_fusion_sweep.py" ;;
        *)
            echo "Cannot tell the mode of $condition: name it adapter* or fusion*" >&2
            exit 1
            ;;
    esac

    for seed in $SEEDS; do
        seed_dir="$OUT_DIR/$condition/seed_$seed"
        if [[ -f "$seed_dir/test_metrics.json" ]]; then
            echo "=== $condition seed $seed: already evaluated, skipping ==="
            continue
        fi

        echo "=== $condition seed $seed: train ==="
        uv run python "$sweep_script" \
            --model-config "$MODEL_CONFIG" \
            --sweep-config "$config" \
            --dataset fidel_ts \
            --entities "${ENTITIES[@]}" \
            --count 1 \
            --seed "$seed" \
            --keep-best-val-loss \
            --best-checkpoint-dir "$seed_dir"

        echo "=== $condition seed $seed: evaluate ==="
        uv run python scripts/eval_tsfmx_checkpoint.py \
            --model-config "$MODEL_CONFIG" \
            --checkpoint-path "$seed_dir/best_val_loss.pt" \
            --dataset fidel_ts \
            --domains "${ENTITIES[@]}" \
            --output "$seed_dir/test_metrics.json"
    done
done

echo "=== Summary ==="
uv run python scripts/summarize_seed_repeats.py "$OUT_DIR" --reference adapter --output "$OUT_DIR/summary.json"

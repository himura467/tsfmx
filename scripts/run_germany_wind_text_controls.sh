#!/usr/bin/env bash
# Separate what the text's content adds on Germany wind from what any fusion head adds.
#
# The mean ablation of a fusion head trained on the real text keeps most of its gain over the
# text-free forecast, so that gain may not need the text at all. Four conditions share one seed,
# one sweep budget and, for the three fusion runs, one search space:
#
#   adapter   Adapter fine-tuned without text: the unimodal baseline.
#   text      Fusion head trained on the real text.
#   constant  Fusion head trained on the training-split mean embedding: only an offset is learnable.
#   centered  Fusion head trained on mean-subtracted text: no offset can come from the text.
#
# text - constant is what the content adds; centered checks whether that survives without an offset.
# Sweeps run one after another because each mode's trials share outputs/sweeps/<mode>/checkpoints.
#
# Usage: ./scripts/run_germany_wind_text_controls.sh [COUNT]
#   COUNT          Trials per sweep (default 30).
#   CONDITIONS     Subset to run, e.g. "constant centered" (default: all four).
#   SEED           Seed shared by every sweep (default 42).
#   FUSION_SWEEP   Fusion search space (default fusion_1layer.yml; centering is exact only for one layer).
set -euo pipefail

COUNT="${1:-30}"
CONDITIONS="${CONDITIONS:-adapter text constant centered}"
SEED="${SEED:-42}"
FUSION_SWEEP="${FUSION_SWEEP:-examples/time_mmd/configs/sweeps/fusion_1layer.yml}"
ADAPTER_SWEEP="examples/time_mmd/configs/sweeps/adapter.yml"
MODEL_CONFIG="examples/time_mmd/configs/models/chronos.yml"
DATASET_CONFIG="examples/fidel_ts/configs/datasets/germany_renewable_energy_grid.yml"
ENTITIES=(wind_50Hertz wind_Amprion wind_TenneT wind_TransnetBW)
OUT_DIR="outputs/germany_wind_text_controls"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$REPO_DIR"
export PYTHONPATH=.

./scripts/download_fidel_ts.sh Germany_Renewable_Energy_Grid

# Sweeps train on the augmented cache and validate and test on the plain one, so both are needed.
for augment in "" "--augment"; do
    uv run python scripts/cache_fidel_ts_datasets.py \
        --model-config "$MODEL_CONFIG" \
        --dataset-config "$DATASET_CONFIG" \
        --text-encoder-type english \
        --entities "${ENTITIES[@]}" \
        ${augment:+"$augment"}
done

for condition in $CONDITIONS; do
    checkpoint_dir="$OUT_DIR/$condition"
    common_args=(
        --model-config "$MODEL_CONFIG"
        --dataset fidel_ts
        --entities "${ENTITIES[@]}"
        --count "$COUNT"
        --seed "$SEED"
        --keep-best-val-loss
        --best-checkpoint-dir "$checkpoint_dir"
    )

    echo "=== Sweep: $condition ==="
    case "$condition" in
        adapter)
            uv run python scripts/tune_time_mmd_adapter_sweep.py --sweep-config "$ADAPTER_SWEEP" "${common_args[@]}"
            ;;
        text)
            uv run python scripts/tune_time_mmd_fusion_sweep.py --sweep-config "$FUSION_SWEEP" "${common_args[@]}"
            ;;
        constant)
            uv run python scripts/tune_time_mmd_fusion_sweep.py --sweep-config "$FUSION_SWEEP" "${common_args[@]}" \
                --constant-text
            ;;
        centered)
            uv run python scripts/tune_time_mmd_fusion_sweep.py --sweep-config "$FUSION_SWEEP" "${common_args[@]}" \
                --center-text
            ;;
        *)
            echo "Unknown condition: $condition" >&2
            exit 1
            ;;
    esac

    echo "=== Test metrics: $condition ==="
    uv run python scripts/eval_tsfmx_checkpoint.py \
        --model-config "$MODEL_CONFIG" \
        --checkpoint-path "$checkpoint_dir/best_val_loss.pt" \
        --dataset fidel_ts \
        --domains "${ENTITIES[@]}" \
        --output "$checkpoint_dir/test_metrics.json"

    # Only heads that see sample-specific text have ablations worth reading: under centering
    # `mean` collapses onto `drop`, which leaves `shuffle` as the test of reading.
    if [[ "$condition" == "text" || "$condition" == "centered" ]]; then
        echo "=== Text ablations: $condition ==="
        uv run python scripts/eval_time_mmd_text_ablation.py \
            --model-config "$MODEL_CONFIG" \
            --checkpoint-path "$checkpoint_dir/best_val_loss.pt" \
            --dataset fidel_ts \
            --domains "${ENTITIES[@]}" \
            --ablations none drop mean shuffle cross_domain \
            --output "$checkpoint_dir/text_ablation.json"
    fi
done

echo "=== Summary (test MSE; % relative to adapter, negative is better) ==="
uv run python - "$OUT_DIR" "${ENTITIES[@]}" <<'EOF'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
entities = sys.argv[2:]
conditions = ["adapter", "text", "constant", "centered"]
metrics = {}
for condition in conditions:
    path = out_dir / condition / "test_metrics.json"
    if path.exists():
        metrics[condition] = json.loads(path.read_text())

rows = [*entities, "macro"]
def mse(condition, row):
    per_entity = metrics[condition]
    if row == "macro":
        return sum(per_entity[e]["mse"] for e in entities) / len(entities)
    return per_entity[row]["mse"]

present = [c for c in conditions if c in metrics]
print(f"{'series':<18}" + "".join(f"{c:>20}" for c in present))
for row in rows:
    cells = []
    for condition in present:
        value = mse(condition, row)
        if "adapter" in metrics and condition != "adapter":
            base = mse("adapter", row)
            cells.append(f"{value:.4f} ({(value - base) / base * 100:+.1f}%)")
        else:
            cells.append(f"{value:.4f}")
    print(f"{row:<18}" + "".join(f"{cell:>20}" for cell in cells))
EOF

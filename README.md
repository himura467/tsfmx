# TSFMx

**TSFMx** (**T**SFMx **S**tandardizes **F**usion of **M**ultimodal e**x**ogenous features) is a framework for extending TSFMs (including [TimesFM](https://github.com/google-research/timesfm) and [Chronos](https://github.com/amazon-science/chronos-forecasting)) with multimodal inputs such as text.

## Installation

```sh
pip install tsfmx[all]
```

## Quick Start

### 1. Setup

Clone the Time-MMD dataset:

```sh
./scripts/clone_time_mmd.sh
```

Split the dataset into train / val / test:

```sh
PYTHONPATH=. uv run python scripts/split_time_mmd_datasets.py \
    --train-ratio 0.7 \
    --val-ratio 0.1
```

### 2. Pre-compute Text Embeddings

**TimesFM**:

```sh
PYTHONPATH=. uv run python scripts/cache_time_mmd_datasets.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --text-encoder-type english
PYTHONPATH=. uv run python scripts/cache_time_mmd_datasets.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --text-encoder-type english --augment
```

**Chronos**:

```sh
PYTHONPATH=. uv run python scripts/cache_time_mmd_datasets.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --text-encoder-type english
PYTHONPATH=. uv run python scripts/cache_time_mmd_datasets.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --text-encoder-type english --augment
```

### 3. Fusion Hyperparameter Tuning

Run a W&B Sweeps search for the fusion mode (adapter frozen, fusion layer trained):

**TimesFM**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_fusion_sweep.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --sweep-config examples/time_mmd/configs/sweeps/fusion_3layers.yml
```

**Chronos**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_fusion_sweep.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --sweep-config examples/time_mmd/configs/sweeps/fusion_3layers.yml
```

To run the adapter mode (adapter fine-tuned, no fusion):

**TimesFM**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_adapter_sweep.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --sweep-config examples/time_mmd/configs/sweeps/adapter.yml
```

**Chronos**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_adapter_sweep.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --sweep-config examples/time_mmd/configs/sweeps/adapter.yml
```

### 4. Fine-tune Hyperparameter Tuning

After fusion tuning, run a W&B Sweeps search for the finetune mode (adapter + fusion trained jointly), starting from the best fusion checkpoint:

**TimesFM**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_finetune_sweep.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --sweep-config examples/time_mmd/configs/sweeps/finetune_1layer.yml \
    --fusion-checkpoint-path outputs/sweeps/fusion/best_checkpoints/best_val_loss.pt
```

**Chronos**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_finetune_sweep.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --sweep-config examples/time_mmd/configs/sweeps/finetune_1layer.yml \
    --fusion-checkpoint-path outputs/sweeps/fusion/best_checkpoints/best_val_loss.pt
```

### 5. Visualize Forecasts

After training, generate per-sample forecast plots from a saved checkpoint:

**TimesFM**:

```sh
PYTHONPATH=. uv run python scripts/visualize_time_mmd_predictions.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --checkpoint-path outputs/sweeps/fusion/best_checkpoints/best_val_loss.pt \
    --output-dir outputs/visualizations/timesfm
```

**Chronos**:

```sh
PYTHONPATH=. uv run python scripts/visualize_time_mmd_predictions.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --checkpoint-path outputs/sweeps/fusion/best_checkpoints/best_val_loss.pt \
    --output-dir outputs/visualizations/chronos
```

Use `--max-samples N` to limit the number of plots per split, and `--splits train val test` to select which splits to visualize.

## Benchmark Comparison with MM-TSFlib

[MM-TSFlib](https://github.com/AdityaLab/MM-TSFlib) is cloned under `third_party/MM-TSFlib` (not tracked by git). MM-TSFlib is run on its own pre-processed Time-MMD CSVs; tsfmx is evaluated on the raw Time-MMD data split 70/10/20. Both cover the same underlying domains and split ratio.

```sh
./scripts/setup_mm_tsflib.sh
```

### 1. Run MM-TSFlib benchmark

```sh
./scripts/run_mm_tsflib_benchmark.sh 0 Autoformer YOUR_HF_TOKEN
```

Requires a HuggingFace token with access to LLaMA 3.

### 2. Evaluate tsfmx checkpoint

```sh
PYTHONPATH=. uv run python scripts/eval_tsfmx_checkpoint.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --checkpoint-path outputs/sweeps/fusion/best_checkpoints/best_val_loss.pt
```

### 3. Compare results

```sh
PYTHONPATH=. uv run python scripts/compare_benchmark_results.py
```

## Acknowledgments

We thank the [Time-MMD](https://github.com/AdityaLab/Time-MMD) team for providing the multimodal time series dataset used in our examples and experiments.

## License

MIT

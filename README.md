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
    --train-ratio 0.6 \
    --val-ratio 0.2
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
    --sweep-config examples/time_mmd/configs/sweeps/fusion_1layer.yml \
    --best-checkpoint-path outputs/sweeps/fusion/best_fusion_timesfm.pt
```

**Chronos**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_fusion_sweep.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --sweep-config examples/time_mmd/configs/sweeps/fusion_1layer.yml \
    --best-checkpoint-path outputs/sweeps/fusion/best_fusion_chronos.pt
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

### 4. Finetune Hyperparameter Tuning

After fusion tuning, run a W&B Sweeps search for the finetune mode (adapter + fusion trained jointly), starting from the best fusion checkpoint:

**TimesFM**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_finetune_sweep.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --sweep-config examples/time_mmd/configs/sweeps/finetune_1layer.yml \
    --fusion-checkpoint-path outputs/sweeps/fusion/best_fusion_timesfm.pt
```

**Chronos**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_finetune_sweep.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --sweep-config examples/time_mmd/configs/sweeps/finetune_1layer.yml \
    --fusion-checkpoint-path outputs/sweeps/fusion/best_fusion_chronos.pt
```

## Acknowledgments

We thank the [Time-MMD](https://github.com/AdityaLab/Time-MMD) team for providing the multimodal time series dataset used in our examples and experiments.

## License

MIT

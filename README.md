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

### 3. Hyperparameter Tuning

Run a W&B Sweeps search for the multimodal model:

**TimesFM**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_sweep.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --sweep-config examples/time_mmd/configs/sweeps/fusion_1layer.yml
```

**Chronos**:

```sh
PYTHONPATH=. uv run python scripts/tune_time_mmd_sweep.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --sweep-config examples/time_mmd/configs/sweeps/fusion_1layer.yml
```

To compare against a fine-tuned baseline:

**TimesFM**:

```sh
PYTHONPATH=. uv run python scripts/tune_baseline_sweep.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --sweep-config examples/time_mmd/configs/sweeps/baseline.yml
```

**Chronos**:

```sh
PYTHONPATH=. uv run python scripts/tune_baseline_sweep.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --sweep-config examples/time_mmd/configs/sweeps/baseline.yml
```

## Acknowledgments

We thank the [Time-MMD](https://github.com/AdityaLab/Time-MMD) team for providing the multimodal time series dataset used in our examples and experiments.

## License

MIT

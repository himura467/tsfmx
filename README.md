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

### 6. Text Ablation Analysis

Beating a unimodal baseline does not prove that a model reads its text: the fusion branch can also act as a plain regularizer, or latch onto a domain identity signal that happens to be encoded in the text embeddings. This script evaluates one checkpoint repeatedly, perturbing only the text side each time, and reports the degradation relative to the unperturbed run.

```sh
PYTHONPATH=. uv run python scripts/eval_time_mmd_text_ablation.py \
    --model-config examples/time_mmd/configs/models/timesfm.yml \
    --checkpoint-path outputs/sweeps/fusion/best_checkpoints/best_val_loss.pt \
    --output outputs/text_ablation_results.json
```

| Ablation | What it does | What a drop in accuracy means |
| --- | --- | --- |
| `none` | Passes text through unchanged. | Reference row that the deltas are measured against. |
| `drop` | Removes text entirely, so the decoder skips fusion. | The fusion branch contributes something, but not necessarily by reading the text. |
| `mean` | Replaces every sample's text with the dataset mean. | Between-sample variation is used. *Matching* `none` instead means the fusion output has collapsed to a learned constant. |
| `shuffle` | Gives each sample another sample's text, via a derangement over the split. | The model uses the *content* of the text, not merely its presence. |
| `cross_domain` | Gives each sample another domain's text, paired by position. | The model reads more than the domain identity the text carries. |
| `permute_patches` | Shuffles patch order within each sample's own text. | The model uses the temporal alignment between text and patches. |
| `noise` | Adds Gaussian noise scaled by the split's embedding std. | Graded robustness curve; scale it with `--noise-scale`. |
| `oracle` | Replaces the text with the sample's own future, written out as numbers. | Read in reverse — see below. |
| `oracle_trend` | Replaces it with the same future described in words. | Read in reverse — see below. |

The telling comparison is `drop` against `shuffle`. If both degrade by a similar amount, the model is reading the text. If `drop` degrades but `shuffle` does not, the fusion branch is contributing independently of what the text actually says, and `mean` distinguishes the two readings. `cross_domain` separates one more explanation from those: `shuffle` leaves the domain intact, so text that only identifies the domain survives it, and only `cross_domain` destroys that too.

The oracles invert the question. Every other row degrades the text and asks whether the forecast notices; these hand the model text that is by construction worth reading, so they should *improve* on `none`. No improvement means the model cannot use text at all, whatever the corpus says; an improvement places the fault in the corpus rather than in the fusion mechanism. `oracle_trend` exists because sentence encoders represent magnitude poorly: `oracle` failing on its own would be ambiguous between a fusion branch that cannot carry sample-specific information and an encoder that cannot read a list of floats. Both consume the labels, so neither is a forecasting result.

Perturbations are applied per sample index rather than per batch, so results are independent of batch size and iteration order, and reproducible for a given `--seed`. Use `--ablations` to run a subset (`none` is always included as the reference), `--domains` to select domains, and `--augment` to evaluate on the augmented cache from step 2.

`cross_domain` takes its text from the next entry in `--domains`, cycling, and is skipped with a warning when fewer than two domains load. The pairing is positional, so it destroys the domain identity without preserving the date alignment. The oracles synthesize text and so load the text encoder named by the model config, which must be the one the cache was built with; they run by default, so pass `--ablations` explicitly to skip that load.

Read the deltas against the per-domain sample counts, which are logged and written to the `num_samples` field of the output JSON. With the default `context_len` and `horizon_len` of 32, a monthly domain's test split holds only a handful of samples, far too few to read a difference of a few percent; `--augment` raises that by up to `patch_len` times. Those added samples are overlapping windows rather than independent draws, so the confidence intervals narrow less than the raw count suggests.

Note that under the current bias-free fusion projection, `drop` and zeroing the text embeddings are equivalent.

### 7. Text Fusion Diagnostics

The ablations above say whether sample-specific text information reaches the forecast. When it does not, this script says where it was lost, which decides whether to fix the text pipeline or the fusion mechanism.

```sh
PYTHONPATH=. uv run python scripts/diagnose_time_mmd_text_fusion.py \
    --model-config examples/time_mmd/configs/models/chronos.yml \
    --checkpoint-path outputs/sweeps/fusion/best_checkpoints/best_val_loss.pt \
    --augment
```

It splits both the text embeddings and the fusion projection output into the component shared by every sample and the component that varies between samples, reported as `text_variation_ratio` and `projection_variation_ratio`. `constant_vs_ts` and `varying_vs_ts` put those two components on the scale of the time series embeddings they are added to.

A low `text_variation_ratio`, or a `text_mean_pairwise_cosine` near 1, means the signal is already gone at the encoder and no fusion mechanism could recover it: `all-MiniLM-L6-v2` truncates at 256 tokens and silently drops the tail of a multi-article patch. A healthy `text_variation_ratio` with a low `projection_variation_ratio` means the additive projection is discarding it instead.

A high `varying_vs_ts` is a third failure, and the one measured so far: the text reaches the backbone intact but at a magnitude rivalling the time series representation, with no way for the model to admit less of it. The `fusion_normalize` option answers that, dividing out the projection's own output scale and replacing it with an explicit learned one. [chronos_normalized.yml](examples/time_mmd/configs/models/chronos_normalized.yml) enables it; pass it to the step 3 sweep as `--model-config`, then re-run this script to see how far `varying_vs_ts` fell. It defaults to off, so checkpoints trained before it are unaffected.

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

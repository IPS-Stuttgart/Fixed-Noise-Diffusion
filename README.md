# Fixed-Noise Diffusion Starter

Minimal image DDPM experiment stack for the WP2 fixed-noise reproduction:
fresh Gaussian noise versus reusable Gaussian template pools.

The original controlled experiments use CIFAR-10. For a non-CIFAR validation
dataset, use STL-10 with the `train+unlabeled` split resized to 32x32. This keeps
the fixed-noise pool memory footprint comparable to CIFAR-10 while moving the
validation away from the Tiny Images lineage.

## Environment

Use Python 3.12. On this machine, PyTorch with CUDA is already available for
`py -3.12`. For a fresh environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python -m pip install -e .
```

If you use the existing Python 3.12 installation without installing the package,
set `PYTHONPATH=src` before running modules.

## Smoke Test

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m fixed_noise_diffusion.train --config smoke.yaml
```

The smoke run uses synthetic images, one train step, one denoising validation
pass, and one sample grid.

## Paper CIFAR-10 Evidence Stack

The paper-facing CIFAR-10 evidence stack is a 100-epoch `base64` DDPM sweep with
a cosine noise schedule. It includes the Gaussian baseline, the fixed-pool ladder

```text
M = 250, 500, 1k, 2k, 5k, 10k, 20k, 50k, 100k
```

and the same ladder with whitened fixed pools. Each condition is run with model
seeds `0,1,2`. This is intentionally heavier than the quick examples below: the
scaling/whitening sweep is 57 training jobs.

PowerShell, matching the Windows examples elsewhere in this README:

```powershell
$env:PYTHONPATH = "src"
.\src\fixed_noise_diffusion\scripts\run_wp2_cifar10_evidence_stack.ps1 `
  -OutputRoot runs/wp2_cifar10_pool_scaling_100ep `
  -DataDir data `
  -DownloadData
```

Linux/macOS shell:

```bash
export PYTHONPATH=src
PYTHON=python \
OUTPUT_ROOT=runs/wp2_cifar10_pool_scaling_100ep \
DATA_DIR=data \
DOWNLOAD_DATA=true \
bash src/fixed_noise_diffusion/scripts/run_wp2_cifar10_evidence_stack.sh
```

By default the scripts do not save model checkpoints because the denoising-gap
evidence stack only needs compact run artifacts. Set `-SaveCheckpoints` in
PowerShell or `SAVE_CHECKPOINTS=true` in Bash if you also want later sample-quality
or timestep-local evaluations from saved checkpoints.

Each run writes the reproducibility artifacts needed to audit the paper rows:

- `metrics.csv` and `metrics.jsonl`, including `train_den_loss`,
  `gaussian_den_loss`, and `denoising_gap` eval rows,
- `config.yaml`,
- `run_metadata.json`,
- `run_summary.json`.

Summarize the CIFAR-10 scaling/whitening sweep into the same column schema used
by the paper curation step:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m fixed_noise_diffusion.summarize_denoising_gaps `
  --sweep-dir runs/wp2_cifar10_pool_scaling_100ep `
  --output runs/wp2_cifar10_pool_scaling_100ep_gap_summary.csv
```

or, from a POSIX shell:

```bash
PYTHONPATH=src python -m fixed_noise_diffusion.summarize_denoising_gaps \
  --sweep-dir runs/wp2_cifar10_pool_scaling_100ep \
  --output runs/wp2_cifar10_pool_scaling_100ep_gap_summary.csv
```

The paper table reports selected rows from this full sweep; the full CSV contains
all checkpoint epochs `1,5,10,25,50,100` and all pool sizes above.

For a quick local sanity check that does not reproduce the paper table, run only a
single seed and a few representative conditions:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m fixed_noise_diffusion.train --config cifar10_base.yaml `
  --set run_name=wp2_100ep_cifar10_gaussian_seed0 `
  --set output_dir=runs/wp2_cifar10_quick `
  --set seed=0 `
  --set diffusion.beta_schedule=cosine `
  --set training.epochs=100 `
  --set training.checkpoint_epochs=[1,5,10,25,50,100] `
  --set training.save_checkpoint=false `
  --set noise.mode=gaussian `
  --set noise.pool_size=null

py -3.12 -m fixed_noise_diffusion.train --config cifar10_base.yaml `
  --set run_name=wp2_100ep_cifar10_fixed_pool_1k_seed0 `
  --set output_dir=runs/wp2_cifar10_quick `
  --set seed=0 `
  --set diffusion.beta_schedule=cosine `
  --set training.epochs=100 `
  --set training.checkpoint_epochs=[1,5,10,25,50,100] `
  --set training.save_checkpoint=false `
  --set noise.mode=fixed_pool `
  --set noise.pool_size=1000
```

The CIFAR-10 pool-seed robustness control is separate from the model-seed ladder.
It holds the model/training seed fixed and varies only the fixed-pool seed. On a
self-hosted runner, use:

```text
.github/workflows/wp2-cifar10-pool-seed-robustness.yml
```

The workflow runs `M in {1k,10k,20k,100k}` with `pool_seed in {111,222,333}` at
fixed model seed `0`, using the same 100-epoch cosine CIFAR-10 setup.

Plot the denoising-gap curves:

```powershell
$env:PYTHONPATH = "src"
$runs = Get-ChildItem runs/wp2_cifar10_pool_scaling_100ep -Directory | ForEach-Object { $_.FullName }
py -3.12 -m fixed_noise_diffusion.plot_results `
  --runs $runs `
  --output runs/wp2_cifar10_pool_scaling_100ep/wp2_cifar10_pool_scaling_100ep.png
```

## STL-10 Validation

STL-10 is intended as a targeted validation, not as a full replacement for the
CIFAR-10 pool-size sweep. The recommended paper-facing check is:

- fresh Gaussian baseline,
- fixed pool with `M=1k`,
- fixed pool with `M=10k`,
- fixed pool with `M=100k`,
- seeds `0,1,2`,
- base64 model, cosine schedule, 100 epochs,
- STL-10 `train+unlabeled` split for training and `test` split for validation,
  resized to 32x32.

The base config is available as:

```powershell
py -3.12 -m fixed_noise_diffusion.train --config stl10_base.yaml
```

For GPU servers registered as self-hosted GitHub runners, use the manual
workflow:

```text
.github/workflows/wp2-stl10-validation.yml
```

The workflow runs the 12 validation jobs above and uploads only compact artifacts
by default: metrics, config, run metadata, and run summary. It does not upload
datasets, generated sample directories, or checkpoints. It uses a persistent
dataset cache on the self-hosted runner and a file lock around the initial
STL-10 download so the matrix jobs do not repeatedly download the dataset.
By default it uses the runner-local virtual environment at
`/home/florianpfaff/fixed-noise-diffusion-work/Fixed-Noise-Diffusion/.venv/bin/python`
instead of `actions/setup-python`, because the latter can hang on the current
self-hosted runners before training starts. Override the `python_bin` workflow
input if a runner uses a different environment path.

## CelebA-64 Validation

CelebA-64 is available as a larger, face-domain validation after the CIFAR-10 and
STL-10 denoising-gap checks are stable. The packaged config center-crops CelebA
images to 178x178 and resizes them to 64x64:

```powershell
py -3.12 -m fixed_noise_diffusion.train --config celeba64_base.yaml
```

For self-hosted GPU runners, use:

```text
.github/workflows/wp2-celeba64-validation.yml
```

The workflow mirrors the STL-10 validation matrix over Gaussian, fixed-pool 1k,
10k, and 100k conditions with seeds 0, 1, and 2. CelebA is larger and uses
64x64 noise pools, so the default workflow budget is 50 epochs and compact
artifacts only. The public torchvision CelebA download depends on Google Drive
and may hit quota limits, so the workflow assumes by default that the dataset has
already been staged in the persistent cache. Set `download_data=true` only when
the runner can access the upstream CelebA files.

## Sample-Quality Evaluation

Evaluate saved checkpoints with Inception FID/KID:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m fixed_noise_diffusion.evaluate_sample_quality `
  --sweep-dir runs/wp2_50ep_3seed `
  --output-dir runs/wp2_fid2048_10k `
  --epochs 50 `
  --sample-count 10000 `
  --fid-feature 2048
```

For a larger FID run, use the training split for real statistics:

```powershell
py -3.12 -m fixed_noise_diffusion.evaluate_sample_quality `
  --sweep-dir runs/wp2_100ep_reduced `
  --output-dir runs/wp2_fid2048_50k `
  --epochs 100 `
  --sample-count 50000 `
  --real-count 50000 `
  --real-split train `
  --fid-feature 2048
```

Combine one or more `sample_quality.csv` outputs and optionally join denoising
gap summaries:

```powershell
py -3.12 -m fixed_noise_diffusion.summarize_sample_quality `
  --quality runs/wp2_fid2048_10k_gpu0 `
  --quality runs/wp2_fid2048_10k_gpu1 `
  --gap-summary runs/wp2_50ep_gap_summary.csv `
  --output-dir runs `
  --prefix wp2_fid2048_10k_epoch50
```

Evaluate timestep-local denoising gaps from saved checkpoints:

```powershell
py -3.12 -m fixed_noise_diffusion.evaluate_timestep_diagnostics `
  --sweep-dir runs/wp2_100ep_reduced `
  --output-dir runs/wp2_timestep_diagnostics `
  --epochs 50,100 `
  --timesteps 0,25,50,100,200,400,600,800,999 `
  --batches 16
```

Evaluate fixed-pool generalization from saved checkpoints:

```powershell
py -3.12 -m fixed_noise_diffusion.evaluate_pool_generalization `
  --sweep-dir runs/wp2_100ep_reduced `
  --output-dir runs/wp2_pool_generalization `
  --epochs 100 `
  --batches 16
```

For future fixed-pool training runs, the same held-out pool diagnostic can be
logged during checkpoint evaluation without saving checkpoints:

```powershell
py -3.12 -m fixed_noise_diffusion.train --config cifar10_fixed_pool_1k.yaml `
  --set evaluation.enable_heldout_pool=true
```

This adds `heldout_pool_den_loss`, `heldout_pool_gap`, and
`gaussian_minus_heldout_gap` to eval rows. Gaussian runs ignore this option.

## Key Diagnostic

The main WP2 diagnostic is:

```text
denoising_gap = held_out_gaussian_denoising_loss - training_law_denoising_loss
```

For fixed pools, a positive and growing gap means the model is fitting the
realized reusable noise law better than held-out fresh Gaussian noise. That is
the first signal of support-limited overspecialization.

## Citation and License

This repository is released under the MIT License. See `LICENSE`.

If you use this code, please cite the software release and the accompanying
paper. Citation metadata is provided in `CITATION.cff`.

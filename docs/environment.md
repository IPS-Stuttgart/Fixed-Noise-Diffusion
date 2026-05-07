# Environment and Reproducibility Notes

This repository is intentionally lightweight. The Python package metadata lists
the non-PyTorch dependencies, while PyTorch and torchvision should be installed
according to the local GPU or CPU setup.

## Recommended paper environment

Use Python 3.12 for reproducing the paper runs.

The package metadata permits Python 3.13 as well, but the current paper-facing
scripts and self-hosted-runner workflows have primarily been exercised with
Python 3.12.

A typical Linux/macOS setup is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[test]'
```

For sample-quality metrics:

```bash
python -m pip install -e '.[metrics,test]'
```

A typical Windows PowerShell setup is:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python -m pip install -e '.[test]'
```

For sample-quality metrics on Windows:

```powershell
.\.venv\Scripts\python -m pip install -e '.[metrics,test]'
```

Use a different PyTorch installation command when the target machine has a
different CUDA runtime or should run CPU-only. Install the PyTorch wheel
appropriate for the machine first, then install this repository in editable mode.

## CPU-only smoke test

The smoke configuration verifies installation and basic execution. It does not
reproduce the paper experiments.

Linux/macOS:

```bash
PYTHONPATH=src python -m fixed_noise_diffusion.train --config smoke.yaml
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m fixed_noise_diffusion.train --config smoke.yaml
```

## Paper-run artifact metadata

Each training run writes:

- `config.yaml`
- `metrics.csv`
- `metrics.jsonl`
- `run_metadata.json`
- `run_summary.json`

The `run_metadata.json` file records the environment and execution context needed
to audit a run, including:

- command line
- current working directory
- Git branch and commit
- Python executable, version, and platform
- Torch version
- CUDA availability and CUDA version
- cuDNN version
- selected device and device name
- deterministic-algorithm flag
- run seed
- data configuration
- noise mode, pool size, pool memory, and whitening flag

For paper archival, keep the raw per-run artifacts together with the curated
summary CSV files and generated paper tables and figures.

## Environment check command

Use this command to print the most important runtime versions before launching a
large experiment batch:

```bash
python -c "import platform, sys, torch, torchvision; print('python=', sys.version); print('platform=', platform.platform()); print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda=', torch.version.cuda)"
```

The same information is captured automatically in each run's `run_metadata.json`.

## Reproducibility policy

The canonical reproducibility record for this project is the per-run metadata and
artifact chain: command, Git commit, config hash, run config, metrics, curated CSV
summaries, and generated paper tables or figures. A separate frozen requirements
file is not required for the paper artifact path.

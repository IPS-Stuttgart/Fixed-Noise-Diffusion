# Exposure-Matched Noise Persistence Study

This opt-in Linux/GPU runner tests temporal noise ordering, not a claim that
fixed pools improve diffusion training. Legacy training and evaluation remain
unchanged. Scheduled configurations are rejected by legacy noise reconstruction.

## Frozen protocol

The production protocol is defined in `persistence_protocol.protocol()`:
CIFAR-10, base64 U-Net, cosine schedule, 1000 timesteps, batch 128, AdamW at
0.0002, zero weight decay, no augmentation, float32 pools/training, AMP off,
100 epochs and 39000 updates. TF32 is disabled and deterministic algorithms are
required. Full CIFAR-10 training data are used with incomplete batches dropped.

Pilot seed 0 runs Gaussian, persistent 1k, refresh-10, shuffled-union-10,
refresh-1, and shuffled-union-1. A bank contains 100 independently generated
1k blocks, using seeds `4242 + 1000 * replicate + block`. All conditions share
the first block. Index and permutation streams have separate seed domains.
Each union is an exact permutation of its refresh counterpart's complete draw
sequence. Membership and final per-template exposure counts must match exactly.
Exposure histories during training intentionally differ. The intervention also
changes which template is paired with a given image/timestep; it does not identify
an internal network mechanism or isolate every possible temporal statistic.

Initialization, dropout randomness, data order and timestep draws are matched
within each replicate. Evaluation snapshots preserve all training RNG states and
never increment exposure counts. The current-pool evaluation samples uniformly
from the currently available support; it does not replay the training sequence.
The retired pool is the immediately preceding block. Persistent/union/Gaussian
conditions have no retired-pool measurement. The unseen pool is independently
generated and matches the active support size. Epoch 100 is never refreshed.

A fixed test-set permutation (seed 20261003) reserves the first 2048 images for
development and the remaining 7952 for confirmation endpoint evaluation. This
partition is held out from selection in this study, not claimed historically
unused by earlier work. Curves and timestep diagnostics use development images.
Endpoints average four independent draws; fresh noise/images/timesteps are paired
across conditions. Pilot seed 0 is excluded from confirmation inference.

## Gates and budget

The pilot must reproduce persistent-1k fresh-minus-current loss and excess fresh
loss over Gaussian of at least 0.05 each. A refresh-minus-union fresh-loss contrast
must exceed both 0.005 in magnitude and three paired Monte Carlo standard errors.
Choose the largest qualifying absolute contrast, with interval 10 winning exact
ties. The selected interval, direction and six endpoint hashes are frozen before
confirmation. No qualifying contrast means stop after six full training runs.

Confirmation runs only Gaussian, persistent, selected refresh and its union for
new seeds 1, 2 and 3: twelve more runs, maximum eighteen. Each replicate must have
matched hardware/software, source and initial model weights. Require the selected
direction in all three, mean magnitude at least 0.005, and the paired 95% t interval
(df=2) to exclude zero. Report the small-sample assumptions and individual effects.
A failed gate is inconclusive, not evidence of equivalence. Seeds jointly vary
model/data randomness and pool realization; their variance components are not
separately estimated. Technical smoke tests are not scientific replicates.

## Commands

Install the project with PyTorch, torchvision, test and metrics dependencies in
an isolated Python 3.12 environment. Preserve its package freeze with the study.
The coordinator launches one worker per idle GPU (under 256 MiB and 5% utilization),
leaves other jobs alone, and stops its own workers on failure. Existing results
are checked, not silently overwritten. Run paired conditions on one host/GPU model.

```bash
python -m pytest tests -q
python -m fixed_noise_diffusion.persistence_study init --root runs/persistence-smoke --smoke
python -m fixed_noise_diffusion.persistence_study execute --root runs/persistence-smoke --stage pilot --gpus 0,1
python -m fixed_noise_diffusion.persistence_study init --root runs/persistence-v1 --smoke-receipt runs/persistence-smoke/smoke_receipt.json
python -m fixed_noise_diffusion.persistence_study execute --root runs/persistence-v1 --stage pilot --data-dir /data/cifar10 --gpus 0,1
python -m fixed_noise_diffusion.persistence_study status --root runs/persistence-v1
python -m fixed_noise_diffusion.persistence_study decide --root runs/persistence-v1
python -m fixed_noise_diffusion.persistence_study execute --root runs/persistence-v1 --stage confirmation --data-dir /data/cifar10 --gpus 0,1
python -m fixed_noise_diffusion.persistence_study quality --root runs/persistence-v1 --device cuda
```

Confirmation is refused unless the pilot passes and all original artifacts still
verify. Sample quality is refused unless confirmation passes. It evaluates all
twelve final checkpoints using FID2048, 50000 generated images, all 50000 training
images as reference, 50-step DDIM and common sampling seed settings. KID is also
recorded by the existing evaluator. Incompatible protocols are rejected rather
than combined. Sample-quality files refer to checksum-verified checkpoints.

On another host, copy the frozen source, manifest, pilot decision and compact run
artifacts, use the same environment, then use `run --condition ... --seed ...` for
all four conditions of a whole confirmation replicate. Copy its compact artifacts
back before aggregation. Do not run duplicate workers for a replicate on different
hosts. Preserve original checkpoints. For the twelve-checkpoint sample-quality
stage, copy final checkpoints, receipts and configs to one evaluation host with
the recorded dataset paths available; keep the sampling environment fixed.

## Recovery and artifacts

Source and protocol hashes are frozen before the first run. Do not edit source
in a running study. Epoch-boundary checkpoints contain model, optimizer, RNG,
schedule cursor, exposure counters, evaluation rows and active-pool fingerprints.
Hash receipts validate recovery and retained checkpoints. Interrupted workers may
resume using `run ... --resume` or the coordinator. Legacy checkpoints are not
treated as exactly resumable. A recorded numerical/protocol failure blocks
automatic continuation and requires investigation; it is never deleted by the
runner. A low-disk guard protects recovery files.

Each run produces `manifest.json`, `evaluation.csv`, `endpoint.json`, hash receipts,
checkpoints and `status.json`. The study produces immutable decisions, coordinator
status/logs, and `report/endpoints.csv`, `report/decisions.json`, a diagnostic figure
and a summary. Preserve raw measurements, including unfavorable results. Export
compact artifacts into the private paper repository only after validating them;
do not publish datasets, credentials or large checkpoint files.

# Large dataset caches

Large torchvision datasets used by the self-hosted validation workflows should be staged outside the GitHub workspace so they survive between jobs on the same runner.

The default cache root is:

```text
$HOME/github-runner/.cache/datasets/fixed-noise-diffusion
```

The workflow below prepares the large dataset cache without starting the expensive training matrices:

```text
.github/workflows/prepare-large-dataset-caches.yml
```

Use it from **Actions -> Prepare Large Dataset Caches -> Run workflow**.

Recommended inputs:

- `dataset=stl10` to prepare STL-10 `train+unlabeled` and `test`.
- `dataset=celeba64` to validate or prepare the CelebA cache used by the CelebA-64 workflow.
- `dataset=all` only when both datasets should be checked on the same runner.
- `data_dir=` empty to use the default cache root above.
- `runner_labels_json=["self-hosted", "Linux"]` or the label set for the intended dataset-capable runner.

CelebA is treated conservatively. By default, the pre-warmer and validation workflow only verify a staged CelebA cache and fail if it is missing. Set `download_celeba=true` in the pre-warmer, or `download_data=true` in the CelebA-64 validation workflow, only when the runner can reliably access the upstream torchvision/CelebA download source.

The validation workflows still check the dataset path themselves. The pre-warmer is only a convenience to avoid first-run downloads inside large training matrices; it is not the only correctness gate.

The workflows use `flock` files inside the cache root so that concurrent matrix jobs do not download or validate the same dataset concurrently.

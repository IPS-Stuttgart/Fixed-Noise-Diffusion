# Release process

Paper-facing releases are tagged snapshots of the code used for the fixed-noise diffusion experiments.

## v0.1.0-paper-submission

Intended paper-submission snapshot for the accompanying draft:

> Fixed Noise Supports in Diffusion Training: Reproducibility and Finite-Support Specialization

This release should include:

- DDPM training code for Gaussian and fixed-pool noise laws.
- CIFAR-10, STL-10, and CelebA-64 experiment configurations.
- Denoising-gap evaluation utilities for train-law, held-out-pool, and fresh-Gaussian comparisons.
- Sample-quality evaluation utilities for optional FID/KID sanity checks.
- Repository metadata needed for reuse and citation: `LICENSE`, `CITATION.cff`, and package license metadata.

The release does not include datasets, generated samples, large run directories, or model checkpoints. Paper-specific curated result tables and figures are maintained in the companion paper repository.

## Recommended tag command

After the metadata commit is merged or accepted as the paper-facing snapshot, create an annotated tag:

```bash
git tag -a v0.1.0-paper-submission -m "Paper submission release for fixed-noise diffusion experiments"
git push origin v0.1.0-paper-submission
```

Then create a GitHub Release from the tag and archive it through Zenodo or an equivalent long-term artifact service if a DOI is desired.

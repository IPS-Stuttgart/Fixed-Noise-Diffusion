# WP2 CIFAR-10 Pool-Dtype Control

This control checks whether the fixed-pool denoising-gap effect could be explained by storing reusable noise templates
in half precision. The default fixed-pool implementation stores pools as `float16` for memory efficiency, while fresh
Gaussian samples are generated directly as `float32`. The control repeats selected CIFAR-10 fixed-pool runs with
`noise.pool_dtype=float16` and `noise.pool_dtype=float32` under the same training setup.

## Matrix

The manual workflow `.github/workflows/wp2-cifar10-pool-dtype-control.yml` runs:

- dataset: CIFAR-10,
- model/config: `cifar10_base.yaml` with `diffusion.beta_schedule=cosine`,
- epochs: 100 by default,
- seeds: `0,1,2`,
- pool seed: `4242`,
- pool sizes: `250, 1000, 10000, 100000`,
- pool dtypes: `float16, float32`,
- checkpoint evaluation epochs: `1,5,10,25,50,100`,
- sample generation and FID/KID disabled for compact diagnostic artifacts.

The acceptance criterion is qualitative rather than bitwise equality: the `float32`
runs should preserve the same support-size hierarchy as the default `float16` runs.
In particular, small pools should retain large positive denoising gaps, the 10k pool
should retain at most a smaller residual gap, and the 100k pool should remain near
zero.

## Summarizing artifacts

The workflow has a downstream summary job. To summarize downloaded compact artifacts manually, run:

```bash
python -m fixed_noise_diffusion.summarize_pool_dtype_control \
  --runs-root dtype-control-artifacts \
  --output-dir dtype-control-summary \
  --prefix wp2_cifar10_pool_dtype_control_100ep \
  --epoch 100
```

This writes:

- `wp2_cifar10_pool_dtype_control_100ep_eval_rows.csv`, containing one row per run and checkpoint epoch,
- `wp2_cifar10_pool_dtype_control_100ep_gap_summary.csv`, grouped by dataset, pool size, epoch, and pool dtype,
- `wp2_cifar10_pool_dtype_control_100ep.png`, a log-scale pool-size plot comparing `float16` and `float32` gaps.

The summary CSV includes `float32_minus_float16_gap_mean` for each matched
pool-size/epoch pair. This value should be interpreted relative to the much larger
between-pool-size effect, not as a requirement that both storage formats match
exactly.

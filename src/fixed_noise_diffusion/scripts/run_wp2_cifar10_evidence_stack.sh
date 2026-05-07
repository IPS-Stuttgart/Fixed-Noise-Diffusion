#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/wp2_cifar10_pool_scaling_100ep}"
DATA_DIR="${DATA_DIR:-data}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-true}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-false}"
NO_PLOT="${NO_PLOT:-false}"

POOL_SIZES=(250 500 1000 2000 5000 10000 20000 50000 100000)
SEEDS=(0 1 2)

pool_label() {
  local pool_size="$1"
  if (( pool_size >= 1000 && pool_size % 1000 == 0 )); then
    printf '%sk' "$(( pool_size / 1000 ))"
  else
    printf '%s' "$pool_size"
  fi
}

run_train() {
  local run_name="$1"
  local seed="$2"
  local noise_mode="$3"
  local pool_size="$4"
  local whiten="$5"

  local args=(
    -m fixed_noise_diffusion.train
    --config cifar10_base.yaml
    --set "run_name=${run_name}"
    --set "output_dir=${OUTPUT_ROOT}"
    --set "seed=${seed}"
    --set "data.data_dir=${DATA_DIR}"
    --set "data.download=${DOWNLOAD_DATA}"
    --set "diffusion.beta_schedule=cosine"
    --set "training.epochs=100"
    --set "training.checkpoint_epochs=[1,5,10,25,50,100]"
    --set "training.save_checkpoint=${SAVE_CHECKPOINTS}"
    --set "evaluation.enable_metrics=false"
    --set "noise.mode=${noise_mode}"
    --set "noise.whiten=${whiten}"
  )

  if [[ "$pool_size" == "" ]]; then
    args+=(--set "noise.pool_size=null")
  else
    args+=(--set "noise.pool_size=${pool_size}")
  fi

  printf '==> %q ' "$PYTHON" "${args[@]}"
  printf '\n'
  "$PYTHON" "${args[@]}"
}

for seed in "${SEEDS[@]}"; do
  run_train \
    "wp2_100ep_cifar10_gaussian_seed${seed}" \
    "$seed" \
    gaussian \
    "" \
    false
done

for seed in "${SEEDS[@]}"; do
  for pool_size in "${POOL_SIZES[@]}"; do
    label="$(pool_label "$pool_size")"
    run_train \
      "wp2_100ep_cifar10_fixed_pool_${label}_seed${seed}" \
      "$seed" \
      fixed_pool \
      "$pool_size" \
      false
  done
done

for seed in "${SEEDS[@]}"; do
  for pool_size in "${POOL_SIZES[@]}"; do
    label="$(pool_label "$pool_size")"
    run_train \
      "wp2_100ep_cifar10_fixed_pool_whitened_${label}_seed${seed}" \
      "$seed" \
      fixed_pool_whitened \
      "$pool_size" \
      true
  done
done

if [[ "$NO_PLOT" != "true" ]]; then
  mapfile -t run_dirs < <(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
  if (( ${#run_dirs[@]} > 0 )); then
    "$PYTHON" -m fixed_noise_diffusion.plot_results --runs "${run_dirs[@]}" --output "$OUTPUT_ROOT/wp2_cifar10_pool_scaling_100ep.png"
  fi
fi

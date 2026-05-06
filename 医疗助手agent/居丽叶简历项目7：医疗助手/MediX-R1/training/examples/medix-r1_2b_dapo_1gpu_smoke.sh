#!/bin/bash

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/rivermind-data/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TMPDIR="${TMPDIR:-/root/rivermind-data/tmp}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
TRAIN_FILES="${TRAIN_FILES:-../data/medix-rl-data/data@train}"
VAL_FILES="${VAL_FILES:-../data/medix-rl-data/data@test}"

echo "Running 1-GPU DAPO smoke training with model path: ${MODEL_PATH}"
echo "Train files: ${TRAIN_FILES}"
echo "Val files: ${VAL_FILES}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.format_prompt=./examples/format_prompt/medical_format.jinja \
    data.answer_key=solution \
    data.image_key=image \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.reward.reward_function=./examples/reward_function/medical_smoke.py:compute_score \
    worker.rollout.max_num_batched_tokens=3072 \
    data.rollout_batch_size=1 \
    data.mini_rollout_batch_size=1 \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.fsdp.enable_cpu_offload=True \
    worker.actor.offload.offload_params=True \
    worker.actor.offload.offload_optimizer=True \
    worker.ref.fsdp.enable_cpu_offload=True \
    worker.ref.offload.offload_params=True \
    worker.rollout.n=1 \
    worker.rollout.gpu_memory_utilization=0.35 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=False \
    trainer.experiment_name=medix-r1_2b_dapo_1gpu_smoke \
    trainer.total_epochs=1 \
    trainer.max_steps="${MAX_STEPS:-1}" \
    trainer.logger='["file"]' \
    trainer.val_before_train=False \
    trainer.val_freq=-1 \
    trainer.save_freq=-1 \
    trainer.n_gpus_per_node=1

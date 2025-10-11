#!/bin/bash

# for x in $(seq 1 40); do ./batch.sh; done

NUM_NODES=4

JOB_NAME=grpo-n5p5_12b-intervlparams-mmpr-tiny-4n
export COMMAND="uv run examples/run_vlm_grpo.py --config examples/configs/vlm_grpo_3B_mmpr_tiny.yaml \
cluster.num_nodes=$NUM_NODES \
checkpointing.checkpoint_dir='results/${JOB_NAME}' \
logger.wandb_enabled=True \
logger.wandb.name='${JOB_NAME}'"

export NRL_FORCE_REBUILD_VENVS=true  # Force Ray workers to rebuild venvs with mathruler

CONTAINER=/lustre/fsw/portfolios/llmservice/users/jseppanen/sqsh/nemo-rl-main-2b55598e.sqsh \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=llmservice_fm_vision \
    --job-name=nemo-rl-${JOB_NAME} \
    --partition=batch_block1 \
    --dependency=singleton \
    --time=4:00:00 \
    --gres=gpu:8 \
    ray.sub

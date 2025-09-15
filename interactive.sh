#!/bin/bash
CONTAINER=/lustre/fsw/portfolios/llmservice/users/jseppanen/sqsh/nemo-rl-main-2b55598e.sqsh \
MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/llmservice/users/jseppanen/dev:/code" \
sbatch \
    --nodes=1 \
    --account=llmservice_fm_vision \
    --job-name=nemo-rl-dev:interactive \
    --partition=interactive \
    --time=4:00:00 \
    --gres=gpu:8 \
    ray.sub

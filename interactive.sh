#!/bin/bash
HF_HOME=/lustre/fsw/portfolios/llmservice/users/jseppanen/.cache/huggingface \
srun \
    --pty \
    -p interactive \
    -A llmservice_fm_vision \
    --no-container-mount-home \
    --container-mounts "/lustre:/lustre,/lustre/fsw/portfolios/llmservice/users/jseppanen/dev:/code" \
    --container-image /lustre/fsw/portfolios/llmservice/users/jseppanen/sqsh/nemo-rl-main-2b55598e.sqsh \
    --gpus 8 \
    --exclusive \
    --mpi=pmix \
    --job-name "nemo-rl-dev:interactive" \
    -t 4:00:00 \
    bash -l

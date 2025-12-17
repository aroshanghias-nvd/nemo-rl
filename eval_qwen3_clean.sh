#!/bin/bash
#SBATCH --job-name=eval-qwen3-clean
#SBATCH --partition=batch_block1,batch_short,batch_singlenode
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH -A llmservice_fm_vision
#SBATCH --mem=0


MODEL_NAME="Qwen/Qwen3-VL-8B-Thinking"
BENCHMARK="MathVista_MINI"
EXPERIMENT_NAME="qwen3-vl-8b-thinking-sysprompt-v2"

# Paths
NEMORL=/lustre/fsw/portfolios/llmservice/users/ikarmanov/nemo-rl
VLMEVALKIT=/lustre/fsw/portfolios/llmservice/users/ikarmanov/VLMEvalKitMcore
QWEN3_CHAT_TPL="$NEMORL/qwen3_chat_template.jinja"
CONTAINER_IMAGE=/lustre/fsw/portfolios/llmservice/users/matthieul/docker/megatron-dev-img-05142025-pytorch-dev-te-cd37379-energon-fix_repeat_dataset-mamba-fix-vlmeval-vllm-budget.sqsh

CACHE_ROOT=/lustre/fsw/portfolios/llmservice/users/ikarmanov/.cache
OUTPUT_DIR="$NEMORL/results/${EXPERIMENT_NAME}/eval/step_0"

# Setup output directories
mkdir -p "${OUTPUT_DIR}/benchmark_logs"
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

# Run command executed inside container
run_command='
set -e

#---------------------------------------------------------------------------
# Cache configuration (all on lustre, not home)
# Only HF_HOME is needed - TRANSFORMERS_CACHE is deprecated
#---------------------------------------------------------------------------
CACHE_ROOT='"$CACHE_ROOT"'
mkdir -p "$CACHE_ROOT"/{huggingface,torch,triton,pip,tmp}

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export XDG_CACHE_HOME="$CACHE_ROOT"
export TMPDIR="$CACHE_ROOT/tmp"
export VLLM_CACHE_ROOT="$CACHE_ROOT"

#---------------------------------------------------------------------------
# HuggingFace settings
#---------------------------------------------------------------------------
set -a
source '"$NEMORL"'/.env 2>/dev/null || true
set +a
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_OFFLINE=0

#---------------------------------------------------------------------------
# vLLM settings
#---------------------------------------------------------------------------
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SHOW_VLLM_LOGS=1
export SERVE_BIN="'"$NEMORL"'/qwen3_serve.py"
# System prompt v2 - Nemotron-style detailed reasoning instructions
export VLLM_SYSTEM_PROMPT="You are an AI assistant that rigorously follows this response protocol: 1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags. 2. After the thinking section, provide a clear, concise, and direct answer to the users question. Separate the answer from the think section with a newline. Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section."

#---------------------------------------------------------------------------
# VLMEvalKit settings
#---------------------------------------------------------------------------
export OPENAI_API_BASE=https://prod.api.nvidia.com/llm/v1/azure/chat/completions
export LMUData=/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/vlmevalkit_cache
export RD_TABLEBENCH_SRC=/lustre/fsw/portfolios/llmservice/users/amalasanjayd/dev/rd-tablebench

#---------------------------------------------------------------------------
# Distributed settings
#---------------------------------------------------------------------------
export RANK=${SLURM_PROCID:-0}
export LOCAL_RANK=${SLURM_LOCALID:-0}
export WORLD_SIZE=${SLURM_NTASKS:-1}
export LOCAL_WORLD_SIZE=${SLURM_NTASKS_PER_NODE:-1}

cd '"$VLMEVALKIT"'

#---------------------------------------------------------------------------
# Dependency installation (rank 0 only, others wait)
#---------------------------------------------------------------------------
INSTALL_SENTINEL='"${OUTPUT_DIR}"'/.deps_installed_v1

if [ "${SLURM_PROCID:-0}" -eq 0 ]; then
    echo "Rank 0/$WORLD_SIZE: Installing vLLM 0.12..."
    pip install -q --root-user-action=ignore "vllm>=0.11.0" openai 2>&1 | tail -3
    python -c "import vllm; print(\"vLLM\", vllm.__version__)"
    touch "$INSTALL_SENTINEL"
else
    sleep 8
    until [ -f "$INSTALL_SENTINEL" ]; do sleep 2; done
    echo "Rank $RANK/$WORLD_SIZE: Ready"
fi

sleep 2

#---------------------------------------------------------------------------
# Run evaluation with official Qwen3-VL Thinking settings
# Using GPT-4o for answer extraction (instead of default GPT-4o-mini)
#---------------------------------------------------------------------------
python run.py \
    --data '"$BENCHMARK"' \
    --model vllm_local \
    --verbose \
    --work-dir '"$OUTPUT_DIR"' \
    --vllm-model-path '"$MODEL_NAME"' \
    --vllm-chat-tpl '"$QWEN3_CHAT_TPL"' \
    --vllm-autospawn \
    --reasoning \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --vllm-max-tokens 40960 \
    --judge gpt-4o

echo "Evaluation complete!"
'

#===============================================================================
# Launch
#===============================================================================
echo "=========================================="
echo "Qwen3-VL Evaluation (Clean Version)"
echo "=========================================="
echo "Model:      $MODEL_NAME"
echo "Benchmark:  $BENCHMARK"
echo "Output:     $OUTPUT_DIR"
echo "=========================================="

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"
rm -f "${OUTPUT_DIR}/.deps_installed_v"*

srun -l --verbose --mpi=pmix \
    --container-image "$CONTAINER_IMAGE" \
    --container-mounts "/lustre" \
    --error="${OUTPUT_DIR}/benchmark_logs/eval-qwen3_${SLURM_JOB_ID}_${DATETIME}.err" \
    --output="${OUTPUT_DIR}/benchmark_logs/eval-qwen3_${SLURM_JOB_ID}_${DATETIME}.log" \
    sh -c "$run_command"


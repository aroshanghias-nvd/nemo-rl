#!/bin/bash

input_path=$1
output_dir=$2
shard_id=$3
num_shards=$4
source /venv/bin/activate


python infer_keensight_qwen.py $input_path $output_dir --shard-id=$shard_id --num-shards=$num_shards
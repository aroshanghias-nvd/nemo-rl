#!/bin/bash

input_path=$1
output_dir=$2
shard_id=$3
num_shards=$4
mode=$5
source /venv/bin/activate
pip install mathruler
pip install pylatexenc

python infer_eval_filter.py $input_path $output_dir --shard-id=$shard_id --num-shards=$num_shards --mode=$mode
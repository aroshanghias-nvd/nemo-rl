#!/bin/bash

input_path=$1
output_dir=$2
image_root=$3
shard_id=$4
num_shards=$5
mode=$6
source /venv/bin/activate
pip install mathruler
pip install pylatexenc

python infer_eval_filter.py $input_path $output_dir $image_root --shard-id=$shard_id --num-shards=$num_shards --mode=$mode
#!/usr/bin/env bash
cd ./slurpp

export OUTPUT_DIR=../outputs/training/${1}
export BASE_CKPT_DIR="$(pwd)/../models"
export SCRATCH_DATA_DIR=  #FILL IN YOUR DATA DIRECTORY HER
CONFIG="../config/${1}.yaml"

python train.py --config $CONFIG --job_name_prefix test_ --output_dir $OUTPUT_DIR --add_datetime_prefix 

GPU_ID=0

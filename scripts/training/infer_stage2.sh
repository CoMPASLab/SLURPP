#!/usr/bin/env bash
cd ./slurpp

export BASE_CKPT_DIR="$(pwd)/../models"
export SCRATCH_DATA_DIR=  #FILL IN YOUR DATA DIRECTORY HERE

#FILL IN YOUR DIFFUSION TRAINING OUTPUT DIRECTORY HERE
RUN_DIR=
#FILL IN YOUR CHECKPOINT ITERATION HERE
# For example, if your checkpoint is iter_003133, set CKPT_ITR=iter_003133
# If you want to use the latest checkpoint, you can set CKPT=latest
CKPT_ITR=latest
CKPT=${RUN_DIR}/checkpoint/${CKPT_ITR}
#FILL IN THE OUTPUT DIRECTORY OF GENRTED PAIRED IMAGES HERE
OUTPUT_DIR=${RUN_DIR}/${CKPT_ITR}/decoder_train_imgs/ 

GPU_ID=0
EXP_NAME=test
MG_CONFIG=${RUN_DIR}/config.yaml

python infer_stage2.py --config $MG_CONFIG --checkpoint $CKPT --output_dir $OUTPUT_DIR --seed 2024 


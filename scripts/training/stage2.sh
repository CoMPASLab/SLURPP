#!/usr/bin/env bash
export BASE_CKPT_DIR="$(pwd)/models"

cd ./slurpp/stage2


#FILL IN YOUR DIFFUSION TRAINING OUTPUT DIRECTORY HERE
RUN_DIR=
CKPT_ITR=latest
CKPT=${RUN_DIR}/checkpoint/${CKPT_ITR}
OUTPUT_DIR=${RUN_DIR}/${CKPT_ITR}/decoder_fine_tune/
DATA_DIR=${RUN_DIR}/${CKPT_ITR}/decoder_train_imgs/train
#FILL IN YOUR FIELD NAME HERE
# For example, if you want to train on 'clear', set FIELD="clear"
FIELD="clear"

MG_CONFIG=${RUN_DIR}/config.yaml

python train_stage2.py --field $FIELD --data_dir $DATA_DIR --config $MG_CONFIG --output_dir $OUTPUT_DIR --preserve_encoder 
GPU_ID=0

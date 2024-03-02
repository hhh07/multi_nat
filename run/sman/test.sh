#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

savedir=model/sman
dataset=/sda/huzhoujue/NMT/MultiTaskNAT/data-bin/distill_iwslt14.tokenized.de-en
userdir=multitasknat
task=nat_ctc_task
subset=test
time=700
tags_path=${dataset}/dp 




echo "=============Averaging checkpoints============="

# python scripts/average_checkpoints.py \
#     --inputs ${savedir} \
#     --num-best-checkpoints 5 \
#     --output ${savedir}/${time}checkpoint.best_average_5.pt

# python scripts/average_checkpoints.py \
#     --inputs ${savedir} \
#     --num-top-checkpoints 5 \
#     --output ${savedir}/${time}checkpoint.top5_average_5.pt

echo "=============Generating by average============="

python generate.py \
    --path ${savedir}/${time}checkpoint.best_average_5.pt \
    ${dataset} \
    --gen-subset ${subset} \
    --user-dir ${userdir} \
    --task ${task} \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 1 \
    --remove-bpe \
    --quiet \
    --reset-meters \
    --enc-sman-attn-layers 0,1,2,3,4,5  --sman-mode 1 --sman-width 4 \
    --batch-size 256  > ${savedir}/${time}best_test.txt

python generate.py \
    --path ${savedir}/${time}checkpoint.top5_average_5.pt \
    ${dataset} \
    --gen-subset ${subset} \
    --user-dir ${userdir} \
    --task ${task} \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 1 \
    --remove-bpe \
    --enc-sman-attn-layers 0,1,2,3,4,5  --sman-mode 1 --sman-width 4 \
    --batch-size 256  > ${savedir}/${time}top5_output.txt

echo "=============Finish============="
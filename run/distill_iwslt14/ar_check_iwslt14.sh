#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

savedir=model/distill_iwslt14
dataset=/sda/huzhoujue/NMT/MultiTaskNAT/data-bin/distill_iwslt14.tokenized.de-en
userdir=multitasknat
task=ar_ctc_task
subset=test




# echo "=============Averaging checkpoints============="

# python scripts/average_checkpoints.py \
#     --inputs ${savedir} \
#     --num-best-checkpoints 5 \
#     --output ${savedir}/checkpoint.best_average_5.pt

# python scripts/average_checkpoints.py \
#     --inputs ${savedir} \
#     --num-top-checkpoints 5 \
#     --output ${savedir}/checkpoint.top5_average_5.pt

echo "=============Generating by average============="

python generate.py \
    --path ${savedir}/checkpoint.best_average_5.pt \
    ${dataset} \
    --gen-subset ${subset} \
    --user-dir ${userdir} \
    --task ${task} \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 1 \
    --remove-bpe \
    --batch-size 256 > ${savedir}/best5_output.txt

# python generate.py \
#     --path ${savedir}/checkpoint.top5_average_5.pt \
#     ${dataset} \
#     --gen-subset ${subset} \
#     --user-dir ${userdir} \
#     --task ${task} \
#     --iter-decode-max-iter 0 \
#     --iter-decode-eos-penalty 0 \
#     --beam 1 \
#     --remove-bpe \
#     --batch-size 256 > ${savedir}/top5_output.txt

echo "=============Finish============="
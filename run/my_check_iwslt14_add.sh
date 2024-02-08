#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

savedir=model/iwslt14_without_enc_add_enc
dataset=data-bin/iwslt14.tokenized.de-en 
userdir=multitasknat
task=mt_ctc_task
criterion=mt_ctc_loss
arch=mt_ctc_multi
max_token=4096
max_update=250000
update_freq=1


echo "=============Averaging checkpoints============="

python scripts/average_checkpoints.py \
    --inputs ${savedir} \
    --num-top-checkpoints 5 \
    --output ${savedir}/checkpoint.top5_average_5.pt

echo "=============Generating by average============="

python generate.py \
    --path ${savedir}/checkpoint.best_bleu_30.00.pt \
    ${dataset} \
    --gen-subset test \
    --user-dir ${userdir} \
    --task ${task} \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 1 \
    --remove-bpe \
    --print-step \
    --batch-size 256 > ${savedir}/best_output.txt

python generate.py \
    --path ${savedir}/checkpoint.top5_average_5.pt \
    ${dataset} \
    --gen-subset test \
    --user-dir ${userdir} \
    --task ${task} \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 1 \
    --remove-bpe \
    --print-step \
    --batch-size 256 > ${savedir}/top5_output.txt

echo "=============Finish============="
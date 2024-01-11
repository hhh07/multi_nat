#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0

savedir=model/iwslt16
dataset=data-bin/iwslt14.tokenized.de-en
userdir=multitasknat
task=mt_ctc_task
criterion=mt_ctc_loss
arch=mt_ctc_multi
max_token=8192
max_update=250000
update_freq=1
lr=3e-4
layers=6
dim=256
log=${savedir}/train.log

echo "=============Training============="

python train.py \
    --save-dir ${savedir} \
    --user-dir ${userdir} \
    ${dataset} \
    --arch ${arch} \
    --task ${task} \
    --criterion ${criterion} \
    --encoder-layers ${layers} --encoder-embed-dim ${dim} --decoder-layers ${layers} --decoder-embed-dim ${dim} \
    --fp16 \
    --ddp-backend=no_c10d \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr ${lr} --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 4000 \
    --warmup-init-lr '1e-07' --label-smoothing 0 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --max-tokens ${max_token} \
    --max-update ${max_update} \
    --eval-bleu \
    --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 5 | tee -a ${log}



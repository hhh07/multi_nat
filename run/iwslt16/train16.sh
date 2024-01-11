#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0

savedir=model/iwslt16/multi
dataset=data-bin/iwslt16_de-en/bin
userdir=multitasknat
task=mt_ctc_task
criterion=mt_ctc_loss
arch=mt_ctc_multi
max_token=3000
max_update=250000
update_freq=1
layers=5
dim=256
upsample_scale=3 #default=3
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
    --upsample-scale ${upsample_scale} \
    --ddp-backend=no_c10d \
    --shallow-at-decoder-layers 1 \
    --lambda-nat-at 0.5 \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens ${max_token} \
    --update-freq ${update_freq} \
    --max-update ${max_update} \
    --eval-bleu \
    --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 3 \
    --keep-best-checkpoints 5 | tee -a ${log}

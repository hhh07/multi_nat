#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

savedir=model/dp
dataset=data-bin/distill_iwslt14.tokenized.de-en
userdir=multitasknat
task=dp_ctc_task
criterion=dp_ctc_loss
arch=dp_ctc_multi
max_token=8192
max_update=250000
update_freq=2
layers=6
dim=512
upsample_scale=2 #default=3
lambda_nat_at=0.5
log=${savedir}/train.log
tag_path=data-bin/distill_iwslt14.tokenized.de-en/dp
tgt_pos=pos
tgt_dphead=dphead
tgt_dplable=dplable
echo "=============Training============="

python train.py \
    --save-dir ${savedir} \
    --user-dir ${userdir} \
    ${dataset} \
    --arch ${arch} \
    --task ${task} \
    --criterion ${criterion} --lambda-nat-at ${lambda_nat_at} \
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
    --keep-best-checkpoints 5 \
    --tags-path ${tag_path} | tee -a ${log}
    # --is-random \
    # --share-at-decoder \

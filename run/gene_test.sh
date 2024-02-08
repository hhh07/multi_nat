savedir=model/未蒸馏iwslt14
dataset=data-bin/iwslt14.tokenized.de-en 
userdir=multitasknat
task=mt_ctc_task
src=de
tgt=en


python generate.py \
    --path ${savedir}/checkpoint.best_average_5.pt \
    ${dataset} \
    --gen-subset test \
    --user-dir ${userdir} \
    --task ${task} \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --iter-decode-with-beam 5 \
    --remove-bpe \
    --print-step \
    --source-lang ${src} --target-lang ${tgt} \
    --batch-size 256 > ${savedir}/test5.log
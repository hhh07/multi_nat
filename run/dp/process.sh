echo "binary"
TEXT=/sda/huzhoujue/NMT/MultiTaskNAT/data-bin/dp_iwslt14-de-en
savepath=/sda/huzhoujue/NMT/MultiTaskNAT/data-bin/distill_iwslt14.tokenized.de-en/dp0

# fairseq-preprocess --source-lang pos  --only-source \
#     --trainpref $TEXT/train --validpref $TEXT/valid \
#     --destdir ${savepath}\
#     #--srcdict /sda/huzhoujue/NMT/MultiTaskNAT/data-bin/dp_iwslt14-de-en/dphead_dict.txt

# fairseq-preprocess --source-lang dphead  --only-source \
#     --trainpref $TEXT/train --validpref $TEXT/valid \
#     --destdir ${savepath} \


# fairseq-preprocess --source-lang dplable  --only-source \
#     --trainpref $TEXT/train --validpref $TEXT/valid \
#     --destdir ${savepath} \


### test没有和valid+train一起生成。之后一起生成

## 加test
fairseq-preprocess --source-lang pos  --only-source \
    --testpref $TEXT/test \
    --destdir ${savepath}\
    --srcdict /sda/huzhoujue/NMT/MultiTaskNAT/data-bin/distill_iwslt14.tokenized.de-en/dp/dict.pos.txt
fairseq-preprocess --source-lang dphead  --only-source \
    --testpref $TEXT/test \
    --destdir ${savepath}\
    --srcdict /sda/huzhoujue/NMT/MultiTaskNAT/data-bin/distill_iwslt14.tokenized.de-en/dp/dict.dphead.txt
fairseq-preprocess --source-lang dplable  --only-source \
    --testpref $TEXT/test \
    --destdir ${savepath}\
    --srcdict /sda/huzhoujue/NMT/MultiTaskNAT/data-bin/distill_iwslt14.tokenized.de-en/dp/dict.dplable.txt

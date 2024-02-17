#python main.py --model-dir /sda/huzhoujue/NMT/MultiTaskNAT/model/dp
cd log_util
dir=/sda/huzhoujue/NMT/MultiTaskNAT/model
python main.py --model-dir-list  ${dir}/0dp-1,${dir}/0dp,${dir}/distill_iwslt14-de-en_ctc
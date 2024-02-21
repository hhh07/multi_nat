#python main.py --model-dir /sda/huzhoujue/NMT/MultiTaskNAT/model/dp
cd log_util
dir=/sda/huzhoujue/NMT/MultiTaskNAT/model
python main.py --model-dir-list  ${dir}/0dp-3,${dir}/enhance,${dir}/distill_iwslt14-de-en_ctc
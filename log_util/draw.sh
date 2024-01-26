#python main.py --model-dir /sda/huzhoujue/NMT/MultiTaskNAT/model/dp
cd log_util
dir=/sda/huzhoujue/NMT/MultiTaskNAT/model
python main.py --model-dir-list  ${dir}/dp,${dir}/distill_iwslt14_mtl_512-16k
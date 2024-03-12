#python main.py --model-dir /sda/huzhoujue/NMT/MultiTaskNAT/model/dp
cd log_util
dir=/mnt/sda1_hd/huzhoujue/multi_nat/model

python main.py \
    --model-dir-list  ${dir}/sman,${dir}/sman-old \
    --x-min 0 --x-max 900 \
    --y-min 31 --y-max 35.5 \
    --show-extra-info

# 如果要show-extra-info，请把作为x基准线的实验文件夹放在model-dir-list第一个位置

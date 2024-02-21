cd 01draw_pic/len_bleu
dir=/sda/huzhoujue/NMT/MultiTaskNAT/01draw_pic/len_bleu/enhance_old
log=${dir}/test.log
src=${dir}/src.txt
target=${dir}/target.txt
gen=${dir}/gen.txt
output=${dir}/output

grep ^H ${log} | cut -f3- | sed 's/@@ //g' > ${gen}
grep ^S ${log} | cut -f2- | sed 's/@@ //g' > ${src} 
grep ^T ${log} | cut -f2- | sed 's/@@ //g' > ${target} 


python split.py --input-src ${src} --input-gen ${gen} --input-target ${target} --output ${output}


cd /sda/huzhoujue/NMT/MultiTaskNAT

for i in {0,10,20,30,40,50,60,70}; do
    k=9
    file=$i-$((i+k))
    file_gen=${file}.gen
    file_target=${file}.target
    fairseq-score --sys ${output}/$file_gen --ref ${output}/$file_target >> ${dir}/bleu.log
done
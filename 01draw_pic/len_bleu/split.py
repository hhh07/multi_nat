import os
import argparse
import time



def write_sentences_to_folders(sentence_lengths_target, sentence_lengths_gen, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for length_group, sentences in sentence_lengths_target.items():
        file_name = f'{length_group}-{length_group + 9}.target'
        file_path = os.path.join(output_folder, file_name)
        with open(file_path, 'w', encoding='utf-8') as output_file:
            for sentence in sentences:
                    output_file.write(str(sentence))
    for length_group, sentences in sentence_lengths_gen.items():
        file_name = f'{length_group}-{length_group + 9}.gen'
        file_path = os.path.join(output_folder, file_name)
        with open(file_path, 'w', encoding='utf-8') as output_file:
            for sentence in sentences:
                    output_file.write(str(sentence))

def categorize_sentences(input_src_path,input_target_path,input_gen_path):
    sentence_lengths_src = {}
    sentence_lengths_target = {}
    sentence_lengths_gen = {}
    with open(input_src_path, 'r', encoding='utf-8') as file_src:
        lines_src = file_src.readlines()
    with open(input_target_path, 'r', encoding='utf-8') as file_target:
        lines_target = file_target.readlines()
    with open(input_gen_path, 'r', encoding='utf-8') as file_gen:
        lines_gen = file_gen.readlines()
    for index, line in enumerate(lines_src):
            sentence = line.strip().split()
            length = len(sentence)

            # 将长度按照 10 的间隔进行分组
            length_group = length // 10 * 10
            #大于50的分为一组
            if(length_group >= 70):
                length_group = 70
            if length_group not in sentence_lengths_target:
                sentence_lengths_target[length_group] = []
                sentence_lengths_gen[length_group] = []

            sentence_lengths_target[length_group].append(lines_target[index])
            sentence_lengths_gen[length_group].append(lines_gen[index])

    return sentence_lengths_target, sentence_lengths_gen

def main(args):
    input_src_path = args.input_src
    input_target_path = args.input_target
    input_gen_path = args.input_gen
    output_folder = args.output

    sentence_lengths_target, sentence_lengths_gen = categorize_sentences(input_src_path,input_target_path,input_gen_path)
    write_sentences_to_folders(sentence_lengths_target, sentence_lengths_gen, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-src", default="", type=str)
    parser.add_argument("--input-target", default="", type=str)
    parser.add_argument("--input-gen", default="", type=str)
    parser.add_argument("--output", default="output", type=str)

    main(parser.parse_args())

import argparse
import time

import matplotlib.pyplot as plt
from typing import List

from entity.FairseqLogEntity import FairseqLogEntity


def parse_single_log(log_dir_path: str):
    # 生成日志实体
    fairser_log_entity = FairseqLogEntity.get_instance_by_folder(log_dir_path)

    # 创建图片
    plt.figure(figsize=(15, 6))

    # 第一个子图，放Bleu Score曲线
    plt.subplot(1, 2, 1)
    x_epoch = sorted(fairser_log_entity.valid_info_dict.keys())
    y_valid_bleu = [fairser_log_entity.valid_info_dict[epoch].bleu for epoch in x_epoch]
    plt.plot(x_epoch, y_valid_bleu, label="Valid Bleu")

    if fairser_log_entity.test_info_dict:
        y_test_bleu = [fairser_log_entity.test_info_dict[epoch].bleu for epoch in x_epoch]
        plt.plot(x_epoch, y_test_bleu, label="Test Bleu")

    plt.xlabel("Epochs")
    plt.ylabel("Bleu Score")
    plt.title("Bleu")
    plt.legend()
    plt.grid(True)

    # 第二个子图，放loss曲线
    plt.subplot(1, 2, 2)
    x_updates = sorted(fairser_log_entity.train_info_dict.keys())
    y_total_loss = [fairser_log_entity.train_info_dict[update].loss for update in x_updates]
    plt.plot(x_updates, y_total_loss, label="Total Loss")
    y_nll_loss = [fairser_log_entity.train_info_dict[update].nll_loss for update in x_updates]
    plt.plot(x_updates, y_nll_loss, label="NLL Loss")

    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    # 自动调整子图布局，避免重叠
    plt.tight_layout()
    # 保存图片
    png_save_dir = f"{log_dir_path}/log_{int(time.time())}.png"
    print(png_save_dir)
    plt.savefig(png_save_dir)


def parse_multi_log(log_dir_path_list_str:str):
    # 生成日志实体
    fairser_log_entity_dict = dict()
    log_dir_path_list = log_dir_path_list_str.split(',')
    for path in log_dir_path_list:
        fairser_log_entity_dict[path] = FairseqLogEntity.get_instance_by_folder(path)

    # 创建图片
    plt.figure(figsize=(15, 6))

    # 第一个子图，放Bleu Score曲线
    plt.subplot(1, 2, 1)
    for path, fairser_log_entity in fairser_log_entity_dict.items():
        x_epoch = sorted(fairser_log_entity.valid_info_dict.keys())
        y_valid_bleu = [fairser_log_entity.valid_info_dict[epoch].bleu for epoch in x_epoch]
        plt.plot(x_epoch, y_valid_bleu, label=f"Valid Bleu: {path.split('/')[-1]}")
        plt.ylim(31, 35)
        #plt.xlim(0, 300)

    plt.xlabel("Epochs")
    plt.ylabel("Bleu Score")
    plt.title("Bleu")
    plt.legend()
    plt.grid(True)

    # 第二个子图，放loss曲线
    plt.subplot(1, 2, 2)
    for path, fairser_log_entity in fairser_log_entity_dict.items():
        x_updates = sorted(fairser_log_entity.train_info_dict.keys())
        y_total_loss = [fairser_log_entity.train_info_dict[update].loss for update in x_updates]
        plt.plot(x_updates, y_total_loss, label=f"Loss: {path.split('/')[-1]}")

    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    # 自动调整子图布局，避免重叠
    plt.tight_layout()
    # 保存图片
    png_save_dir = f"log_{int(time.time())}.png"
    print(png_save_dir)
    plt.savefig(png_save_dir)


def main(args):
    if args.model_dir:
        parse_single_log(args.model_dir)
    if args.model_dir_list:
        parse_multi_log(args.model_dir_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="", type=str)
    parser.add_argument("--model-dir-list", default="", type=str)

    main(parser.parse_args())

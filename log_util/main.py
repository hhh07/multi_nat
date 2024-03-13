import argparse
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List

from entity.FairseqLogEntity import FairseqLogEntity


def _lighten_color(color, amount=0.25):
    """
    将颜色变浅
    :param color: 要变浅的颜色
    :param amount: 变浅的程度，取值范围为 0~1，0 表示不变，1 表示完全变白，默认为 0.25
    :return: 变浅后的颜色
    """
    rgb = mcolors.colorConverter.to_rgb(color)
    new_rgb = [min(1, c + amount) for c in rgb]
    new_color = mcolors.rgb2hex(new_rgb)
    return new_color


def parse_single_log(log_dir_path: str, args):
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


def parse_multi_log(log_dir_path_list_str:str, args):
    # 生成日志实体
    fairser_log_entity_dict = dict()
    log_dir_path_list = log_dir_path_list_str.split(',')
    for path in log_dir_path_list:
        fairser_log_entity_dict[path] = FairseqLogEntity.get_instance_by_folder(path)

    # 创建图片
    plt.figure(figsize=(15, 6))

    # 第一个子图，放Bleu Score曲线
    plt.subplot(1, 2, 1)
    
    first_dir_max_x = -1
    for path, fairser_log_entity in fairser_log_entity_dict.items():
        x_epoch = sorted(fairser_log_entity.valid_info_dict.keys())
        y_valid_bleu = [fairser_log_entity.valid_info_dict[epoch].bleu for epoch in x_epoch]
        plt.plot(x_epoch, y_valid_bleu, label=f"Valid Bleu: {path.split('/')[-1]}")

        if args.show_extra_info:
            max_y = max(y_valid_bleu)
            max_y_color = plt.gca().lines[-1].get_color()
            plt.axhline(max_y, color=max_y_color, linestyle='dashed')
            plt.text(-0.1, max_y, str(max_y), color=max_y_color, ha='right', va='center')

            first_dir_color = plt.gca().lines[0].get_color()
            if first_dir_max_x == -1:
                first_dir_max_x = max(x_epoch)
                plt.axvline(first_dir_max_x, color=_lighten_color(first_dir_color), linestyle='dashed')
            else:
                max_y_in_first_dir_epoch = max([fairser_log_entity.valid_info_dict[epoch].bleu for epoch in x_epoch if epoch <= first_dir_max_x])
                plt.axhline(max_y_in_first_dir_epoch, color=_lighten_color(max_y_color), linestyle='dashed')
                plt.text(-0.1, max_y_in_first_dir_epoch, str(max_y_in_first_dir_epoch), color=_lighten_color(max_y_color), ha='right', va='center')


        if args.y_min >= 0:
            plt.ylim(args.y_min, args.y_max)
        if args.x_min >= 0:
            plt.xlim(args.x_min, args.x_max)

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
    png_save_dir = f"log.png"
    print(png_save_dir)
    plt.savefig(png_save_dir)


def main(args):
    if args.model_dir:
        parse_single_log(args.model_dir, args)
    if args.model_dir_list:
        parse_multi_log(args.model_dir_list, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="", type=str)
    parser.add_argument("--model-dir-list", default="", type=str)
    parser.add_argument("--x-min", default=-1, type=float)
    parser.add_argument("--x-max", default=300, type=float)
    parser.add_argument("--y-min", default=-1, type=float)
    parser.add_argument("--y-max", default=37, type=float)
    parser.add_argument("--show-extra-info", action="store_true")

    main(parser.parse_args())

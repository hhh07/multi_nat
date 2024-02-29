import os

import re
from typing import List
from typing import Optional
from .TrainInfoEntity import TrainInfoEntity
from .ValidInfoEntity import ValidInfoEntity


class FairseqLogEntity(object):
    """
    Fairseq Log实体类
    """

    TRAIN_LOG_FILE_NAME = "train.log"
    RESULT_GEN_FILE_NAME = "result.gen"

    def __init__(self):
        self.dir_path: str = ""
        ### train log
        # 训练信息，key是num_updates
        self.train_info_dict: dict[int, TrainInfoEntity] = dict()
        # valid集验证信息，key是epoch
        self.valid_info_dict: dict[int, ValidInfoEntity] = dict()
        # test集验证信息，key是epoch
        self.test_info_dict: dict[int, ValidInfoEntity] = dict()

        ### result gen
        # 测试集Bleu Score
        self.test_bleu4: float = 0.0

        # 最高valid_bleu
        self.best_bleu_valid: Optional[ValidInfoEntity] = None

        pass

    @classmethod
    def get_instance_by_folder(cls, dir_path: str) -> 'FairseqLogEntity':
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"无此文件夹: {dir_path}")
        train_log_path = os.path.join(dir_path, FairseqLogEntity.TRAIN_LOG_FILE_NAME)
        result_gen_path = os.path.join(dir_path, FairseqLogEntity.RESULT_GEN_FILE_NAME)

        fairseq_log_entity = FairseqLogEntity()
        fairseq_log_entity.dir_path = os.path.abspath(dir_path)

        ### train log
        train_log_lines = FairseqLogEntity._get_file_lines(train_log_path)
        for line in train_log_lines:
            if "| train |" in line:
                train_info_entity = TrainInfoEntity.get_instance_by_line(line.strip())
                fairseq_log_entity.train_info_dict[train_info_entity.num_updates] = train_info_entity
            if "valid on 'valid' subset" in line:
                valid_info_entity = ValidInfoEntity.get_instance_by_line(line.strip())
                fairseq_log_entity.valid_info_dict[valid_info_entity.epoch] = valid_info_entity
            if "valid on 'test' subset" in line:
                test_info_entity = ValidInfoEntity.get_instance_by_line(line.strip())
                fairseq_log_entity.test_info_dict[test_info_entity.epoch] = test_info_entity

        ### result gen
        result_gen_lines = FairseqLogEntity._get_file_lines(result_gen_path)
        for line in result_gen_lines:
            matches = re.findall(r"BLEU4\s*=\s*(\d+(\.\d+)?)", line)
            if matches:
                fairseq_log_entity.test_bleu4 = float(matches[0][0])

        fairseq_log_entity.best_bleu_valid = max(fairseq_log_entity.valid_info_dict.values(), key=lambda x: x.best_bleu)

        return fairseq_log_entity

    @staticmethod
    def _get_file_lines(file_path: str) -> List[str]:
        train_log_lines = list()
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                train_log_lines = f.readlines()
        return train_log_lines

    def __repr__(self):
        train_info_lines = "\n".join([f"\t\tnum_updates {k}: {v}" for k, v in self.train_info_dict.items()])
        if train_info_lines:
            train_info_lines = f"\n{train_info_lines}\n\t"
        valid_info_lines = "\n".join([f"\t\tepoch {k}: {v}" for k, v in self.valid_info_dict.items()])
        if valid_info_lines:
            valid_info_lines = f"\n{valid_info_lines}\n\t"
        test_info_lines = "\n".join([f"\t\tepoch {k}: {v}" for k, v in self.test_info_dict.items()])
        if test_info_lines:
            test_info_lines = f"\n{test_info_lines}\n\t"

        return f"FairseqLogEntity(\n\t" \
               f"dir_path={self.dir_path},\n\t" \
               f"train_info_dict=[{train_info_lines}],\n\t" \
               f"valid_info_dict=[{valid_info_lines}],\n\t" \
               f"test_info_dict=[{test_info_lines}],\n\t" \
               f"test_bleu4={self.test_bleu4},\n\t" \
               f"best_bleu_valid={self.best_bleu_valid}" \
               f"\n)"

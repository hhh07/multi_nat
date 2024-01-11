import re
from datetime import datetime


class ValidInfoEntity(object):
    def __init__(self):
        # 时间
        self.time: datetime = datetime.now()
        # epoch
        self.epoch: int = 0
        # 测试集名
        self.subset: str = ""
        # loss
        self.loss: float = 0.0
        # nll_loss
        self.nll_loss: float = 0.0
        # bleu
        self.bleu: float = 0.0
        # 更新次数
        self.num_updates: int = 0
        # 此前最好的bleu
        self.best_bleu: float = 0.0

    @classmethod
    def get_instance_by_line(cls, line: str) -> 'ValidInfoEntity':
        valid_info_entity = ValidInfoEntity()

        text_part = line.split("|")
        valid_info_entity.time = datetime.strptime(text_part[0].strip(), "%Y-%m-%d %H:%M:%S")

        matches = re.findall(r"epoch (\d+)", line)
        if matches:
            valid_info_entity.epoch = int(matches[0])

        matches = re.findall(r"valid on '(.*?)' subset", line)
        if matches:
            valid_info_entity.subset = matches[0]

        matches = re.findall(r"\bloss (\d+(\.\d+)?)", line)
        if matches:
            valid_info_entity.loss = float(matches[0][0])

        matches = re.findall(r"nll_loss (\d+(\.\d+)?)", line)
        if matches:
            valid_info_entity.nll_loss = float(matches[0][0])

        matches = re.findall(r"bleu (\d+(\.\d+)?)", line)
        if matches:
            valid_info_entity.bleu = float(matches[0][0])

        matches = re.findall(r"num_updates (\d+)", line)
        if matches:
            valid_info_entity.num_updates = int(matches[0])

        matches = re.findall(r"best_bleu (\d+(\.\d+)?)", line)
        if matches:
            valid_info_entity.best_bleu = float(matches[0][0])

        return valid_info_entity

    def __repr__(self):
        return f"ValidInfoEntity(" \
               f"time={self.time.strftime('%Y-%m-%d %H:%M:%S')}, " \
               f"epoch={self.epoch}, " \
               f"subset={self.subset}, " \
               f"loss={self.loss}, " \
               f"nll_loss={self.nll_loss}, " \
               f"bleu={self.bleu}, " \
               f"num_updates={self.num_updates}, " \
               f"best_bleu={self.best_bleu}" \
               f")"

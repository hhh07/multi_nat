import re
from datetime import datetime


class TrainInfoEntity(object):
    def __init__(self):
        # 时间
        self.time: datetime = datetime.now()
        # epoch
        self.epoch: int = 0
        # loss
        self.loss: float = 0.0
        # nll_loss
        self.nll_loss: float = 0.0
        # 更新次数
        self.num_updates: int = 0
        # 学习率
        self.lr: float = 0.0
        # loss scale
        self.loss_scale: int = 0

    @classmethod
    def get_instance_by_line(cls, line: str) -> 'TrainInfoEntity':
        train_info_entity = TrainInfoEntity()

        text_part = line.split("|")
        train_info_entity.time = datetime.strptime(text_part[0].strip(), "%Y-%m-%d %H:%M:%S")

        matches = re.findall(r"epoch (\d+)", line)
        if matches:
            train_info_entity.epoch = int(matches[0])

        matches = re.findall(r"\bloss (\d+(\.\d+)?)", line)
        if matches:
            train_info_entity.loss = float(matches[0][0])

        matches = re.findall(r"nll_loss (\d+(\.\d+)?)", line)
        if matches:
            train_info_entity.nll_loss = float(matches[0][0])

        matches = re.findall(r"num_updates (\d+)", line)
        if matches:
            train_info_entity.num_updates = int(matches[0])

        matches = re.findall(r"lr (\d+(\.\d+)?(?:[eE][-+]?\d+)?)", line)
        if matches:
            train_info_entity.lr = float(matches[0][0])

        matches = re.findall(r"loss_scale (\d+)", line)
        if matches:
            train_info_entity.loss_scale = int(matches[0])

        return train_info_entity

    def __repr__(self):
        return f"TrainInfoEntity(" \
               f"time={self.time.strftime('%Y-%m-%d %H:%M:%S')}, " \
               f"epoch={self.epoch}, " \
               f"loss={self.loss}, " \
               f"nll_loss={self.nll_loss}, " \
               f"num_updates={self.num_updates}, " \
               f"lr={self.lr}, " \
               f"loss_scale={self.loss_scale}" \
               f")"

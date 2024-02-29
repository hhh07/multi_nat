# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
from fairseq.data.raw_label_dataset import RawLabelDataset
from typing import Optional


logger = logging.getLogger(__name__)

SIGMA = 1 #0.75
VAR_TIMES_2 = torch.tensor(2 * SIGMA ** 2)

def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_dep=-1,
):
    if len(samples) == 0:
        return {}

    def check_dependency(dependency, seq_len):
        if dependency is None or len(dependency) == 0:
            return False
        if dependency.max().item() >= seq_len:
            logger.warning("dependency size mismatch found, skipping dependency!")
            return False
        return True

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    prev_tgt_pos_tokens = None
    prev_tgt_dphead_tokens = None
    prev_tgt_dplable_tokens = None
    target = None
    tgt_pos = None
    tgt_dphead = None
    tgt_dplable = None

    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()
    
    if samples[0].get("tgt_pos", None) is not None:
        #长度和tgt一样，用一样的pad_to_length
        tgt_pos = merge(
            "tgt_pos",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        tgt_pos = tgt_pos.index_select(0, sort_order)

        if samples[0].get("prev_tgt_pos_tokens", None) is not None:
            prev_tgt_pos_tokens = merge("prev_tgt_pos_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_tgt_pos_tokens = merge(
                "tgt_pos",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    if samples[0].get("tgt_dphead", None) is not None:
        #长度和tgt一样，用一样的pad_to_length
        tgt_dphead = merge(
            "tgt_dphead",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        tgt_dphead = tgt_dphead.index_select(0, sort_order)

        if samples[0].get("prev_tgt_dphead_tokens", None) is not None:
            prev_tgt_dphead_tokens = merge("prev_tgt_dphead_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_tgt_dphead_tokens = merge(
                "tgt_dphead",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    if samples[0].get("tgt_dplable", None) is not None:
        #长度和tgt一样，用一样的pad_to_length
        tgt_dplable = merge(
            "tgt_dplable",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        tgt_dplable = tgt_dplable.index_select(0, sort_order)

        if samples[0].get("prev_tgt_dplable_tokens", None) is not None:
            prev_tgt_dplable_tokens = merge("prev_tgt_dplable_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_tgt_dplable_tokens = merge(
                "tgt_dplable",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths,},
        "target": target,
        "tgt_pos": tgt_pos,
        "tgt_dphead": tgt_dphead,
        "tgt_dplable": tgt_dplable, 

    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )
    if prev_tgt_pos_tokens is not None:
        batch["net_input"]["prev_tgt_pos_tokens"] = prev_tgt_pos_tokens.index_select(
            0, sort_order
        )
    if prev_tgt_dphead_tokens is not None:
        batch["net_input"]["prev_tgt_dphead_tokens"] = prev_tgt_dphead_tokens.index_select(
            0, sort_order
        )
    if prev_tgt_dplable_tokens is not None:
        batch["net_input"]["prev_tgt_dplable_tokens"] = prev_tgt_dplable_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints.index_select(0, sort_order)

    def _get_batch_dep(batch_samples, batch_tokens, lengths, batch_sort_order, dep_name: str,
                       left_pad: bool, pad_dep=-1):
        batch_size, seq_size = batch_tokens.shape

        batch_dependency = list()
        for dep_idx, snt_len in zip(batch_sort_order, lengths):
            for dependency in [batch_samples[dep_idx][dep_name]]:
                if check_dependency(dependency, snt_len):
                    if seq_size - snt_len > 0:
                        pad_len = seq_size - snt_len
                        padding_dependency = torch.full((pad_len, ), pad_dep)
                        if left_pad:
                            dependency += pad_len
                            dependency = torch.cat([padding_dependency, dependency], dim=0)
                        else:
                            dependency = torch.cat([dependency, padding_dependency], dim=0)
                    batch_dependency.append(dependency)

        return torch.stack(batch_dependency, dim=0) if len(batch_dependency) > 0 else None
    #sman_mask
    def _calc_batch_sman_mask(batch_dep):
        batch_size, seq_size = batch_dep.shape
        #创建mask矩阵
        indices = torch.arange(seq_size).unsqueeze(1)  # 形状为 (D, 1)
        # 计算绝对值差值矩阵
        abs_diff = torch.abs(indices - indices.t())  # 形状为 (D, D)
        
        # 利用逻辑运算设置满足条件的元素为1
        mask = (abs_diff <= 4).float()
        

        return mask
    
    
    #根据距离生成dist，生成dist包括了掩码-1的部分，感觉不对劲
    #待修改：应该先生成dist，再加上掩码的部分，掩码部分不参加dist的计算?不然算的不是dist。但是还有问题，
    def _calc_batch_dep_dist(batch_dep):
        batch_size, seq_size = batch_dep.shape

        mask_condition = batch_dep < 0
        #创建mask矩阵
        dep_dist_mask = (~(
                mask_condition.unsqueeze(2).repeat(1, 1, seq_size) |
                mask_condition.unsqueeze(1).repeat(1, seq_size, 1)
        )).float()
        #计算dist
        minus_square = (torch.arange(seq_size) - batch_dep.unsqueeze(-1)) ** 2
        denominator = torch.sqrt(torch.pi * VAR_TIMES_2)
        exp_part = torch.exp(-minus_square / VAR_TIMES_2)
        dep_dist = exp_part / denominator
        dep_dist *= dep_dist_mask

        return dep_dist

    #binary的dist矩阵
    def _calc_batch_dep_dist_binary(batch_dep):
        batch_size, s = batch_dep.shape

        #去掉batch_dep中每个句子的bos和eos
        batch_dep = batch_dep[:, 1:-1]
        s=s-2

        # 创建一个 (batch_size, s, s) 的零张量
        dep_dist = torch.zeros(batch_size, s, s, dtype=torch.float32, device=batch_dep.device).bool()
        # 使用广播和逻辑运算来实现目标张量的生成
        mask = torch.arange(s, dtype=torch.long, device=batch_dep.device).unsqueeze(0).repeat(batch_size, 1)
    
        dep_dist |= (mask.unsqueeze(2) == batch_dep.unsqueeze(1))
        dep_dist |= (mask.unsqueeze(1) == batch_dep.unsqueeze(2))
        # 将对角线元素设置为 1
        dep_dist |= torch.eye(s, device=batch_dep.device).unsqueeze(0).bool()

        #给dep_dist矩阵加上eos和bos
        # 创建一个形状为 (batch_size, s+2, s+2) 的零张量
        padded_dep_dist = torch.zeros(batch_size, s+2, s+2)

        # 将 dep_dist 的值复制到 padded_dep_dist 的中心区域
        padded_dep_dist[:, 1:-1, 1:-1] = dep_dist.float()
        
        return padded_dep_dist

    if samples[0].get("src_dep", None) is not None:
        batch_dep = _get_batch_dep(samples, batch["net_input"]["src_tokens"], src_lengths, sort_order, "src_dep",
                                   left_pad_source, pad_dep)
        #batch_lable
        #batch_lable = 
        if batch_dep is not None:
            batch["net_input"]["src_dep"] = batch_dep
            batch["net_input"]["src_dep_dist"] = _calc_batch_dep_dist(batch_dep)

    if samples[0].get("tgt_dep", None) is not None:
        batch_dep = _get_batch_dep(samples, batch["target"], tgt_lengths, sort_order, "tgt_dep",
                                   left_pad_target, pad_dep)
        if batch_dep is not None:
            batch["tgt_dep"] = batch_dep
            # batch["tgt_dep_dist"] = _calc_batch_dep_dist(batch_dep)


    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,src_sizes,src_dict,
        tgt=None,tgt_sizes=None,tgt_dict=None,
        tgt_pos=None,tgt_pos_sizes=None,tgt_pos_dict=None, #hzj
        tgt_dphead=None,tgt_dphead_sizes=None,tgt_dphead_dict=None,
        tgt_dplable=None,tgt_dplable_sizes=None,tgt_dplable_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        src_dep: Optional[RawLabelDataset] = None,
        tgt_dep: Optional[RawLabelDataset] = None,
    ):
        if tgt_dict is not None:
            #其他要加吗？
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        #hzj
        self.tgt_pos = tgt_pos
        self.tgt_dphead = tgt_dphead
        self.tgt_dplable = tgt_dplable
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        #hzj
        self.tgt_pos_sizes = np.array(tgt_pos_sizes)
        self.tgt_dphead_sizes = np.array(tgt_dphead_sizes)
        self.tgt_dplable_sizes = np.array(tgt_dplable_sizes)

        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        #hzj
        self.tgt_pos_dict = tgt_pos_dict
        self.tgt_dphead_dict = tgt_dphead_dict
        self.tgt_dplable_dict = tgt_dplable_dict

        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

        # yzh dep
        self.src_dep = src_dep
        self.tgt_dep = tgt_dep
        self.pad_dep = -1
        self.eos_dep = -1

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        #hzj
        tgt_pos_item = self.tgt_pos[index] if self.tgt_pos is not None else None
        tgt_dphead_item = self.tgt_dphead[index] if self.tgt_dphead is not None else None
        tgt_dplable_item = self.tgt_dplable[index] if self.tgt_dplable is not None else None
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
                tgt_pos_item = self.tgt_pos[index][:-1]
                tgt_dphead_item = self.tgt_dphead[index][:-1]
                tgt_dplable_item = self.tgt_dplable[index][:-1]

        #hzj
        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "tgt_pos": tgt_pos_item,
            "tgt_dphead": tgt_dphead_item,
            "tgt_dplable": tgt_dplable_item,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]

        # yzh dep
        def _get_dep_item(snt_dep, using_eos: bool):
            if using_eos:
                eos_dep = torch.tensor([self.eos_dep], dtype=snt_dep.dtype)
                return torch.cat([snt_dep, eos_dep], dim=0)
            else:
                return snt_dep

        if self.src_dep is not None:
            example["src_dep"] = _get_dep_item(
                self.src_dep[index],
                not self.remove_eos_from_source or (
                        len(src_item) - len(self.src_dep[index]) == 1 and src_item[-1] == self.eos
                )
            )
        if self.tgt_dep is not None:
            example["tgt_dep"] = _get_dep_item(
                self.tgt_dep[index],
                self.append_eos_to_target or (
                        len(tgt_item) - len(self.tgt_dep[index]) == 1 and tgt_item[-1] == self.eos
                )
            )

        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
            pad_dep=self.pad_dep,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
            self.tgt_pos_sizes[index] if self.tgt_pos_sizes is not None else 0,
            self.tgt_dphead_sizes[index] if self.tgt_dphead_sizes is not None else 0,
            self.tgt_dplable_sizes[index] if self.tgt_dplable_sizes is not None else 0
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        ) and (
            getattr(self.tgt_pos, "supports_prefetch", False) or self.tgt_pos is None
        ) and (
            getattr(self.tgt_dphead, "supports_prefetch", False) or self.tgt_dphead is None
        ) and (
            getattr(self.tgt_dplable, "supports_prefetch", False) or self.tgt_dplable is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
        if self.tgt_pos is not None:
            self.tgt_pos.prefetch(indices)
        if self.tgt_dphead is not None:
            self.tgt_dphead.prefetch(indices)
        if self.tgt_dplable is not None:
            self.tgt_dplable.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes, self.tgt_sizes, indices, max_sizes,
        )

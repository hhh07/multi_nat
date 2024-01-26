# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy
import logging

import torch

from dataclasses import dataclass, field
from fairseq.tasks import register_task
from fairseq import utils
from multitasknat.tasks.nat_ctc_task import NATCTCConfig, NATCTC_Task

#hzj
import itertools
from typing import Optional
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)

EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)

#hzj
def load_langpair_dataset(
    data_path,
    tags_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    tgt_pos,
    tgt_pos_dict,
    tgt_dphead,
    tgt_dphead_dict,
    tgt_dplable,
    tgt_dplable_dict,                                
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    tgt_pos_datasets = []
    tgt_dphead_datasets = []
    tgt_dplable_datasets = []


    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break
    #上面都一样
    #hzj
    for k in itertools.count():
        #hzj
        #只有trian数据集需要dp信息，但是还是读取一下valid的
        
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, tgt_dphead, None, tgt_dphead, tags_path):
            prefix_pos = os.path.join(tags_path, "{}.{}-{}.".format(split_k, tgt_pos, None))
            prefix_dphead = os.path.join(tags_path, "{}.{}-{}.".format(split_k, tgt_dphead, None))
            prefix_dplable = os.path.join(tags_path, "{}.{}-{}.".format(split_k, tgt_dplable, None))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        # 这个地方的prefix + 什么？？

        tgt_pos_dataset = data_utils.load_indexed_dataset(
            prefix_pos + tgt_pos, tgt_pos_dict, dataset_impl
        )
        if tgt_pos_dataset is not None:
            tgt_pos_datasets.append(tgt_pos_dataset)

        tgt_dphead_dataset = data_utils.load_indexed_dataset(
            prefix_dphead + tgt_dphead, tgt_dphead_dict, dataset_impl
        )
        if tgt_dphead_dataset is not None:
            tgt_dphead_datasets.append(tgt_dphead_dataset)

        tgt_dplable_dataset = data_utils.load_indexed_dataset(
            prefix_dplable + tgt_dplable, tgt_dplable_dict, dataset_impl
        )
        if tgt_dplable_dataset is not None:
            tgt_dplable_datasets.append(tgt_dplable_dataset)

        if not combine:
            break


    assert len(src_datasets) == len(tgt_datasets) == len(tgt_pos_datasets) == len(tgt_dphead_datasets) == len(tgt_dplable_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        tgt_pos_dataset = tgt_pos_datasets[0]
        tgt_dphead_dataset = tgt_dphead_datasets[0]
        tgt_dplable_dataset = tgt_dplable_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None
        tgt_pos_dataset = ConcatDataset(tgt_pos_datasets, sample_ratios)
        tgt_dphead_dataset = ConcatDataset(tgt_dphead_datasets, sample_ratios)
        tgt_dplable_dataset = ConcatDataset(tgt_dplable_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

        # tgt_pos_dataset = tgt_pos_datasets[0]
        # tgt_dphead_dataset = tgt_dphead_datasets[0]
        # tgt_dplable_dataset = tgt_dplable_dataset[0]
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,src_dataset.sizes,src_dict,
        tgt_dataset,tgt_dataset_sizes,tgt_dict,
        tgt_pos_dataset,tgt_pos_dataset.sizes,tgt_pos_dict,
        tgt_dphead_dataset,tgt_dphead_dataset.sizes,tgt_dphead_dict,
        tgt_dplable_dataset,tgt_dplable_dataset.sizes,tgt_dplable_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )



def drop_sentences_(sample, rate, indexes=None):
    def gen_randperm(bsz, droprate):
        nbsz = max(1, int((1.0 - droprate) * bsz))
        return torch.randperm(bsz)[:nbsz]

    bsz = sample['nsentences']
    if indexes is None:
        indexes = gen_randperm(bsz, rate)
    nbsz = indexes.size(0)
    for k, v in sample['net_input'].items():
        if isinstance(v, torch.Tensor):
            sample['net_input'][k] = v[indexes]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v[indexes]
    sample['ntokens'] = sample['ntokens'] * nbsz // bsz
    sample['nsentences'] = nbsz
    return sample


def normal(src_tokens, scale, src_dict):
    pad = src_dict.pad()
    bos = src_dict.bos()
    eos = src_dict.eos()
    unk = src_dict.unk()

    _mask = (
            src_tokens.eq(bos) | src_tokens.eq(eos) | src_tokens.eq(pad)
    )
    src_tokens = src_tokens.masked_fill(~_mask, unk)
    bsz = src_tokens.size(0)
    upsample_src_tokens = src_tokens.unsqueeze(-1).expand(bsz, -1, scale).reshape(bsz, -1)
    return upsample_src_tokens


@dataclass
class DPCTCConfig(NATCTCConfig):
    if_deepcopy_at_sample: bool = field(
        default=False, metadata={"help": "if set, shuffle at sample."}
    )
    start_p: float = field(
        default=0.5, metadata={"help": "minus prob"}
    )
    minus_p: float = field(
        default=0.2, metadata={"help": "minus prob"}
    )
    total_up: int = field(
        default=300000, metadata={"help": "total updates"}
    )
    glat: bool = field(
        default=False,
    )
    tgt_pos: Optional[str] = field(
        default="pos",
        metadata={
            "help": "tgt_pos",
        },
    )
    tgt_dphead: Optional[str] = field(
        default="dphead",
        metadata={
            "help": "tgt_dphead",
        },
    )
    tgt_dplable: Optional[str] = field(
        default="dplable",
        metadata={
            "help": "tgt_dplable",
        },
    )
    tgt_text: Optional[str] = field(
        default="text",
        metadata={
            "help": "tgt_text",
        },
    )
    tags_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "tags_path",
        },
    )



@register_task('dp_ctc_task', dataclass=DPCTCConfig)
class DP_CTC_Task(NATCTC_Task):

    #hzj
    def __init__(self, cfg, src_dict, tgt_dict, tgt_pos_dict, tgt_dphead_dict, tgt_dplable_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.tgt_pos_dict = tgt_pos_dict
        self.tgt_dphead_dict = tgt_dphead_dict
        self.tgt_dplable_dict = tgt_dplable_dict
        

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        #text = self.cfg.tgt_text
        pos = self.cfg.tgt_pos
        dphead = self.cfg.tgt_dphead
        dplable = self.cfg.tgt_dplable
        tags_path = self.cfg.tags_path


        self.datasets[split] = load_langpair_dataset(
            data_path,
            tags_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            pos, self.tgt_pos_dict,
            dphead, self.tgt_dphead_dict,
            dplable, self.tgt_dplable_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        model_args = kwargs['model']
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(args, model_args, os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(args, model_args, os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        #hzj add dp
        if args.tags_path is None:
            raise Exception('Path(s) to tags data directorie(s) missing')

        tgt_pos_dict = cls.load_dictionary(args, model_args, os.path.join(args.tags_path, 'dict.{}.txt'.format(args.tgt_pos)))
        tgt_dphead_dict = cls.load_dictionary(args, model_args, os.path.join(args.tags_path, 'dict.{}.txt'.format(args.tgt_dphead)))
        tgt_dplable_dict = cls.load_dictionary(args, model_args, os.path.join(args.tags_path, 'dict.{}.txt'.format(args.tgt_dplable)))
        print('| [{}] tgt_pos dictionary: {} types'.format(args.tgt_pos, len(tgt_pos_dict)))
        print('| [{}] tgt_dphead dictionary: {} types'.format(args.tgt_dphead, len(tgt_dphead_dict)))
        print('| [{}] tgt_dplable dictionary: {} types'.format(args.tgt_dplable, len(tgt_dplable_dict)))

        return cls(args, src_dict, tgt_dict, tgt_pos_dict, tgt_dphead_dict, tgt_dplable_dict)


    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False, **kwargs):
        model.train()
        glat = None
        if getattr(self.cfg, "glat", False):
            train_ratio = max(0, min(1, update_num / self.cfg.total_up))
            glat = {"context_p": self.cfg.start_p - self.cfg.minus_p * train_ratio}
        at_sample = sample
        if getattr(self.cfg, "if_deepcopy_at_sample", False):
            at_sample = deepcopy(sample)
            at_sample = drop_sentences_(at_sample, rate=0.0)
        nat_sample = sample
        src_tokens = sample["net_input"]["src_tokens"].clone()
        upsample_src_tokens = normal(src_tokens, self.cfg.upsample_scale, self.src_dict)
        nat_sample['prev_target'] = upsample_src_tokens
        loss, sample_size, logging_output = criterion(model, at_sample, nat_sample, None, glat=glat, **kwargs)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, **kwargs):
        model.eval()
        with torch.no_grad():
            at_sample = deepcopy(sample)
            nat_sample = sample
            src_tokens = sample["net_input"]["src_tokens"].clone()
            upsample_src_tokens = normal(src_tokens, self.cfg.upsample_scale, self.src_dict)
            nat_sample['prev_target'] = upsample_src_tokens
            loss, sample_size, logging_output = criterion(model, at_sample, nat_sample)
            # 以下为后来添加
            if self.cfg.eval_bleu:
                bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
                logging_output["_bleu_sys_len"] = bleu.sys_len
                logging_output["_bleu_ref_len"] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            extra_symbols_to_ignore = []
            if hasattr(self.tgt_dict, "blank_index"): extra_symbols_to_ignore.append(self.tgt_dict.blank_index)
            if hasattr(self.tgt_dict, "mask_index"): extra_symbols_to_ignore.append(self.tgt_dict.mask_index)
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
                extra_symbols_to_ignore=extra_symbols_to_ignore or None
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = gen_out[i][0]['tokens']
            if not self.cfg.use_ctc_bs:
                _toks = hyp.int().tolist()
                hyp = hyp.new_tensor([v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]])
            hyps.append(decode(hyp))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    @property
    def tgt_pos_dictionary(self):
        return self.tgt_pos_dict

    @property
    def tgt_dphead_dictionary(self):
        return self.tgt_dphead_dict

    @property
    def tgt_dplable_dictionary(self):
        return self.tgt_dplable_dict
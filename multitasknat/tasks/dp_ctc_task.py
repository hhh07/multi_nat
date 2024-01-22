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
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
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
    pos_datasets = []
    dphead_datasets = []


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

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

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

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
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
class MTCTCConfig(NATCTCConfig):
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


@register_task('dp_ctc_task', dataclass=MTCTCConfig)
class MT_CTC_Task(NATCTC_Task):

    #hzj
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

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
        )

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

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor


@dataclass
class MyCtcCriterionConfig(FairseqDataclass):
    upsample_scale: int = field(
        default=3,
        metadata={"help": "the amount of src_tokens upsample scale."}
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "epsilon for label smoothing, 0 means no label smoothing",
        },
    )
    is_ds_loss: bool = field(
        default=False,
        metadata={
            "help": "calculate ds loss",
        },
    )


@register_criterion("upsample_ctc_loss", dataclass=MyCtcCriterionConfig)
class UpsampleCTCLoss(FairseqCriterion):
    def __init__(self, cfg, task):
        super().__init__(task)
        self.upsample_scale = cfg.upsample_scale
        self.label_smoothing = cfg.label_smoothing
        self.is_ds_loss = cfg.is_ds_loss
        self.blank_idx = task.target_dictionary.blank_index
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.zero_infinity = True

    def forward(self, model, sample, reduce=True):
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )
        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
       
       
        #dslp
        if self.is_ds_loss:
            net_output, output_logits_list = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
            
            input_lengths = src_lengths * self.upsample_scale

            pad_mask = (sample["target"] != self.pad_idx) & (
                    sample["target"] != self.eos_idx
            )
            targets_flat = sample["target"].masked_select(pad_mask)
            if "target_lengths" in sample:
                target_lengths = sample["target_lengths"]
            else:
                target_lengths = pad_mask.sum(-1)


            #本来的loss
            # lprobs = model.get_normalized_probs(
            #     net_output, log_probs=True
            # ).contiguous()  # (T, B, C) from the decoder

            # with torch.backends.cudnn.flags(enabled=False):
            #     nll_loss = F.ctc_loss(
            #         lprobs.float(),  # to fix with fp16
            #         targets_flat,
            #         input_lengths,
            #         target_lengths,
            #         blank=self.blank_idx,
            #         reduction="mean",
            #         zero_infinity=self.zero_infinity,
            #     )
            # if self.label_smoothing > 0:
            #     loss = nll_loss * (1 - self.label_smoothing) - mean_ds(lprobs) * self.label_smoothing
            # else:
            #     loss = nll_loss
            
            
            #dslp loss
            all_layer_ctc_loss = 0
            normalized_factor = 0

            for layer_idx, word_ins_out in enumerate(output_logits_list):
                lprobs_layer = model.get_normalized_probs(
                    word_ins_out, log_probs=True
                ).contiguous()  # (T, B, C) from the decoder

                with torch.backends.cudnn.flags(enabled=False):
                    layer_nll_loss = F.ctc_loss(
                        lprobs_layer.float(),  # to fix with fp16
                        targets_flat,
                        input_lengths,
                        target_lengths,
                        blank=self.blank_idx,
                        reduction="mean",
                        zero_infinity=self.zero_infinity,
                    )

                if self.label_smoothing > 0:
                    layer_loss = layer_nll_loss * (1 - self.label_smoothing) - mean_ds(lprobs) * self.label_smoothing
                else:
                    layer_loss = layer_nll_loss

                factor = 1  # math.sqrt(layer_idx + 1)
                all_layer_ctc_loss += layer_loss * factor
                normalized_factor += factor
            avg_layer_ctc_loss = all_layer_ctc_loss / normalized_factor

            ntokens = (
                sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
            )

            sample_size = 1
            logging_output = {
                "loss": utils.item(loss.data),  # * sample['ntokens'],
                "nll_loss": utils.item(nll_loss.data),
                "ntokens": ntokens,
                "nsentences": sample["id"].numel(),
                "sample_size": sample_size,
            }

            return avg_layer_ctc_loss, sample_size, logging_output
        else:
            #net_output = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
            net_output, output_logits_list = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
            lprobs = model.get_normalized_probs(
                net_output, log_probs=True
            ).contiguous()  # (T, B, C) from the decoder

            input_lengths = src_lengths * self.upsample_scale

            pad_mask = (sample["target"] != self.pad_idx) & (
                    sample["target"] != self.eos_idx
            )
            targets_flat = sample["target"].masked_select(pad_mask)
            if "target_lengths" in sample:
                target_lengths = sample["target_lengths"]
            else:
                target_lengths = pad_mask.sum(-1)

            with torch.backends.cudnn.flags(enabled=False):
                nll_loss = F.ctc_loss(
                    lprobs.float(),  # to fix with fp16
                    targets_flat,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction="mean",
                    zero_infinity=self.zero_infinity,
                )

            if self.label_smoothing > 0:
                loss = nll_loss * (1 - self.label_smoothing) - mean_ds(lprobs) * self.label_smoothing
            else:
                loss = nll_loss

            ntokens = (
                sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
            )

            sample_size = 1
            logging_output = {
                "loss": utils.item(loss.data),  # * sample['ntokens'],
                "nll_loss": utils.item(nll_loss.data),
                "ntokens": ntokens,
                "nsentences": sample["id"].numel(),
                "sample_size": sample_size,
            }

            return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

import torch
import torch.nn.functional as F
import math
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.nat_loss import LabelSmoothedDualImitationCriterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
import numpy as np
import gc


@register_criterion("t1dp_ctc_loss")
class t1dp_ctc_loss(LabelSmoothedDualImitationCriterion):
    def __init__(self, task, lambda_nat_at, label_smoothing, zero_infinity,
                 ):
        super().__init__(task, label_smoothing)
        self.lambda_nat_at = lambda_nat_at
        self.blank_idx = task.target_dictionary.blank_index
        self.upsample_scale = task.cfg.upsample_scale
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.zero_infinity = zero_infinity

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--lambda-nat-at",
            default=0.5,
            type=float,
        )
        parser.add_argument(
            "--label-smoothing",
            default=0.1,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument(
            '--zero-infinity', action='store_false', default=True)

    def forward(self, model, at_sample, nat_sample, reduce=True, **kwargs):
        """
        Compute the loss of Multi-task learning.
        Loss = \lambda Loss_{at} + (1 - \lambda) Loss_{nat}
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = at_sample["nsentences"], at_sample["ntokens"]

        # B x T
        at_src_tokens, src_lengths, nat_src_tokens = (
            at_sample["net_input"]["src_tokens"],
            at_sample["net_input"]["src_lengths"],
            nat_sample["net_input"]["src_tokens"]
        )
        at_tgt_tokens, prev_nat, prev_at = at_sample["target"], \
                                           nat_sample["prev_target"], \
                                           at_sample["net_input"]["prev_output_tokens"]
        #hzj
        prev_pos, prev_dphead, prev_dplable = at_sample["net_input"]["prev_tgt_pos_tokens"], \
                                              at_sample["net_input"]["prev_tgt_dphead_tokens"], \
                                              at_sample["net_input"]["prev_tgt_dplable_tokens"]
        tgt_pos_tokens, tgt_dphead_tokens, tgt_dplable_tokens = at_sample["tgt_pos"], \
                                                                at_sample["tgt_dphead"], \
                                                                at_sample["tgt_dplable"]

        # TODO 根据model的forward函数来决定这里传什么参数

        hybrid_outputs = model(at_src_tokens, nat_src_tokens, src_lengths, prev_nat, prev_at, prev_pos, prev_dphead, prev_dplable, at_tgt_tokens, **kwargs)
        hybrid_loss = {}

        for outputs in hybrid_outputs:
            if outputs['name'] == "NAT":
                net_output = outputs['out']
                lprobs = model.get_normalized_probs(
                    net_output, log_probs=True
                ).contiguous()  # (T, B, C) from the decoder
                input_lengths = nat_sample["net_input"]["src_lengths"] * self.upsample_scale
                nat_target = model.get_targets(nat_sample, net_output, "NAT")
                pad_mask = (nat_target != self.pad_idx) & (
                        nat_target != self.eos_idx
                )
                targets_flat = nat_target.masked_select(pad_mask)
                if "target_lengths" in nat_sample:
                    target_lengths = nat_sample["target_lengths"]
                else:
                    target_lengths = pad_mask.sum(-1)

                with torch.backends.cudnn.flags(enabled=False):
                    loss = F.ctc_loss(
                        lprobs.float(),  # to fix with fp16
                        targets_flat,
                        input_lengths,
                        target_lengths,
                        blank=self.blank_idx,
                        reduction="mean",
                        zero_infinity=self.zero_infinity,
                    )
                hybrid_loss["NAT"] = loss
            elif outputs['name'] == "AT":
                if outputs.get("loss", None) is None:
                    at_net_outputs = outputs['out']
                    at_loss_list, at_nll_loss_list = [], []
                    output_property = outputs.get("property")
                    for i, at_net_output in enumerate(at_net_outputs):
                        at_lprobs = model.get_normalized_probs(at_net_output, log_probs=True)
                        if output_property is not None:
                            at_target = model.get_targets(at_sample, at_net_output, "AT", output_property[i])
                        else:
                            at_target = model.get_targets(at_sample, at_net_output, "AT")

                        at_loss, at_nll_loss = label_smoothed_nll_loss(
                            at_lprobs.view(-1, at_lprobs.size(-1)), at_target.view(-1, 1), self.label_smoothing,
                            ignore_index=self.padding_idx,
                            reduce=reduce,
                        )
                        at_loss, at_nll_loss = at_loss.mean(), at_nll_loss.mean()
                        at_loss_list.append(at_loss)
                        at_nll_loss_list.append(at_nll_loss)
                    hybrid_loss["AT"] = sum(l for l in at_loss_list) / len(at_loss_list)
                else:
                    hybrid_loss["AT"] = outputs["loss"]
                    at_loss_list = outputs['at_loss_list']
            #hzj
            elif outputs['name'] == "POS":
                if outputs.get("loss", None) is None:
                    pos_net_outputs = outputs['out']
                    pos_loss_list, pos_nll_loss_list = [], []
                    output_property = outputs.get("property")
                    for i, pos_net_output in enumerate(pos_net_outputs):
                        pos_lprobs = model.get_normalized_probs(pos_net_output, log_probs=True)
                        if output_property is not None:
                            pos_target = model.get_targets(at_sample, pos_net_output, "POS", output_property[i])
                        else:
                            pos_target = model.get_targets(at_sample, pos_net_output, "POS")

                        pos_loss, pos_nll_loss = label_smoothed_nll_loss(
                            pos_lprobs.view(-1, pos_lprobs.size(-1)), pos_target.view(-1, 1), self.label_smoothing,
                            ignore_index=self.padding_idx,
                            reduce=reduce,
                        )
                        pos_loss, pos_nll_loss = pos_loss.mean(), pos_nll_loss.mean()
                        pos_loss_list.append(pos_loss)
                        pos_nll_loss_list.append(pos_nll_loss)
                    hybrid_loss["POS"] = sum(l for l in pos_loss_list) / len(pos_loss_list)
                else:
                    hybrid_loss["POS"] = outputs["loss"]
                    pos_loss_list = outputs['pos_loss_list']
            elif outputs['name'] == "DPHEAD":
                if outputs.get("loss", None) is None:
                    dphead_net_outputs = outputs['out']
                    dphead_loss_list, dphead_nll_loss_list = [], []
                    output_property = outputs.get("property")
                    for i, dphead_net_output in enumerate(dphead_net_outputs):
                        dphead_lprobs = model.get_normalized_probs(dphead_net_output, log_probs=True)
                        if output_property is not None:
                            dphead_target = model.get_targets(at_sample, dphead_net_output, "DPHEAD", output_property[i])
                        else:
                            dphead_target = model.get_targets(at_sample, dphead_net_output, "DPHEAD")

                        dphead_loss, dphead_nll_loss = label_smoothed_nll_loss(
                            dphead_lprobs.view(-1, dphead_lprobs.size(-1)), dphead_target.view(-1, 1), self.label_smoothing,
                            ignore_index=self.padding_idx,
                            reduce=reduce,
                        )
                        dphead_loss, dphead_nll_loss = dphead_loss.mean(), dphead_nll_loss.mean()
                        dphead_loss_list.append(dphead_loss)
                        dphead_nll_loss_list.append(dphead_nll_loss)
                    hybrid_loss["DPHEAD"] = sum(l for l in dphead_loss_list) / len(dphead_loss_list)
                else:
                    hybrid_loss["DPHEAD"] = outputs["loss"]
                    dphead_loss_list = outputs['dphead_loss_list']
            elif outputs['name'] == "DPLABLE":
                if outputs.get("loss", None) is None:
                    dplable_net_outputs = outputs['out']
                    dplable_loss_list, dplable_nll_loss_list = [], []
                    output_property = outputs.get("property")
                    for i, dplable_net_output in enumerate(dplable_net_outputs):
                        dplable_lprobs = model.get_normalized_probs(dplable_net_output, log_probs=True)
                        if output_property is not None:
                            dplable_target = model.get_targets(at_sample, dplable_net_output, "DPLABLE", output_property[i])
                        else:
                            dplable_target = model.get_targets(at_sample, dplable_net_output, "DPLABLE")

                        dplable_loss, dplable_nll_loss = label_smoothed_nll_loss(
                            dplable_lprobs.view(-1, dplable_lprobs.size(-1)), dplable_target.view(-1, 1), self.label_smoothing,
                            ignore_index=self.padding_idx,
                            reduce=reduce,
                        )
                        dplable_loss, dplable_nll_loss = dplable_loss.mean(), dplable_nll_loss.mean()
                        dplable_loss_list.append(dplable_loss)
                        dplable_nll_loss_list.append(dplable_nll_loss)
                    hybrid_loss["DPLABLE"] = sum(l for l in dplable_loss_list) / len(dplable_loss_list)
                else:
                    hybrid_loss["DPLABLE"] = outputs["loss"]
                    dplable_loss_list = outputs['dplable_loss_list']
            else:
                raise NotImplementedError

        #hzj  计算loss
        #要改
        loss = self.lambda_nat_at * (hybrid_loss["AT"] + hybrid_loss["POS"] + hybrid_loss["DPHEAD"] + hybrid_loss["DPLABLE"] ) / 4 + \
               (1 - self.lambda_nat_at) * hybrid_loss["NAT"]

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        #要改
        logging_output = {
            "loss": loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "nat-ctc-loss": hybrid_loss["NAT"].data,
            "at-average-loss": hybrid_loss["AT"].data,
            "pos-loss": hybrid_loss["POS"].data,
            "dphead-loss": hybrid_loss["DPHEAD"].data,
            "dplable-loss": hybrid_loss["DPLABLE"].data
        }
        if "glat_accu" in hybrid_outputs[0]:
            logging_output["glat_accu"] = hybrid_outputs[0]['glat_accu']
        if "glat_context_p" in hybrid_outputs[0]:
            logging_output['glat_context_p'] = hybrid_outputs[0]['glat_context_p']
        num_at_loss = 1
        for at_loss in at_loss_list:
            logging_output["at-" + str(num_at_loss) + "-loss"] = at_loss.data
            num_at_loss += 1
        num_at_loss = 1
        for pos_loss in pos_loss_list:
            logging_output["pos-" + str(num_at_loss) + "-loss"] = pos_loss.data
            num_at_loss += 1
        num_at_loss = 1
        for dphead_loss in dphead_loss_list:
            logging_output["dphead-" + str(num_at_loss) + "-loss"] = dphead_loss.data
            num_at_loss += 1
        num_at_loss = 1
        for dplable_loss in dplable_loss_list:
            logging_output["dplable-" + str(num_at_loss) + "-loss"] = dplable_loss.data
            num_at_loss += 1
        
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        def log_metric(key, logging_outputs):
            # print(type(logging_outputs[0].get(key, 0)))
            if len(logging_outputs) > 0 and key in logging_outputs[0]:
                metrics.log_scalar(
                    key,
                    utils.item(np.mean([log.get(key, 0).cpu() for log in logging_outputs])) / sample_size
                    if sample_size > 0 else 0.0,
                    sample_size,
                    round=3
                )

        log_metric("glat_accu", logging_outputs)
        log_metric("glat_context_p", logging_outputs)
        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

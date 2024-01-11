import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerModel
from fairseq.models.nat.nonautoregressive_transformer import NATransformerModel, NATransformerDecoder
from multitasknat.models.nat_ctc import NAT_ctc_model, NAT_ctc_encoder, NAT_ctc_decoder
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder
from fairseq.modules.transformer_layer import TransformerDecoderLayer
import copy
import random
from typing import Optional, List, Dict
from torch import Tensor
from omegaconf import DictConfig

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


class ARTranformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.shallow_at_decoder_layers)
            ]
        )


class myhybrid_decoder(NAT_ctc_decoder):
    def __init__(self, args, dictionary, embed_tokens, 
                 at_dec):
        super().__init__(args, dictionary, embed_tokens)
        #self.nat_dec = nat_dec   nat在super里面就有，只需要额外加at
        self.at_dec = at_dec

#decoder的forward也要写 #inference的时候用到
#不修改了，训练的时候也要用到
    def forward(self, encoder_out, prev_output_tokens, normalize: bool = False, features_only: bool = False):
        features, _ = self.extract_features(
            encoder_out=encoder_out,
            prev_output_tokens=prev_output_tokens
        )
        if features_only:  # used for mt_ctc
            return features, _
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out


@register_model("my_ctc_ar")
class my_ctc_ar_model(NAT_ctc_model):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        #ctc decoder?
        #nat_decoder = NAT_ctc_model.build_decoder(args, tgt_dict, embed_tokens)
        at_dec = ARTranformerDecoder(args, tgt_dict, embed_tokens)
        return myhybrid_decoder(args, tgt_dict, embed_tokens, at_dec)
        

    def add_args(parser):
        NAT_ctc_model.add_args(parser)
        parser.add_argument("--shallow-at-decoder-layers", type=int, metavar='N',
                            help="the number of at decoder.")

    def forward(self, at_src_tokens, nat_src_tokens, src_lengths, prev_nat, prev_at, tgt_tokens, **kwargs):
        nat_encoder_out = self.encoder(nat_src_tokens, src_lengths=src_lengths, **kwargs)
        at_encoder_out = nat_encoder_out
        #nat_decoder_ouput
        if getattr(self.args, "if_deepcopy_at_sample", False):
            at_encoder_out = self.encoder(at_src_tokens, src_lengths=src_lengths, **kwargs)
            nat_decode_output = self.decoder(nat_encoder_out,
                                             prev_nat,
                                             normalize=False,
                                             features_only=False)
            _, dec_each_layer_output_and_attn = self.decoder(at_encoder_out,
                                                             prev_nat,
                                                             normalize=False,
                                                             features_only=True)
        else:
            nat_decode_features, dec_each_layer_output_and_attn = self.decoder(nat_encoder_out,
                                                                               prev_nat,
                                                                               normalize=False,
                                                                               features_only=True)

            nat_decode_output = self.decoder.output_layer(nat_decode_features)

        # AT  decoding
        at_dec_nat_output = []
        #看dec_each_layer_ouput是什么结构
        dec_each_layer_output = dec_each_layer_output_and_attn['inner_states']
        dec_layer_output = dec_each_layer_output[-1]
        ar_dec = self.decoder.at_dec
        shallow_at_encode_output = {
            "encoder_out": [dec_layer_output],
            "encoder_padding_mask": [at_encoder_out["upsample_mask"]]
        }
        at_dec_layer_output, _ = ar_dec(prev_at,
                                                encoder_out=shallow_at_encode_output,
                                                features_only=False,
                                                return_all_hiddens=False)
        at_dec_nat_output.append(at_dec_layer_output)

        return ({
                    "out": nat_decode_output,  # T x B x C
                    "name": "NAT"
                },
                {
                    "out": at_dec_nat_output,  # B x T x C
                    "name": "AT"
                }
        )

    #hzj
    #inference的时候用到
    #直接执行model的decoder的forward函数
    def forward_decoder(
        self,
        tokens,
        encoder_out,
        incremental_state,
        temperature: float = 1.0,
    ):
        at_dec = self.decoder.at_dec
        return at_dec.forward(tokens, encoder_out=encoder_out)
        # at_dec = self.decoder.at_dec
        # decoder_out = at_dec.forward(tokens, encoder_out=encoder_out)
        # attn: Optional[Tensor] = None
        # decoder_len = len(decoder_out)
        # if decoder_len > 1 and decoder_out[1] is not None:
        #     if isinstance(decoder_out[1], Tensor):
        #         attn = decoder_out[1]
        #     else:
        #         print("errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        #     if attn is not None:
        #         attn = attn[:, -1, :]

        # decoder_out_tuple = (
        #     decoder_out[0][:, -1:, :].div_(temperature),
        #     None if decoder_len <= 1 else decoder_out[1],
        # )

        # probs = at_dec.get_normalized_probs(
        #     decoder_out_tuple, log_probs=True, sample=None
        # )
        # probs = probs[:, -1, :]
        # return probs, attn
    

    def forward_encoder(self, encoder_inputs,upsample_scale, src_dict):
        #encoder_outs = self.model.forward_encoder(net_input)
        #input = net_input
        src_tokens = encoder_inputs["src_tokens"].clone()
        upsample_src_tokens = normal(src_tokens, upsample_scale, src_dict)
        prev_nat = upsample_src_tokens
        at_src_tokens, src_lengths, nat_src_tokens = (
            encoder_inputs["src_tokens"],
            encoder_inputs["src_lengths"],
            encoder_inputs["src_tokens"]
        )
        prev_at = encoder_inputs["prev_output_tokens"]
        # at_tgt_tokens, prev_nat, prev_at = at_sample["target"], \
        #                                    nat_sample["prev_target"], \
        #                                    at_sample["net_input"]["prev_output_tokens"]

        nat_encoder_out = self.encoder(nat_src_tokens, src_lengths=src_lengths)
        at_encoder_out = nat_encoder_out
        #nat_decoder_ouput
        if getattr(self.args, "if_deepcopy_at_sample", False):
            at_encoder_out = self.encoder(at_src_tokens, src_lengths=src_lengths)
            nat_decode_output = self.decoder(nat_encoder_out,
                                             prev_nat,
                                             normalize=False,
                                             features_only=False)
            _, dec_each_layer_output_and_attn = self.decoder(at_encoder_out,
                                                             prev_nat,
                                                             normalize=False,
                                                             features_only=True)
        else:
            nat_decode_features, dec_each_layer_output_and_attn = self.decoder(nat_encoder_out,
                                                                               prev_nat,
                                                                               normalize=False,
                                                                               features_only=True)

            nat_decode_output = self.decoder.output_layer(nat_decode_features)

        # AT  decoding
        dec_each_layer_output = dec_each_layer_output_and_attn['inner_states']
        dec_layer_output = dec_each_layer_output[-1]
        shallow_at_encode_output = {
            "encoder_out": [dec_layer_output],
            "encoder_padding_mask": [at_encoder_out["upsample_mask"]]
        }
        return shallow_at_encode_output

    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
        }


    

    # nat ctc的decoder
    # def forward_decoder(self, decoder_out, encoder_out, **kwargs):
    #     # 包含了upsample_x和upsample_mask
    #     history = decoder_out.history
    #     output_tokens = decoder_out.output_tokens
    #     output_scores = decoder_out.output_scores

    #     if self.ctc_decode_with_beam > 1 or (self.ctc_decode_with_beam == 1 and self.use_ctc_bs):
    #         log_probs = self.decoder(
    #             encoder_out=encoder_out,
    #             normalize=True,
    #             prev_output_tokens=output_tokens,
    #         ).transpose(0, 1)
    #         output_masks = output_tokens.ne(self.pad)
    #         output_tokens, output_scores = self.ctc_beamsearch(
    #             log_probs, output_masks, output_tokens, output_scores
    #         )
    #     else:
    #         # greedy decoding
    #         _scores, _tokens = self.decoder(
    #             encoder_out=encoder_out,
    #             prev_output_tokens=output_tokens,
    #             normalize=True
    #         ).transpose(0, 1).max(-1)
    #         output_masks = output_tokens.ne(self.pad)
    #         output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
    #         output_scores.masked_scatter_(output_masks, _scores[output_masks])

    #     return decoder_out._replace(
    #         output_tokens=output_tokens,
    #         output_scores=output_scores,
    #         attn=None,
    #         history=history,
    #     )

    


@register_model_architecture("my_ctc_ar", "my_ctc_ar")
def base_architecture(args):
    # This is actually nat_ctc_decoder.
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.upsample_scale = getattr(args, "upsample_scale", 3)
    args.shallow_at_decoder_layers = getattr(args, "shallow_at_decoder_layers", 1)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


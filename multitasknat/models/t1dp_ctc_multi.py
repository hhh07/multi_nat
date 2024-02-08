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

#hzj
DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)
from fairseq.distributed import fsdp_wrap


class ShallowTranformerDecoder(TransformerDecoder):
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


class dp_decoder(NAT_ctc_decoder):
    def __init__(self, args, dictionary, embed_tokens, at_dec_nat_enc,
                 at_dec_1,
                 at_dec_2,
                 at_dec_3,
                 at_dec_4,
                 at_dec_5,
                 at_dec_6,
                 pos_dec,
                 dphead_dec,
                 dplable_dec):
        super().__init__(args, dictionary, embed_tokens)
        self.at_dec_nat_enc = at_dec_nat_enc
        self.at_dec_1 = at_dec_1
        self.at_dec_2 = at_dec_2
        self.at_dec_3 = at_dec_3
        self.at_dec_4 = at_dec_4
        self.at_dec_5 = at_dec_5
        self.at_dec_6 = at_dec_6
        self.pos_dec = pos_dec
        self.dphead_dec = dphead_dec
        self.dplable_dec = dplable_dec
    

    def forward(self, encoder_out, prev_output_tokens, normalize: bool = False, features_only: bool = False):
        features, _ = self.extract_features(
            encoder_out=encoder_out,
            prev_output_tokens=prev_output_tokens
        )
        if features_only:  # used for mt_ctc
            return features, _
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out


@register_model("t1dp_ctc_multi")
class t1dp_ctc_multi_model(NAT_ctc_model):

    #hzj
    def get_targets(self, sample, net_output, key, output_property=None):
        """
        You can define different case for return according to key and output property.
        """
        if key == "NAT":
            return sample["target"]
        elif key == "AT":
            return sample["target"]
        elif key == "POS":
            return sample["tgt_pos"]
        elif key == "DPHEAD":
            return sample["tgt_dphead"]
        elif key == "DPLABLE":
            return sample["tgt_dplable"]
        else:
            raise NotImplementedError()

    #hzj
    #修改build_decoder的参数

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        #hzj
        pos_dict, dphead_dict, dplabel_dict = task.tgt_pos_dictionary, task.tgt_dphead_dictionary, task.tgt_dplable_dictionary
        

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        
        #hzj dp信息的embedding
        #decoder_embed_path为none，使用随机初始化的词嵌入矩阵
        decoder_embed_pos = cls.build_embedding(
                args, pos_dict, args.decoder_embed_dim, args.decoder_embed_path
            ) 
        decoder_embed_dphead = cls.build_embedding(
                args, dphead_dict, args.decoder_embed_dim, args.decoder_embed_path
            ) 
        decoder_embed_dplabel = cls.build_embedding(
                args, dplabel_dict, args.decoder_embed_dim, args.decoder_embed_path
            ) 

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, pos_dict, dphead_dict, dplabel_dict, decoder_embed_tokens, decoder_embed_pos, decoder_embed_dphead, decoder_embed_dplabel )
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder)

    @classmethod
    def build_decoder(cls, args, tgt_dict, pos_dict, dphead_dict, dplabel_dict, embed_tokens, decoder_embed_pos, decoder_embed_dphead, decoder_embed_dplabel ):
        # nat_decoder = NATransformerModel.build_decoder(args, tgt_dict, embed_tokens)
        if getattr(args, "share_at_decoder", False):
            at_dec = ShallowTranformerDecoder(args, tgt_dict, embed_tokens)
            pos_dec = ShallowTranformerDecoder(args, pos_dict, decoder_embed_pos)
            dphead_dec = ShallowTranformerDecoder(args, dphead_dict, decoder_embed_dphead)
            dplable_dec = ShallowTranformerDecoder(args, dplabel_dict, decoder_embed_dplabel)
            return dp_decoder(args, tgt_dict, embed_tokens,
                                  at_dec, at_dec, at_dec, at_dec, at_dec, at_dec, at_dec,
                                  pos_dec,
                                  dphead_dec,
                                  dplable_dec
                                  )
        else:
            print("hzj-只支持共享decoder,share_at_decoder=true")
        


    def add_args(parser):
        NAT_ctc_model.add_args(parser)
        parser.add_argument("--shallow-at-decoder-layers", type=int, metavar='N',
                            help="the number of at decoder.")
        parser.add_argument("--share-at-decoder", default=True, action='store_true',
                            help='if set, share all at decoder\'s param.')
        parser.add_argument("--is-random", default=False, action='store_true',
                            help='if set, randomly select at decoder layer.')
        parser.add_argument("--without-enc", default=False, action='store_true',
                            help='do not use nat encoder output.')

    def forward(self, at_src_tokens, nat_src_tokens, src_lengths, prev_nat, prev_at, prev_pos, prev_dphead, prev_dplable, tgt_tokens, **kwargs):
        nat_encoder_out = self.encoder(nat_src_tokens, src_lengths=src_lengths, **kwargs)
        at_encoder_out = nat_encoder_out
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

        # AT shallow decoding
        at_dec_nat_output = []
        #hzj
        dphead_dec_output = []
        dplable_dec_output = []
        pos_dec_output = []
        
        if not getattr(self.args, "without_enc", False):
            # on nat encoder
            at_dec_nat_enc_output, _ = self.decoder.at_dec_nat_enc(prev_at,
                                                                   encoder_out=at_encoder_out,
                                                                   features_only=False,
                                                                   return_all_hiddens=False)
            at_dec_nat_output.append(at_dec_nat_enc_output)
        # on each nat decoder layer
        dec_each_layer_output = dec_each_layer_output_and_attn['inner_states']
        #hzj
        #ar层
        dec_dict = {
            1: self.decoder.at_dec_1,
            2: self.decoder.at_dec_2,
            3: self.decoder.at_dec_3,
            4: self.decoder.at_dec_4,
            5: self.decoder.at_dec_5,
            6: self.decoder.at_dec_6
        }
        dec_list = [1, 2, 3, 4, 5, 6]
        if getattr(self.args, "is_random", True):
            random.shuffle(dec_list)
            dec_list = dec_list[:3]
        for idx, dec_layer_output in enumerate(dec_each_layer_output):
            # initial x
            if idx not in dec_list:
                continue
            shallow_at_encode_output = {
                "encoder_out": [dec_layer_output],
                "encoder_padding_mask": [at_encoder_out["upsample_mask"]]
            }
            at_dec_nat_dec_layer_output, _ = dec_dict[idx](prev_at,
                                                           encoder_out=shallow_at_encode_output,
                                                           features_only=False,
                                                           return_all_hiddens=False)
            at_dec_nat_output.append(at_dec_nat_dec_layer_output)

        #dphead层
        #1-6层中选3个
        dphead_lay_list=[1,2,3,4,5,6]
        random.shuffle(dphead_lay_list)
        dphead_lay_list = dphead_lay_list[:1]
        dphead_dec = self.decoder.dphead_dec
        for idx, dec_layer_output in enumerate(dec_each_layer_output):
            # initial x
            if idx not in dphead_lay_list:
                continue
            shallow_dphead_encode_output = {
                "encoder_out": [dec_layer_output],
                "encoder_padding_mask": [at_encoder_out["upsample_mask"]]
            }
            dphead_dec_layer_output, _ = dphead_dec(prev_dphead,
                                                    encoder_out=shallow_dphead_encode_output,
                                                    features_only=False,
                                                    return_all_hiddens=False)
            dphead_dec_output.append(dphead_dec_layer_output)
        
        #dplable层
        #1-6层中选3个
        dplable_lay_list=[1,2,3,4,5,6]
        random.shuffle(dplable_lay_list)
        dplable_lay_list = dplable_lay_list[:1]
        dplable_dec = self.decoder.dplable_dec
        for idx, dec_layer_output in enumerate(dec_each_layer_output):
            # initial x
            if idx not in dplable_lay_list:
                continue
            shallow_dplable_encode_output = {
                "encoder_out": [dec_layer_output],
                "encoder_padding_mask": [at_encoder_out["upsample_mask"]]
            }
            dplable_dec_layer_output, _ = dplable_dec(prev_dplable,
                                                    encoder_out=shallow_dplable_encode_output,
                                                    features_only=False,
                                                    return_all_hiddens=False)
            dplable_dec_output.append(dplable_dec_layer_output)
       
        #pos层
        #1-6层中选3个
        pos_lay_list=[1,2,3,4,5,6]
        random.shuffle(pos_lay_list)
        pos_lay_list = pos_lay_list[:1]
        pos_dec = self.decoder.pos_dec
        for idx, dec_layer_output in enumerate(dec_each_layer_output):
            # initial x
            if idx not in pos_lay_list:
                continue
            shallow_pos_encode_output = {
                "encoder_out": [dec_layer_output],
                "encoder_padding_mask": [at_encoder_out["upsample_mask"]]
            }
            pos_dec_layer_output, _ = pos_dec(prev_pos,
                                                    encoder_out=shallow_pos_encode_output,
                                                    features_only=False,
                                                    return_all_hiddens=False)
            pos_dec_output.append(pos_dec_layer_output)

        return ({
                    "out": nat_decode_output,  # T x B x C
                    "name": "NAT"
                },
                {
                    "out": at_dec_nat_output,  # B x T x C
                    "name": "AT"
                },
                {
                    "out": dphead_dec_output,  # B x T x C
                    "name": "DPHEAD"
                },
                {
                    "out": dplable_dec_output,  # B x T x C
                    "name": "DPLABLE"
                },
                {
                    "out": pos_dec_output,  # B x T x C
                    "name": "POS"
                }
        )


@register_model_architecture("t1dp_ctc_multi", "t1dp_ctc_multi")
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

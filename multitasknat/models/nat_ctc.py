import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.models.fairseq_decoder import FairseqDecoder
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, FairseqNATEncoder, ensemble_decoder
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerModel, TransformerEncoder
from fairseq.models.nat.nonautoregressive_transformer import NATransformerModel, NATransformerDecoder
from fairseq.modules.transformer_layer import TransformerEncoderLayer
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder
from typing import Any, Dict, List, Optional, Tuple
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from omegaconf import DictConfig
import multiprocessing
from ctcdecode import CTCBeamDecoder


@register_model("nat_ctc")
class NAT_ctc_model(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.scale = args.upsample_scale
        self.blank_idx = decoder.dictionary.blank_index
        # self.ctc_decode_with_beam = args.ctc_decode_with_beam
        # self.use_ctc_bs = args.use_ctc_bs
        self.use_ctc_bs = True
        self.ctc_decode_with_beam = 1

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument("--ctcdecoder-positional-embedding", default=False, action='store_true',
                            help='if set, ables ctc decoder\'s positional embeddings (outside self attention)')
        parser.add_argument("--share-all-nat-dec-layer", default=False, action='store_true',
                            help='if set, the model will work on ctc-loss.')
        parser.add_argument(
            "--curriculum-type",
            type=str,
            default="nat",
            help="at_forward or at_backward or nat",
        )
        parser.add_argument("--is-lp", default=False, action='store_true',
                            help='if set, the model will work on lp.')

    @classmethod
    def build_encoder(cls, args, src_dict, encoder_embed_tokens):
        """
        The encoder implements upsampling.
        """
        return NAT_ctc_encoder(args, src_dict, encoder_embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        """
        The decoder is actually TransformerEncoder.
        """
        return NAT_ctc_decoder(args, tgt_dict, embed_tokens)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens,at_prev_output_tokens=None,**kwargs):
        #ctc的长度是src*2，与tgt无关
        # if hasattr(self.args, 'curriculum_type') and self.args.curriculum_type != 'nat':
        #     prev_output_tokens = at_prev_output_tokens
        if hasattr(self.args, 'curriculum_type') and self.args.curriculum_type == 'at_backward':
            #prev_output_tokens = at_prev_output_tokens
            tmp_tgt_tokens = tgt_tokens.clone()
            tmp_prev_output_tokens = prev_output_tokens.clone()
            nonpad_num = tmp_tgt_tokens.ne(self.pad).sum(1)
            tmp_a = nonpad_num.repeat(max(nonpad_num),1).transpose(0,1)
            tmp_b = (torch.arange(max(nonpad_num))).unsqueeze(0).repeat(nonpad_num.size(0),1).cuda()
            tmp_index = tmp_a - tmp_b - 1
            mask = tmp_index.lt(0)
            tmp_index=tmp_index.masked_fill(mask, max(nonpad_num)-1)
            tgt_tokens = torch.gather(tmp_tgt_tokens, 1, tmp_index)
            # 正确反向target： C B A [EOS]
            # if hasattr(self.args, 'right_type') and self.args.right_type == 'first':
            #     tmp_prev_output_tokens = torch.gather(tmp_prev_output_tokens, 1, tmp_index)
            #     # for second decoding to update
            #     tgt_tokens = tmp_prev_output_tokens
        encoder_out = self.encoder(src_tokens, src_lengths)
        output, output_logits_list = self.decoder(encoder_out, prev_output_tokens, normalize=False)
        return output, output_logits_list

    '''
    def initialize_output_tokens(self, encoder_out, src_tokens):
        # 不能像之前一样简单粗暴地用unk，因为本模型没有encoder，
        # 所以应该要把待翻译的src句子embed+upsample之后的结果返回
        # encoder_out为(upsample_x, upsample_mask)
        initial_output_tokens = src_tokens.unsqueeze(-1).expand(-1, -1, self.scale).reshape(batch_size, -1)
        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(initial_output_tokens)
        
        # We just use encoder_out at forward_decoder and dont use output_tokens and output_scores.
        # Usually use encoder_out to predict length, and create initial_output_tokens
        # with length predicted. but in NAT_CTC model we dont need to predict length, 
        # so we write this function just for matching iterative_refinement_generator.
        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )
    '''

    def initialize_output_tokens(self, encoder_out, src_tokens):
        upsample_x = encoder_out["upsample_x"]
        upsample_mask = encoder_out["upsample_mask"]
        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), src_tokens.size(1) * self.scale
        ).fill_(self.pad)
        initial_output_tokens = initial_output_tokens.masked_fill_(~upsample_mask, self.unk)
        if self.ctc_decode_with_beam is not None:
            if self.ctc_decode_with_beam > 1:
                initial_output_tokens = initial_output_tokens.unsqueeze(0).expand(self.ctc_decode_with_beam, -1, -1)
                initial_output_tokens = initial_output_tokens.reshape(-1, src_tokens.size(1) * self.scale)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(upsample_x)
        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def _convert_lengths_to_mask(self, lens, maxlen=None):
        # lens: (bsz)
        maxlen = maxlen or lens.max()
        lens = lens.view(-1)
        mask = torch.arange(maxlen, device=lens.device)[None, :] < lens[:, None]
        return mask

    def ctc_beamsearch(
            self, log_probs, probs_mask, output_tokens, src_tokens=None
    ):
        def _get_ctcdecoder():
            if not hasattr(self, 'ctcdecoder'):
                lang = "tgt"
                labels = self.tgt_dict.symbols

                nproc = multiprocessing.cpu_count()

                ctcdecoder = CTCBeamDecoder(
                    labels,
                    model_path=None,
                    alpha=0,
                    beta=0,
                    cutoff_top_n=max(40, self.ctc_decode_with_beam),
                    cutoff_prob=1.0,
                    beam_width=self.ctc_decode_with_beam,
                    num_processes=nproc,
                    blank_id=self.blank_idx,
                    log_probs_input=True
                )
                self.ctcdecoder = ctcdecoder
            return self.ctcdecoder

        decoder = _get_ctcdecoder()
        probs_lens = probs_mask.int().sum(-1)
        device = probs_lens.device

        k = self.ctc_decode_with_beam if self.ctc_decode_with_beam >= 10 else 10
        log_probs, idx = torch.topk(log_probs, k, dim=-1)
        # BATCHSIZE x N_BEAMS X N_TIMESTEPS
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(log_probs, idx, probs_lens)

        bbsz = beam_results.size(0) * beam_results.size(1)
        beam_results = beam_results.type_as(output_tokens).long().view(bbsz, -1)
        beam_scores = beam_scores.type_as(output_tokens).view(bbsz)
        out_lens = out_lens.type_as(output_tokens).view(bbsz)
        beam_results = beam_results[:, :max(out_lens) + 1]

        beam_mask = self._convert_lengths_to_mask(
            out_lens, maxlen=beam_results.size(-1))
        beam_results = beam_results.masked_fill_(~beam_mask, self.pad)

        # beam_scores = (1 / np.exp(beam_scores)).unsqueeze(-1).expand_as(beam_results)
        beam_scores = -(beam_scores / out_lens).unsqueeze(-1).expand_as(beam_results)
        # beam_scores = beam_scores.to(device)

        return beam_results, beam_scores

    def forward_decoder(self, decoder_out, encoder_out, **kwargs):
        # 包含了upsample_x和upsample_mask
        history = decoder_out.history
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores

        if self.ctc_decode_with_beam > 1 or (self.ctc_decode_with_beam == 1 and self.use_ctc_bs):
            output,_  = self.decoder(
                encoder_out=encoder_out,
                normalize=True,
                prev_output_tokens=output_tokens,
            )
            log_probs = output.transpose(0, 1)
            output_masks = output_tokens.ne(self.pad)
            output_tokens, output_scores = self.ctc_beamsearch(
                log_probs, output_masks, output_tokens, output_scores
            )
        else:
            # greedy decoding
            output,_= self.decoder(
                encoder_out=encoder_out,
                prev_output_tokens=output_tokens,
                normalize=True
            )
            _scores, _tokens = output.transpose(0, 1).max(-1)
            output_masks = output_tokens.ne(self.pad)
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def get_normalized_probs_scriptable(
            self,
            net_output,
            log_probs: bool,
            sample
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        if isinstance(net_output, tuple):
            logits = net_output[0]
        else:
            logits = net_output
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def get_targets(self, sample, net_output, key, output_property=None):
        """
        You can define different case for return according to key and output property.
        """
        if key == "NAT":
            return sample["target"]
        elif key == "AT":
            return sample["target"]
        else:
            raise NotImplementedError()

    def prepare_for_inference_(self, cfg: DictConfig):
        """Prepare model for inference."""
        kwargs = {}
        kwargs["beamable_mm_beam_size"] = (
            None
            if getattr(cfg.generation, "no_beamable_mm", False)
            else getattr(cfg.generation, "beam", 5)
        )
        kwargs["need_attn"] = getattr(cfg.generation, "print_alignment", False)
        if getattr(cfg.generation, "retain_dropout", False):
            kwargs["retain_dropout"] = cfg.generation.retain_dropout
            kwargs["retain_dropout_modules"] = cfg.generation.retain_dropout_modules
        self.ctc_decode_with_beam = getattr(cfg.task, "ctc_decode_with_beam", 1)
        self.use_ctc_bs = getattr(cfg.task, "use_ctc_bs", True)
        self.args.ctc_decode_with_beam = getattr(cfg.task, "ctc_decode_with_beam", 1)
        self.args.use_ctc_bs = getattr(cfg.task, "use_ctc_bs", True)
        self.make_generation_fast_(**kwargs)


class NAT_ctc_encoder(FairseqNATEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        embed_dim = embed_tokens.embedding_dim
        self.scale = args.upsample_scale
        self.upsample_Linear = nn.Linear(embed_dim, self.scale * embed_dim)

    def forward(self, src_tokens, src_lengths, token_embeddings: Optional[torch.Tensor] = None, **kwargs):
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        # if encoder_padding_mask is not None:
        #     x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # umsample x
        (sequence_length, batch_size, embed_dim) = x.shape
        upsample_x = x.transpose(0, 1)  # B x T x C
        upsample_x = self.upsample_Linear(upsample_x)
        # reshape x: B x upsample_scale*T x C
        upsample_x = upsample_x.reshape(batch_size, sequence_length * self.scale, embed_dim)
        # upsample mask
        upsample_mask = encoder_padding_mask.unsqueeze(-1).expand(-1, -1, self.scale)
        upsample_mask = upsample_mask.reshape(batch_size, -1)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            # "encoder_embedding": [encoder_embedding],  # B x T x C
            # "encoder_states": encoder_states,  # List[T x B x C]
            # "src_tokens": [],
            # "src_lengths": [],
            "upsample_x": upsample_x,  # B x upsample_scale*T x C
            "upsample_mask": upsample_mask  # B x upsample_scale*T
        }


class NAT_ctc_decoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "share_all_nat_dec_layer", False):
            if self.decoder_layerdrop > 0.0:
                self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
            else:
                self.layers = nn.ModuleList([])
            layer = self.build_decoder_layer(args)
            self.layers.extend(
                [
                    layer
                    for _ in range(args.decoder_layers)
                ]
            )
        if not args.ctcdecoder_positional_embedding:
            self.embed_positions = None
        # hzj dslp
        self.reduce_concat = torch.nn.ModuleList(
                    [torch.nn.Linear(self.args.decoder_embed_dim*2, self.args.decoder_embed_dim, bias=False)
                                                          for _ in range(self.args.decoder_layers - 1)])

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        return self.output_projection(features)

    def forward(self, encoder_out, prev_output_tokens, normalize: bool = False, features_only: bool = False, **kwargs):
        """
        prev_output_tokens: (bsz, src_len*upsample_scale) with all unk
        the prev_output_tokens only helps ctc decoder's positional embedding
        you can use it with setting args.ctcdecoder_positional_embeddings True.
        """
        features , all_features = self.extract_features(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out
        )
        all_layer_output_logits = all_features['all_layer_output_logits']
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out , [F.log_softmax(x, -1) if normalize else x
                for x in all_layer_output_logits]
        
        # features, _ = self.extract_features(
        #     encoder_out=encoder_out,
        #     prev_output_tokens=prev_output_tokens
        # )
        # if features_only:  # used for mt_ctc_6_up_6
        #     return features
        # decoder_out = self.output_layer(features)
        # return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    def extract_features(self, encoder_out, prev_output_tokens, incremental_state=None,early_exit=None):
        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        # embed tokens and positions
        x = encoder_out["upsample_x"]  # B x self.scale*T x C
        self_attn_padding_mask = encoder_out["upsample_mask"]

        #fbd         #把x的内容反转了
        src_embd = x
        src_mask = self_attn_padding_mask
        if hasattr(self.args, 'curriculum_type') and self.args.curriculum_type == 'at_backward':
                tmp_prev_output_tokens = prev_output_tokens.clone()
                nonpad_num = tmp_prev_output_tokens.ne(self.padding_idx).sum(1)
                tmp_a = nonpad_num.repeat(max(nonpad_num), 1).transpose(0, 1)
                tmp_b = (torch.arange(max(nonpad_num))).unsqueeze(0).repeat(nonpad_num.size(0), 1).cuda()
                tmp_index = tmp_a - tmp_b - 1
                mask = tmp_index.lt(0)
                tmp_index = tmp_index.masked_fill(mask, max(nonpad_num) - 1)
                x = torch.gather(x, 1, tmp_index.unsqueeze(-1).repeat(1, 1, x.size(2)))

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        all_layer_output_logits = []

        for idx, layer in enumerate(self.layers):
            layer_out_logits = self.output_layer(x)
            #获得解析的单词的embedding  T x B x C
            layer_out = self.embed_tokens(layer_out_logits.argmax(dim=-1))

            if  idx == 0 or (not getattr(self.args, "is_lp", False)) :
                new_x = x
            else:
                all_layer_output_logits.append(layer_out_logits)
                #此处修改为 加mask的avg  用masked_avg_layer_out替换
                # new_x = torch.cat((x, layer_out), dim=-1)
                # new_x = self.reduce_concat[idx - 1](new_x)


                #####计算mask_avg_layer_out
                #矩阵计算方法 问题OOM
                # T = layer_out.size()[0]
                # # T x T x B x C
                # layout_repeat= layer_out.unsqueeze(0).repeat(T,1,1,1)
                # #下三角矩阵
                # lower_triangular = torch.tril(torch.ones(T, T))
                # # T x T x 1 x 1
                # lower_triangular = lower_triangular.unsqueeze(-1).unsqueeze(-1)
                # # T x T x B x C
                # lower_triangular = lower_triangular.to(device="cuda")
                # masked_layer_out = layout_repeat * lower_triangular
                # # T x B x C 在第二维度（token维度求和）
                # token_count = torch.arange(1, T + 1).unsqueeze(-1).unsqueeze(-1)
                # token_count = token_count.to(device="cuda")
                # avg_masked_layer_out = torch.sum(masked_layer_out,dim=1) / token_count

                #for循环计算方法
                T = layer_out.size()[0]
                masked_layer_out = torch.zeros_like(layer_out)
                temp = torch.zeros_like(layer_out[0])
                for t in range(T):
                    temp = temp + x[t]
                    masked_layer_out[t] = temp
                token_count = torch.arange(1, T + 1).unsqueeze(-1).unsqueeze(-1)
                token_count = token_count.to(device="cuda")
                avg_masked_layer_out = masked_layer_out / token_count


                #线性层
                new_x = torch.cat((x, avg_masked_layer_out), dim=-1)
                new_x = new_x.half()
                new_x = self.reduce_concat[idx - 1](new_x)

            #fbd
            if hasattr(self.args, 'curriculum_type') and self.args.curriculum_type!='nat':
                self_attn_mask=self.buffered_future_mask(x)
            else:
                self_attn_mask=None
            
            x, layer_attn, _ = layer(
                new_x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            inner_states.append(x)
        #hzj
        #第六个是第六层线性层后加
        # layernum x T x B x dict_size
        all_layer_output_logits.append(self.output_layer(x))
        

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states,"all_layer_output_logits": all_layer_output_logits}


@register_model_architecture("nat_ctc", "nat_ctc")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.upsample_scale = getattr(args, "upsample_scale", 3)

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

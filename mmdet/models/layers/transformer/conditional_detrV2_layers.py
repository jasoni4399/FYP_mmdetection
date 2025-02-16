# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import (Linear,build_norm_layer)
import warnings
from typing import Tuple,Union
from mmcv.cnn.bricks.transformer import FFN,MultiheadAttention
from torch import Tensor, nn
from torch.nn import ModuleList
from mmengine.model import BaseModule
from mmengine import ConfigDict
import torch.nn.functional as F
import torch.nn as nn

from mmdet.utils import OptConfigType, OptMultiConfig,ConfigType
from mmcv.cnn.bricks.drop import Dropout
from .detr_layers import DetrTransformerDecoder, DetrTransformerDecoderLayer

from .utils import MLP, ConditionalAttention, coordinate_to_encoding,inverse_sigmoid

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

class ConditionalDetrTransformerV2Decoder(DetrTransformerDecoder):
    """Decoder of Conditional DETR."""
    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 post_norm_cfg: OptConfigType = dict(type='LN'),
                 return_intermediate: bool = True,
                 init_cfg: Union[dict, ConfigDict] = None,
                 positional_encoding: OptConfigType = None,
                 content_width: list=[0.4],
                 content_height: list=[0.4]) -> None:
        super().__init__(init_cfg=init_cfg,
                         layer_cfg = layer_cfg,
                         num_layers = num_layers,
                         post_norm_cfg = post_norm_cfg,
                         return_intermediate = return_intermediate)
        
        self.content_width=content_width
        self.content_height=content_height
        print(self.content_width,self.content_height)
        assert len(self.content_width) == len(self.content_height)
        self._init_layers()
        
    def _init_layers(self) -> None:
        """Initialize decoder layers and other layers."""
        self.layers = ModuleList([
            ConditionalDetrTransformerV2DecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg,
                                          self.embed_dims)[1]
        # conditional detr affline
        self.query_scale = MLP(self.embed_dims, self.embed_dims,
                               self.embed_dims, 2)
        self.ref_point_head = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 2)

        #self.lambda_q=MLP(self.embed_dims, self.embed_dims,
        #                       self.embed_dims, 2)
        self.ref_select=MLP(self.embed_dims, self.embed_dims,
                            2, 2)
        self.key_select=MLP(self.embed_dims, self.embed_dims,
                            2, 2)
        self.content_query=MLP(self.embed_dims*2, self.embed_dims,
                               self.embed_dims, 2)
        self.box_estimation=MLP(self.embed_dims, self.embed_dims,
                               self.embed_dims, 2)
        self.reg_ffn = FFN(
            4,
            4,
            2,
            dict(type='ReLU', inplace=True),
            dropout=0.0,
            add_residual=False)

        self.fc_reg = MLP(4, self.embed_dims,
                               4, 2)
        # we have substitute 'qpos_proj' with 'qpos_sine_proj' except for
        # the first decoder layer), so 'qpos_proj' should be deleted
        # in other layers.
        for layer_id in range(self.num_layers - 1):
            self.layers[layer_id + 1].cross_attn.qpos_proj = None

    def forward(self,
                query: Tensor,
                key: Tensor = None,#encoder embedding
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                key_padding_mask: Tensor = None):
        """Forward function of decoder.

        Args:
            query (Tensor): The input query with shape
                (bs, num_queries, dim).
            key (Tensor): The input key with shape (bs, num_keys, dim) If
                `None`, the `query` will be used. Defaults to `None`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`. If not `None`, it will be added to
                `query` before forward function. Defaults to `None`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If `None`, and `query_pos`
                has the same shape as `key`, then `query_pos` will be used
                as `key_pos`. Defaults to `None`.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_keys).
                Defaults to `None`.
        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). References with shape
            (bs, num_queries, 2).
        """

        #V1
        #reference_unsigmoid = self.ref_point_head(
        #    query_pos)  # [bs, num_queries, 2]
        #reference= reference_unsigmoid.sigmoid()
        #reference_xy=reference[...,:2]
        #V2 box query

        bs,num_queries,dim=query_pos.size()
        #print(bs,num_queries,dim)
        
        #V2
        #reference_point
        #s=FFN(x(Cx,Cy))
        reference_unsigmoid=self.ref_select(key_pos)
        reference_point_selection=reference_unsigmoid[...,:2]

        #selection
        lambda_q = self.ref_point_head(key_pos)# [bs, num_keys, dim]

        reference_point_selection_choose=reference_point_selection.clone()
        #(Cx,Cy)
        key_pos_selection=key_pos.clone()
        key_pos_selection=self.key_select(key_pos_selection)
        key_pos_selection=key_pos_selection[...,:2]

        k=self.box_estimation(key_pos)
        #k=k[...,:2]

        #reference_selected=torch.empty(bs,num_queries,2,device=query_pos.device)
        #key_pos_selected=torch.empty(bs,num_queries,2,device=query_pos.device)
        #lambda_q_selected=torch.empty(bs,num_queries,self.embed_dims,device=query_pos.device)

        def select(before_select ,selection_reference ,bs,num_q,dims):
            selection=torch.empty(bs,num_q,dims,device=query_pos.device)
            for i in range(bs):
                selected=before_select[i][:][selection_reference[i][:,0]== torch.max(selection_reference[i][:,0])]
                if selected.size(0)<num_q:
                    selected = F.pad(selected, (0,0,0,num_q-selected.size(0)), "constant",0)
                elif selected.size(0)>=num_q:
                    selected=selected[:num_q,:dims]
                selection[i]=selected
            return selection
        
        reference_selected=select(reference_point_selection_choose,reference_point_selection_choose,
                                  bs,num_queries,2)
        key_pos_selected=select(key_pos_selection,reference_point_selection_choose,
                                bs,num_queries,2)
        lambda_q_selected=select(lambda_q,reference_point_selection_choose,
                                 bs,num_queries,self.embed_dims)
        k_selected=select(k,reference_point_selection_choose,
                                 bs,num_queries,self.embed_dims)

        #Ps
        selected_reference_sigmoid=reference_selected.sigmoid()
        selected_reference_xy = selected_reference_sigmoid[...,:2]

        #lambda_q = lambda_q.sigmoid()
        #lambda_q=FFN(x(Cx,Cy))
        #print("lambda_q:",lambda_q.size())

        #Cq initial by image content
        #query=self.content_query(reference_xy)
        #or
        #content_w_h=torch.tensor([self.content_width,self.content_height],device=query_pos.device)
        #content_w_h=content_w_h.unsqueeze(0).repeat(key_pos_selected[..., :2].size(0),key_pos_selected[..., :2].size(1),1)
        content_w_h=torch.tensor([self.content_width,self.content_height],device=query_pos.device)
        content_w_h=content_w_h.permute(1,0)
        content_w_h=content_w_h.unsqueeze(0).repeat(key_pos_selected[..., :2].size(0),key_pos_selected[..., :2].size(1),1)
        
        key_pos_selected=key_pos_selected.repeat(1,1,len(self.content_width))
        key_pos_selected=key_pos_selected.view(key_pos_selected.size(0),key_pos_selected.size(1)*len(self.content_width),key_pos_selected.size(2)//len(self.content_width))

        k_selected=k_selected.repeat(1,1,len(self.content_width))
        k_selected=k_selected.view(k_selected.size(0),k_selected.size(1)*len(self.content_width),k_selected.size(2)//len(self.content_width))

        pe_before=inverse_sigmoid(torch.cat([key_pos_selected, content_w_h],dim=2).permute(2,1,0)).permute(2,1,0)#
        #print("pe_before",pe_before.size())
        #a=
        #print("a",a.size())
        tmp_reg_preds = self.fc_reg(self.reg_ffn(pe_before))
        tmp_reg_preds=k_selected[...,:num_queries, :4]+tmp_reg_preds[...,:num_queries,:4]
        pe=coordinate_to_encoding(coord_tensor=tmp_reg_preds.sigmoid())
        #pe: torch.Size([2, 300, 512])
        #print("pe",pe.size())
        query=self.content_query(pe)
        #print("query:",query.size())
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                pos_transformation = self.query_scale(lambda_q_selected)
            else:
                pos_transformation = self.query_scale(query) #lambda_q
            # get sine embedding for the query reference 
            ref_sine_embed = coordinate_to_encoding(coord_tensor=selected_reference_xy)#Ps
            #print("ref_sine_embed",ref_sine_embed.size())
            # apply transformation
            ref_sine_embed = ref_sine_embed * pos_transformation
            #print("ref_sine_embed_tran",ref_sine_embed.size())
            query = layer(
                query,#box query
                key=key,#encoder embedding
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                ref_sine_embed=ref_sine_embed,
                is_first=(layer_id == 0))
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))

        if self.return_intermediate:
            return torch.stack(intermediate), selected_reference_xy

        query = self.post_norm(query)
        return query.unsqueeze(0), selected_reference_xy
    

class ConditionalDetrTransformerV2Encoder(BaseModule):
    """Encoder of DETR.

    Args:
        num_layers (int): Number of encoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        num_cp (int): Number of checkpointing blocks in encoder layer.
            Default to -1.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 num_cp: int = -1,
                 init_cfg: OptConfigType = None,
                 content_width: list=[0.4],
                 content_height: list=[0.4]) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self.content_width=content_width
        self.content_height=content_height
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            ConditionalDetrTransformerV2EncoderLayer(**self.layer_cfg,content_width=self.content_width,content_height=self.content_height)
            for _ in range(self.num_layers)
        ])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of encoder.

        Args:
            query (Tensor): Input queries of encoder, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional embeddings of the queries, has
                shape (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).

        Returns:
            Tensor: Has shape (bs, num_queries, dim) if `batch_first` is
            `True`, otherwise (num_queries, bs, dim).
        """
        for layer in self.layers:
            query = layer(query, query_pos, key_padding_mask, **kwargs)
        return query


class ConditionalDetrTransformerV2DecoderLayer(DetrTransformerDecoderLayer):
    """Implements decoder layer in Conditional DETR transformer."""

    def _init_layers(self):
        """Initialize self-attention, cross-attention, FFN, and
        normalization."""
        self.self_attn = ConditionalAttention(**self.self_attn_cfg)
        self.cross_attn = ConditionalAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)

        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_masks: Tensor = None,
                cross_attn_masks: Tensor = None,
                key_padding_mask: Tensor = None,
                ref_sine_embed: Tensor = None,
                is_first: bool = False):
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim)
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be
                added to `query` before forward function. Defaults to `None`.
            ref_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`. Defaults to None.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not None, it will be added to
                `key` before forward function. If None, and `query_pos` has
                the same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_masks (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), Same in `nn.MultiheadAttention.
                forward`. Defaults to None.
            cross_attn_masks (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), Same in `nn.MultiheadAttention.
                forward`. Defaults to None.
            key_padding_mask (Tensor, optional): ByteTensor, has shape
                (bs, num_keys). Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder. Defaults to False.

        Returns:
            Tensor: Forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_masks)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_masks,
            key_padding_mask=key_padding_mask,
            ref_sine_embed=ref_sine_embed,
            is_first=is_first)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query

class ConditionalDetrTransformerV2EncoderLayer(BaseModule):
    """Implements encoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, attn_drop_H = 0.1, attn_drop_W = 0.1,),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 content_width: list=[0.1,0.2,0.4],
                 content_height: list=[0.1,0.2,0.4]) -> None:

        super().__init__(init_cfg=init_cfg)

        self.HV_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.HV_attn_cfg:
            self.HV_attn_cfg['batch_first'] = True
        else:
            assert self.HV_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
        self.content_width=content_width
        self.content_height=content_height

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        #self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.self_attn = HVAttention(**self.HV_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor,**kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            content_width=self.content_width,
            content_height=self.content_height,
            **kwargs)
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query

class HVAttention(BaseModule):
    """A wrapper of conditional attention, dropout and residual connection.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop: A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        cross_attn (bool): Whether the attention module is for cross attention.
            Default: False
        keep_query_pos (bool): Whether to transform query_pos before cross
            attention.
            Default: False.
        batch_first (bool): When it is True, Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default: True.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 attn_drop_H: float = 0.,
                 attn_drop_W: float = 0.,
                 proj_drop: float = 0.,
                 cross_attn: bool = False,
                 keep_query_pos: bool = False,
                 batch_first: bool = True,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        assert batch_first is True, 'Set `batch_first`\
        to False is NOT supported in ConditionalAttention. \
        First dimension of all DETRs in mmdet is `batch`, \
        please set `batch_first` to True.'

        self.cross_attn = cross_attn
        self.keep_query_pos = keep_query_pos
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn_drop_H = Dropout(attn_drop_H)
        self.attn_drop_W = Dropout(attn_drop_W)
        self.proj_drop = Dropout(proj_drop)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers for qkv projection."""
        embed_dims = self.embed_dims
        self.out_proj = Linear(embed_dims*2, embed_dims)
        self.query_proj = Linear(embed_dims, embed_dims)
        self.qpos_proj = Linear(embed_dims, embed_dims)
        self.kpos_proj = Linear(embed_dims, embed_dims)
        self.key_proj =  Linear(embed_dims, embed_dims)
        self.value_proj_H = Linear(embed_dims, embed_dims)
        self.value_proj_W = Linear(embed_dims, embed_dims)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward_attn(self,
                     query: Tensor,
                     key: Tensor,
                     value: Tensor,
                     attn_mask: Tensor = None,
                     key_padding_mask: Tensor = None,
                     feats_height=None, feats_width=None,**kwargs) -> Tuple[Tensor]:
        """Forward process for `ConditionalAttention`.

        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                embed_dims].
            key (Tensor): The key tensor with shape [bs, num_keys,
                embed_dims].
                If None, the `query` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tuple[Tensor]: Attention outputs of shape :math:`(N, L, E)`,
            where :math:`N` is the batch size, :math:`L` is the target
            sequence length , and :math:`E` is the embedding dimension
            `embed_dim`. Attention weights per head of shape :math:`
            (num_heads, L, S)`. where :math:`N` is batch size, :math:`L`
            is target sequence length, and :math:`S` is the source sequence
            length.
        """

        
        assert feats_height is not None
        assert feats_width is not None
               #src_len       #
        assert key.size(1) == value.size(1), \
            f'{"key, value must have the same sequence length"}'
               #bs 
        assert query.size(0) == key.size(0) == value.size(0), \
            f'{"batch size must be equal for query, key, value"}'
               #hidden_dims
        assert query.size(2) == key.size(2), \
            f'{"q_dims, k_dims must be equal"}'
        assert value.size(2) == self.embed_dims, \
            f'{"v_dims must be equal to embed_dims"}'
        #2, 300    , 256
        bs, tgt_len, hidden_dims = query.size()
        _, src_len, _ = key.size()
        #e.g. 256//8=32
        head_dims = hidden_dims // self.num_heads
        #embed_dims=256, num_heads=8, 
        #v_head_dims=32
        v_head_dims = self.embed_dims // self.num_heads
        assert head_dims * self.num_heads == hidden_dims, \
            f'{"hidden_dims must be divisible by num_heads"}'
        scaling = float(head_dims)**-0.5

        q = query * scaling
        k = key
        v = value

        #None
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or \
                   attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or \
                   attn_mask.dtype == torch.uint8 or \
                   attn_mask.dtype == torch.bool, \
                   'Only float, byte, and bool types are supported for \
                    attn_mask'

            if attn_mask.dtype == torch.uint8:
                warnings.warn('Byte tensor for attn_mask is deprecated.\
                     Use bool tensor instead.')
                attn_mask = attn_mask.to(torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(1), key.size(1)]:
                    raise RuntimeError(
                        'The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                        bs * self.num_heads,
                        query.size(1),
                        key.size(1)
                ]:
                    raise RuntimeError(
                        'The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(
                        attn_mask.dim()))
        # attn_mask's dim is 3 now.
        #None
        if key_padding_mask is not None and key_padding_mask.dtype == int:
            key_padding_mask = key_padding_mask.to(torch.bool)
                                         #2 , 300, 8, 32
       
        #None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bs
            assert key_padding_mask.size(1) == src_len
        # Q mul K.transpose
        #attn_output_weights = torch.bmm(q, k.transpose(1, 2)) 
        #assert list(attn_output_weights.size()) == [
        #    bs * self.num_heads, tgt_len, src_len
        #]
        #None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask
        #None
        #if key_padding_mask is not None:
        #    attn_output_weights = attn_output_weights.view(
        #        bs, self.num_heads, tgt_len, src_len)
        #    attn_output_weights = attn_output_weights.masked_fill(
        #        key_padding_mask.unsqueeze(1).unsqueeze(2),
        #        float('-inf'),
        #    )
        #    attn_output_weights = attn_output_weights.view(
        #        bs * self.num_heads, tgt_len, src_len)
        
        q_H=q.contiguous().view(bs, tgt_len, self.num_heads,
                                            head_dims).permute(0, 2, 1, 3).flatten(0, 1)#.permute(1, 0, 2, 3)
        q_W=q.contiguous().view(bs, tgt_len, self.num_heads,
                                            head_dims).permute(0, 2, 1, 3).flatten(0, 1)#.permute(1, 0, 2, 3)
        #print("q_H",q_H.size())
        #print("q_W",q_W.size())
        k_H=k.contiguous().view(bs, feats_height,feats_width, self.num_heads,
                                            head_dims).permute(0, 3, 1, 2,
                                                            4).flatten(0, 1)
        k_H=torch.sum(k_H, dim=2)/feats_width
        k_W=k.contiguous().view(bs, feats_height,feats_width, self.num_heads,
                                            head_dims).permute(0, 3, 2, 1,
                                                            4).flatten(0, 1)
        k_W=torch.sum(k_W, dim=2)/feats_height
        

        #print("k_H",k_H.size())
        #print("k_W",k_W.size())

        # Compute attention scores
        attn_output_weights_H = torch.matmul(q_H, k_H.transpose(1, 2))
        attn_output_weights_W = torch.matmul(q_W, k_W.transpose(1, 2))

        attn_output_weights_H=F.softmax(
            attn_output_weights_H -
            attn_output_weights_H.max(dim=-1,keepdim=True)[0],
            dim=-1)
        attn_output_weights_H=self.attn_drop_H(attn_output_weights_H)
        attn_output_weights_W=F.softmax(
            attn_output_weights_W -
            attn_output_weights_W.max(dim=-1,keepdim=True)[0],
            dim=-1)
        attn_output_weights_W=self.attn_drop_W(attn_output_weights_W)

        #print("attn_output_weights_H",attn_output_weights_H.size())
        #print("attn_output_weights_W",attn_output_weights_W.size())

        # Apply attention to values (V is reshaped to rows/columns)
        v_H = self.value_proj_H(v).contiguous().view(bs, feats_height,feats_width, self.num_heads,
                                            v_head_dims).permute(0, 3, 2, 1,
                                                                4).flatten(0, 2)
        v_W = self.value_proj_W(v).contiguous().view(bs, feats_height,feats_width, self.num_heads,
                                            v_head_dims).permute(0, 3, 1, 2,
                                                                4).flatten(0, 2)
        #print("v_H",v_H.size())
        #print("v_W",v_W.size())
        attn_output_weights = torch.cat((attn_output_weights_H,attn_output_weights_W),
                                                dim=2).view(bs*self.num_heads,src_len, feats_height+feats_width)
        attn_output_weights_H=attn_output_weights_H.view(bs*self.num_heads, feats_height,feats_width,feats_height).permute(0,2,1,3).flatten(0,1)
        attn_output_weights_W=attn_output_weights_W.view(bs*self.num_heads, feats_height,feats_width,feats_width).permute(0,1,2,3).flatten(0,1)
        #print(attn_output_weights_H.size())
        #print(attn_output_weights_W.size())
        out_H = torch.bmm(attn_output_weights_H,v_H).view(bs*self.num_heads,tgt_len,v_head_dims) # (B*num_heads, W, H, head_dims)
        out_W = torch.bmm(attn_output_weights_W,v_W).view(bs*self.num_heads,tgt_len,v_head_dims)  # (B*num_heads, H, W, head_dims)
        #print(out_H.size())
        #print(out_W.size())
        # Concatenate and reshape
        out = torch.cat([out_H, out_W], dim=-1)
        out = out.reshape(bs, self.num_heads, feats_height, feats_width, 2*head_dims)
        out = out.permute(0, 2, 3, 1, 4).flatten(3, 4).flatten(1, 2)  # (B, H*W, 2C)
        #print(out.size())

        attn_output = self.out_proj(out)
        #print(attn_output.size())

        attn_output_weights_W=self.attn_drop_W(attn_output_weights_W)
        
        #second ouput never used
        return attn_output, (attn_output_weights_W.sum(dim=1)) / self.num_heads

    def forward(self,
                query: Tensor,
                key: Tensor,
                value=Tensor,
                query_pos: Tensor = None,
                ref_sine_embed: Tensor = None,
                key_pos: Tensor = None,
                attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                is_first: bool = False,
                feats_height=None, feats_width=None,**kwargs) -> Tensor:
        """Forward function for `ConditionalAttention`.
        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                embed_dims].
            key (Tensor): The key tensor with shape [bs, num_keys,
                embed_dims].
                If None, the `query` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query in self
                attention, with the same shape as `x`. If not None, it will
                be added to `x` before forward function.
                Defaults to None.
            query_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`. If not None, it
                will be added to `x` before forward function.
                Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder.
                Defaults to False.
        Returns:
            Tensor: forwarded results with shape
            [bs, num_queries, embed_dims].
        """

        #q = torch.cat([q, query_sine_embed], dim=3).view(bs, nq, 2 * c)
        #k = torch.cat([k, k_pos], dim=3).view(bs, hw, 2 * c)

        #self attention for encoder

        #b,_,H,W=query.size()
        value = query
        query = self.query_proj(query)
        key = self.key_proj(key)
        
        
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query_pos = self.qpos_proj(query_pos)
            query = query + query_pos
        if key_pos is not None:
            key_pos = self.kpos_proj(key_pos)
            key = key + key_pos

        sa_output = self.forward_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            feats_height=feats_height, feats_width=feats_width,**kwargs)[0]
    

        query = query + self.proj_drop(sa_output)

        return query

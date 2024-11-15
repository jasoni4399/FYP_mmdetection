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
                 content_width: OptConfigType = 0.4,
                 content_height: OptConfigType = 0.4) -> None:
        super().__init__(init_cfg=init_cfg,
                         layer_cfg = layer_cfg,
                         num_layers = num_layers,
                         post_norm_cfg = post_norm_cfg,
                         return_intermediate = return_intermediate)
        self._init_layers()
        self.content_width=content_width
        self.content_height=content_height

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
        self.ref_point_head = MLP(self.embed_dims, self.embed_dims, 2, 2)

        #self.lambda_q=MLP(self.embed_dims, self.embed_dims,
        #                       self.embed_dims, 2)
        self.ref_select=MLP(self.embed_dims, self.embed_dims,
                               2, 2)
        self.content_query=MLP(self.embed_dims+2, self.embed_dims,
                               self.embed_dims, 2)
        self.box_estimation=MLP(self.embed_dims, self.embed_dims,
                               self.embed_dims+2, 2)
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
        #V2 box query
        
        #lambda_q
        reference_unsigmoid = self.ref_point_head(key_pos)# [bs, num_keys, 2]
        reference_point = reference_unsigmoid.sigmoid()
        reference_point_xy = reference_point[..., :2]#x(Cx,Cy)

        #V2
        #s
        reference_point_select=self.ref_select(key_pos)
        reference_point_selection=reference_point_select[...,:2].contiguous()
        #reference_point_selection[reference_point_selection[0] != 1] = 0
        reference_point_selection = reference_point_selection.sigmoid()
        choose_top=torch.tensor([0.0, 1.0], device=reference_point_selection.device)#
        reference_point_selection[reference_point_selection[:,:,0] != 1] = choose_top
        
        reference_xy = reference_point_selection[..., :2]#x(Cx,Cy)

        #Cq initial by image content
        #query=self.content_query(reference_xy)
        #or
        content_w_h=torch.tensor([self.content_width,self.content_height],device=key_pos.device)
        content_w_h=content_w_h.unsqueeze(0).repeat(reference_xy.size(0),reference_xy.size(1),1)
        
        k=self.box_estimation(key_pos)
        pe=inverse_sigmoid(torch.cat([reference_xy, content_w_h],dim=2).permute(2,1,0)).permute(2,1,0)#
        print(k.size(),pe.size())
        query=self.content_query(coordinate_to_encoding(coord_tensor=k+pe[..., :256]).sigmoid())

        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(reference_point_xy) #lambda_q
            # get sine embedding for the query reference #Ps
            ref_sine_embed = coordinate_to_encoding(coord_tensor=reference_xy)
            # apply transformation
            ref_sine_embed = ref_sine_embed * pos_transformation
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
            return torch.stack(intermediate), reference_point_selection

        query = self.post_norm(query)
        return query.unsqueeze(0), reference_point_selection
    

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
                 content_width: OptConfigType = None,
                 content_height: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self._init_layers()
        self.content_width=content_width
        self.content_height=content_height

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            ConditionalDetrTransformerV2EncoderLayer(**self.layer_cfg)
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
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 content_width: OptConfigType = None,
                 content_height: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
        self.content_width=content_width
        self.content_height=content_height

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
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
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 cross_attn: bool = False,
                 keep_query_pos: bool = False,
                 batch_first: bool = True,
                 init_cfg: OptMultiConfig = None,
                 content_width: OptConfigType = None,
                 content_height: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)

        assert batch_first is True, 'Set `batch_first`\
        to False is NOT supported in ConditionalAttention. \
        First dimension of all DETRs in mmdet is `batch`, \
        please set `batch_first` to True.'

        self.cross_attn = cross_attn
        self.keep_query_pos = keep_query_pos
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn_drop = Dropout(attn_drop)
        self.proj_drop = Dropout(proj_drop)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers for qkv projection."""
        embed_dims = self.embed_dims
        self.qcontent_proj = Linear(embed_dims, embed_dims)
        self.qpos_proj = Linear(embed_dims, embed_dims)
        self.kcontent_proj = Linear(embed_dims, embed_dims)
        self.kpos_proj = Linear(embed_dims, embed_dims)
        self.v_proj = Linear(embed_dims, embed_dims)
        if self.cross_attn:
            self.qpos_sine_proj = Linear(embed_dims, embed_dims)
        self.out_proj = Linear(embed_dims, embed_dims)

        nn.init.constant_(self.out_proj.bias, 0.)

    def forward_attn(self,
                     query: Tensor,
                     key: Tensor,
                     value: Tensor,
                     attn_mask: Tensor = None,
                     key_padding_mask: Tensor = None) -> Tuple[Tensor]:
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
        assert key.size(1) == value.size(1), \
            f'{"key, value must have the same sequence length"}'
        assert query.size(0) == key.size(0) == value.size(0), \
            f'{"batch size must be equal for query, key, value"}'
        assert query.size(2) == key.size(2), \
            f'{"q_dims, k_dims must be equal"}'
        assert value.size(2) == self.embed_dims, \
            f'{"v_dims must be equal to embed_dims"}'
        #2, 300    , 256
        bs, tgt_len, hidden_dims = query.size()
        _, src_len, _ = key.size()
        head_dims = hidden_dims // self.num_heads
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

        proj_query = q.contiguous().view(bs, tgt_len, self.num_heads,
                                head_dims).permute(0, 2, 1, 3).flatten(0, 1)
        if k is not None:
            proj_key = k.contiguous().view(bs, src_len, self.num_heads,
                                    head_dims).permute(0, 2, 1,
                                                       3).flatten(0, 1)
        if v is not None:
            proj_value = v.contiguous().view(bs, src_len, self.num_heads,
                                    v_head_dims).permute(0, 2, 1,
                                                         3).flatten(0, 1)
            proj_value_H = v.permute(0,3,1,2).contiguous().view(bs, src_len, self.num_heads,
                                    head_dims)
            proj_value_W = v.permute(0,2,1,3).contiguous().view(bs, src_len, self.num_heads,
                                    head_dims)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bs
            assert key_padding_mask.size(1) == src_len
        # Q mul K.transpose
        #attn_output_weights = torch.bmm(q, k.transpose(1, 2)) 
        assert list(attn_output_weights.size()) == [
            bs * self.num_heads, tgt_len, src_len
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bs, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(
                bs * self.num_heads, tgt_len, src_len)
        
        def INF(B,H,W):
            return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        print(proj_query_H.size(),proj_query_W)
        proj_key_H=proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).sum(dim=0)/width
        proj_key_H=proj_key_H.expand(m_batchsize*width,-1,height)
        proj_key_W=proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).sum(dim=1)/height
        proj_key_W=proj_key_W.unsqueeze(-1).repeat(1,1,_).permute(0, 2, 1)
        #k=torch.cat((xj,xi),dim=2)
        #proj_key_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).flatten(0,1)
        #proj_key_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).flatten(0,1)
        #print(proj_key_H,proj_key_W)

        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.matmul(proj_query_H, proj_key_H)+INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.matmul(proj_query_W, proj_key_W).view(m_batchsize,height,width,width).permute(0,2,1,3)
        from torch.nn import Softmax
        concate = Softmax(dim=3)(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        print(att_H)
        print(att_W)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        print(out_H,out_W)
        out=torch.cat((out_H,out_W),dim=2)
       
        attn_output = attn_output.view(bs, self.num_heads, tgt_len,
                                       v_head_dims).permute(0, 2, 1,
                                                            3).flatten(2)
        attn_output = self.out_proj(attn_output)

        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bs, self.num_heads,
                                                       tgt_len, src_len)
        
        return attn_output, (attn_output_weights.sum(dim=1)) / self.num_heads

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor = None,
                ref_sine_embed: Tensor = None,
                key_pos: Tensor = None,
                attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                is_first: bool = False) -> Tensor:
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

        if self.cross_attn:
            q_content = self.qcontent_proj(query)
            k_content = self.kcontent_proj(key)
            v = self.v_proj(key)

            bs, nq, c = q_content.size()
            _, hw, _ = k_content.size()

            k_pos = self.kpos_proj(key_pos)
            if is_first or self.keep_query_pos:
                q_pos = self.qpos_proj(query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content
            q = q.view(bs, nq, self.num_heads, c // self.num_heads)
            query_sine_embed = self.qpos_sine_proj(ref_sine_embed)
            query_sine_embed = query_sine_embed.view(bs, nq, self.num_heads,
                                                     c // self.num_heads)
            q = torch.cat([q, query_sine_embed], dim=3).view(bs, nq, 2 * c)
            k = k.view(bs, hw, self.num_heads, c // self.num_heads)
            k_pos = k_pos.view(bs, hw, self.num_heads, c // self.num_heads)
            k = torch.cat([k, k_pos], dim=3).view(bs, hw, 2 * c)

            ca_output = self.forward_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)[0]
            query = query + self.proj_drop(ca_output)
        else:

            #self attention for encoder

            #b,_,H,W=query.size()
            q_content = self.qcontent_proj(query)
            q_pos = self.qpos_proj(query_pos)
            k_content = self.kcontent_proj(query)
            k_pos = self.kpos_proj(query_pos)
            v = self.v_proj(query)
            q = q_content if q_pos is None else q_content + q_pos
            k = k_content if k_pos is None else k_content + k_pos

            sa_output = self.forward_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)[0]
            
            query = query + self.proj_drop(sa_output)

        return query

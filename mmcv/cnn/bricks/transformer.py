import copy

import torch.nn as nn

from mmcv import ConfigDict
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.utils import build_from_cfg
from .registry import (ATTENTION, POSITIONAL_ENCODING, TRANSFORMERCODER,
                       TRANSFORMERLAYER)


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_transformerlayer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMERLAYER, default_args)


def build_transformercoder(cfg, default_args=None):
    """Builder for transformer coder."""
    return build_from_cfg(cfg, TRANSFORMERCODER, default_args)


class MultiheadAttention(nn.Module):
    """A warpper for torch.nn.MultiheadAttention.

    This module implements MultiheadAttention with residual connection,
    and positional encoding used in DETR is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        dropout (float): A Dropout layer on attn_output_weights. Default: 0.0.
    """

    def __init__(self, embed_dims, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        assert embed_dims % num_heads == 0, 'embed_dims must be ' \
            f'divisible by num_heads. got {embed_dims} and {num_heads}.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        Args:
            query (Tensor): The input query with shape [num_query, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            key (Tensor): The key tensor with shape [num_key, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
                Default None. If None, the `query` will be used.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Default None.
                If None, the `key` will be used.
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. Default None. If not None, it will
                be added to `x` before forward function.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Default None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (Tensor): ByteTensor mask with shape [num_query,
                num_key]. Same in `nn.MultiheadAttention.forward`.
                Default None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_key].
                Same in `nn.MultiheadAttention.forward`. Default None.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            residual = query
        if key_pos is None:
            if query_pos is not None and key is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out = self.attn(
            query,
            key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        return residual + self.dropout(out)


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: `ReLU`
        dropout (float, optional): Probability of an element to be
            zeroed. Default 0.0.
        add_residual (bool, optional): Add residual connection.
            Default: `True`.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.dropout = dropout
        self.activate = build_activation_layer(act_cfg)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)


class BaseTransformerLayer(nn.Module):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more customization,
    for example, using any number of `FFN or LN ` and use different kinds
    of `attention` by specifying a list of `ConfigDict` named `attn_cfgs`.
    It is worth mentioning that it supports `prenorm` when you specifying
    `norm` as the first element of `operation_order`. More details about
    the `prenorm`: `On Layer Normalization in the Transformer Architecture
    <https://arxiv.org/abs/2007.08103>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`. Default: None.
        feedforward_channels (int): The hidden dimension for FFNs.
            Default: None.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support prenorm when you specifying first element as `norm`.
            Default：None.
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
            self,
            attn_cfgs=None,
            feedforward_channels=None,
            ffn_dropout=0.0,
            operation_order=None,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN'),
            ffn_num_fcs=2,
    ):

        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order) & set(
            ['self_attn', 'norm', 'ffn', 'cross_attn']) == set(operation_order)
        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, ConfigDict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'
        assert 'embed_dims' in attn_cfgs[0]

        self.num_attn = num_attn
        self.feedforward_channels = feedforward_channels
        self.ffn_dropout = ffn_dropout
        self.operation_order = operation_order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.ffn_num_fcs = ffn_num_fcs
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.ModuleList()

        index = 0
        for operation in operation_order:
            if operation in ['self_attn', 'cross_attn']:
                attention = build_attention(attn_cfgs[index])
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        for _ in range(num_ffns):
            self.ffns.append(
                FFN(self.embed_dims, feedforward_channels, ffn_num_fcs,
                    act_cfg, ffn_dropout))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        Args:
            query (Tensor): Input query with the shape
                `(num_query, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Defaults: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_query]. Only used in `self_attn` layer.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        inp_residual = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    inp_residual if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                inp_residual = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    inp_residual if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                inp_residual = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, inp_residual if self.pre_norm else None)
                ffn_index += 1

        return query


class BaseTransformerCoder(nn.Module):
    """Base coder in vision transformer.

    As base-class of Encoder and Decoder in vision transformer.
    Support customization such as specifying different kind
    of `transformer_layer` in `transformer_coder`.

    Args:
        transformerlayer (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict`): Config of transformerlayer
            in TransformerCoder. If it is obj:`mmcv.ConfigDict`,
             it would be repeated `num_layer` times to a
             list[`mmcv.ConfigDict`]. Default: None.
        num_layers (int): The number of `TransformerLayer`. Default: None.
    """

    def __init__(self, transformerlayers=None, num_layers=None):
        super(BaseTransformerCoder, self).__init__()
        if isinstance(transformerlayers, ConfigDict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        operation_order = transformerlayers[0]['operation_order']
        self.pre_norm = operation_order[0] == 'norm'
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformerlayer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].operation_order[0] == 'norm'

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerCoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            attn_masks (List[Tensor], optional): Each element is 2D Tensor
                which is used in calculation of corresponding attention in
                operation_order. Defaults: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_query]. Only used in self-attention
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        return query

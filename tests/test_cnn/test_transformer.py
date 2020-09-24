from unittest.mock import patch

import torch

from mmcv.cnn.bricks import (FFN, MultiheadAttention, Transformer,
                             TransformerDecoder, TransformerDecoderLayer,
                             TransformerEncoder, TransformerEncoderLayer)


def _ffn_forward(self, x, residual=None):
    if residual is None:
        residual = x
    residual_str = residual.split('_')[-1]
    if '(residual' in residual_str:
        residual_str = residual_str.split('(residual')[0]
    return x + '_ffn(residual={})'.format(residual_str)


def _multihead_attention_forward(self,
                                 x,
                                 key=None,
                                 value=None,
                                 residual=None,
                                 query_pos=None,
                                 key_pos=None,
                                 attn_mask=None,
                                 key_padding_mask=None,
                                 selfattn=True):
    if residual is None:
        residual = x
    residual_str = residual.split('_')[-1]
    if '(residual' in residual_str:
        residual_str = residual_str.split('(residual')[0]
    attn_str = 'selfattn' if selfattn else 'multiheadattn'
    return x + '_{}(residual={})'.format(attn_str, residual_str)


def _encoder_layer_forward(self,
                           x,
                           pos=None,
                           attn_mask=None,
                           key_padding_mask=None):
    norm_cnt = 0
    inp_residual = x
    for layer in self.order:
        if layer == 'selfattn':
            x = self.self_attn(
                x,
                x,
                x,
                inp_residual if self.normalize_before else None,
                query_pos=pos,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)
            inp_residual = x
        elif layer == 'norm':
            x = x + '_norm{}'.format(norm_cnt)
            norm_cnt += 1
        elif layer == 'ffn':
            x = self.ffn(x, inp_residual if self.normalize_before else None)
        else:
            raise ValueError(f'Unsupported layer type {layer}.')
    return x


def _decoder_layer_forward(self,
                           x,
                           memory,
                           memory_pos=None,
                           query_pos=None,
                           memory_attn_mask=None,
                           tgt_attn_mask=None,
                           memory_key_padding_mask=None,
                           tgt_key_padding_mask=None):
    norm_cnt = 0
    inp_residual = x
    for layer in self.order:
        if layer == 'selfattn':
            x = self.self_attn(
                x,
                x,
                x,
                inp_residual if self.normalize_before else None,
                query_pos,
                attn_mask=tgt_attn_mask,
                key_padding_mask=tgt_key_padding_mask)
            inp_residual = x
        elif layer == 'norm':
            x = x + '_norm{}'.format(norm_cnt)
            norm_cnt += 1
        elif layer == 'multiheadattn':
            x = self.multihead_attn(
                x,
                memory,
                memory,
                inp_residual if self.normalize_before else None,
                query_pos,
                key_pos=memory_pos,
                attn_mask=memory_attn_mask,
                key_padding_mask=memory_key_padding_mask,
                selfattn=False)
            inp_residual = x
        elif layer == 'ffn':
            x = self.ffn(x, inp_residual if self.normalize_before else None)
        else:
            raise ValueError(f'Unsupported layer type {layer}.')
    return x


def test_multihead_attention(feat_channels=256,
                             num_heads=8,
                             dropout=0.1,
                             num_query=100,
                             num_key=1000,
                             batch_size=2):
    module = MultiheadAttention(feat_channels, num_heads, dropout)
    # self attention
    query = torch.rand(num_query, batch_size, feat_channels)
    out = module(query)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set key
    key = torch.rand(num_key, batch_size, feat_channels)
    out = module(query, key)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set residual
    residual = torch.rand(num_query, batch_size, feat_channels)
    out = module(query, key, key, residual)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set query_pos and key_pos
    query_pos = torch.rand(num_query, batch_size, feat_channels)
    key_pos = torch.rand(num_key, batch_size, feat_channels)
    out = module(query, key, None, residual, query_pos, key_pos)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set key_padding_mask
    key_padding_mask = torch.rand(batch_size, num_key) > 0.5
    out = module(query, key, None, residual, query_pos, key_pos, None,
                 key_padding_mask)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set attn_mask
    attn_mask = torch.rand(num_query, num_key) > 0.5
    out = module(query, key, key, residual, query_pos, key_pos, attn_mask,
                 key_padding_mask)
    assert out.shape == (num_query, batch_size, feat_channels)


def test_ffn(feat_channels=256,
             feedforward_channels=2048,
             num_fcs=2,
             batch_size=2):
    module = FFN(feat_channels, feedforward_channels, num_fcs)
    x = torch.rand(batch_size, feat_channels)
    out = module(x)
    assert out.shape == (batch_size, feat_channels)
    # set residual
    residual = torch.rand(batch_size, feat_channels)
    out = module(x, residual)
    assert out.shape == (batch_size, feat_channels)


def test_transformer_encoder_layer(feat_channels=256,
                                   num_heads=8,
                                   feedforward_channels=2048,
                                   num_key=1000,
                                   batch_size=2):
    module = TransformerEncoderLayer(feat_channels, num_heads,
                                     feedforward_channels)
    x = torch.rand(num_key, batch_size, feat_channels)
    key_padding_mask = torch.rand(batch_size, num_key) > 0.5
    out = module(x, key_padding_mask=key_padding_mask)
    assert not module.normalize_before
    assert out.shape == (num_key, batch_size, feat_channels)

    # set pos
    pos = torch.rand(num_key, batch_size, feat_channels)
    out = module(x, pos, key_padding_mask=key_padding_mask)
    assert out.shape == (num_key, batch_size, feat_channels)

    # set attn_mask
    attn_mask = torch.rand(num_key, num_key) > 0.5
    out = module(x, pos, attn_mask, key_padding_mask)
    assert out.shape == (num_key, batch_size, feat_channels)

    # set normalize_before
    order = ('norm', 'selfattn', 'norm', 'ffn')
    module = TransformerEncoderLayer(
        feat_channels, num_heads, feedforward_channels, order=order)
    assert module.normalize_before
    out = module(x, pos, attn_mask, key_padding_mask)
    assert out.shape == (num_key, batch_size, feat_channels)

    @patch('mmcv.cnn.bricks.TransformerEncoderLayer.forward',
           _encoder_layer_forward)
    @patch('mmcv.cnn.bricks.FFN.forward', _ffn_forward)
    @patch('mmcv.cnn.bricks.MultiheadAttention.forward',
           _multihead_attention_forward)
    def test_order():
        module = TransformerEncoderLayer(feat_channels)
        out = module('input')
        assert out == 'input_selfattn(residual=input)_norm0_ffn' \
            '(residual=norm0)_norm1'

        # normalize_before
        order = ('norm', 'selfattn', 'norm', 'ffn')
        module = TransformerEncoderLayer(feat_channels, order=order)
        out = module('input')
        assert out == 'input_norm0_selfattn(residual=input)_' \
            'norm1_ffn(residual=selfattn)'

    test_order()


def test_transformer_decoder_layer(feat_channels=256,
                                   num_heads=8,
                                   feedforward_channels=2048,
                                   num_key=1000,
                                   num_query=100,
                                   batch_size=2):
    module = TransformerDecoderLayer(feat_channels, num_heads,
                                     feedforward_channels)
    query = torch.rand(num_query, batch_size, feat_channels)
    memory = torch.rand(num_key, batch_size, feat_channels)
    assert not module.normalize_before
    out = module(query, memory)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set query_pos
    query_pos = torch.rand(num_query, batch_size, feat_channels)
    out = module(query, memory, memory_pos=None, query_pos=query_pos)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set memory_pos
    memory_pos = torch.rand(num_key, batch_size, feat_channels)
    out = module(query, memory, memory_pos, query_pos)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set memory_key_padding_mask
    memory_key_padding_mask = torch.rand(batch_size, num_key) > 0.5
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_key_padding_mask=memory_key_padding_mask)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set tgt_key_padding_mask
    tgt_key_padding_mask = torch.rand(batch_size, num_query) > 0.5
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_key_padding_mask=memory_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set memory_attn_mask
    memory_attn_mask = torch.rand(num_query, num_key)
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_attn_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask)
    assert out.shape == (num_query, batch_size, feat_channels)

    # set tgt_attn_mask
    tgt_attn_mask = torch.rand(num_query, num_query)
    out = module(query, memory, memory_pos, query_pos, memory_attn_mask,
                 tgt_attn_mask, memory_key_padding_mask, tgt_key_padding_mask)
    assert out.shape == (num_query, batch_size, feat_channels)

    # normalize_before
    order = ('norm', 'selfattn', 'norm', 'multiheadattn', 'norm', 'ffn')
    module = TransformerDecoderLayer(
        feat_channels, num_heads, feedforward_channels, order=order)
    assert module.normalize_before
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_attn_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask)
    assert out.shape == (num_query, batch_size, feat_channels)

    @patch('mmcv.cnn.bricks.TransformerDecoderLayer.forward',
           _decoder_layer_forward)
    @patch('mmcv.cnn.bricks.FFN.forward', _ffn_forward)
    @patch('mmcv.cnn.bricks.MultiheadAttention.forward',
           _multihead_attention_forward)
    def test_order():
        module = TransformerDecoderLayer(feat_channels, num_heads,
                                         feedforward_channels)
        out = module('input', 'memory')
        assert out == 'input_selfattn(residual=input)_norm0_multiheadattn' \
            '(residual=norm0)_norm1_ffn(residual=norm1)_norm2'

        # normalize_before
        order = ('norm', 'selfattn', 'norm', 'multiheadattn', 'norm', 'ffn')
        module = TransformerDecoderLayer(
            feat_channels, num_heads, feedforward_channels, order=order)
        out = module('input', 'memory')
        assert out == 'input_norm0_selfattn(residual=input)_norm1_' \
            'multiheadattn(residual=selfattn)_norm2_ffn(residual=' \
            'multiheadattn)'

    test_order()


def test_transformer_encoder(num_layers=4,
                             feat_channels=256,
                             num_heads=8,
                             num_key=1000,
                             batch_size=2):
    module = TransformerEncoder(num_layers, feat_channels, num_heads)
    assert not module.normalize_before
    assert module.norm is None
    x = torch.rand(num_key, batch_size, feat_channels)
    out = module(x)
    assert out.shape == (num_key, batch_size, feat_channels)

    # set pos
    pos = torch.rand(num_key, batch_size, feat_channels)
    out = module(x, pos)
    assert out.shape == (num_key, batch_size, feat_channels)

    # set key_padding_mask
    key_padding_mask = torch.rand(batch_size, num_key) > 0.5
    out = module(x, pos, None, key_padding_mask)
    assert out.shape == (num_key, batch_size, feat_channels)

    # set attn_mask
    attn_mask = torch.rand(num_key, num_key) > 0.5
    out = module(x, pos, attn_mask, key_padding_mask)
    assert out.shape == (num_key, batch_size, feat_channels)

    # normalize_before
    order = ('norm', 'selfattn', 'norm', 'ffn')
    module = TransformerEncoder(
        num_layers, feat_channels, num_heads, order=order)
    assert module.normalize_before
    assert module.norm is not None
    out = module(x, pos, attn_mask, key_padding_mask)
    assert out.shape == (num_key, batch_size, feat_channels)


def test_transformer_decoder(num_layers=3,
                             feat_channels=256,
                             num_heads=8,
                             num_key=1000,
                             num_query=100,
                             batch_size=2):
    module = TransformerDecoder(num_layers, feat_channels, num_heads)
    query = torch.rand(num_query, batch_size, feat_channels)
    memory = torch.rand(num_key, batch_size, feat_channels)
    out = module(query, memory)
    assert out.shape == (1, num_query, batch_size, feat_channels)

    # set query_pos
    query_pos = torch.rand(num_query, batch_size, feat_channels)
    out = module(query, memory, query_pos=query_pos)
    assert out.shape == (1, num_query, batch_size, feat_channels)

    # set memory_pos
    memory_pos = torch.rand(num_key, batch_size, feat_channels)
    out = module(query, memory, memory_pos, query_pos)
    assert out.shape == (1, num_query, batch_size, feat_channels)

    # set memory_key_padding_mask
    memory_key_padding_mask = torch.rand(batch_size, num_key) > 0.5
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_key_padding_mask=memory_key_padding_mask)
    assert out.shape == (1, num_query, batch_size, feat_channels)

    # set tgt_key_padding_mask
    tgt_key_padding_mask = torch.rand(batch_size, num_query) > 0.5
    out = module(
        query,
        memory,
        memory_pos,
        query_pos,
        memory_key_padding_mask=memory_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask)
    assert out.shape == (1, num_query, batch_size, feat_channels)

    # set memory_attn_mask
    memory_attn_mask = torch.rand(num_query, num_key) > 0.5
    out = module(query, memory, memory_pos, query_pos, memory_attn_mask, None,
                 memory_key_padding_mask, tgt_key_padding_mask)
    assert out.shape == (1, num_query, batch_size, feat_channels)

    # set tgt_attn_mask
    tgt_attn_mask = torch.rand(num_query, num_query) > 0.5
    out = module(query, memory, memory_pos, query_pos, memory_attn_mask,
                 tgt_attn_mask, memory_key_padding_mask, tgt_key_padding_mask)
    assert out.shape == (1, num_query, batch_size, feat_channels)

    # normalize_before
    order = ('norm', 'selfattn', 'norm', 'multiheadattn', 'norm', 'ffn')
    module = TransformerDecoder(
        num_layers, feat_channels, num_heads, order=order)
    out = module(query, memory, memory_pos, query_pos, memory_attn_mask,
                 tgt_attn_mask, memory_key_padding_mask, tgt_key_padding_mask)
    assert out.shape == (1, num_query, batch_size, feat_channels)

    # return_intermediate
    module = TransformerDecoder(
        num_layers,
        feat_channels,
        num_heads,
        order=order,
        return_intermediate=True)
    out = module(query, memory, memory_pos, query_pos, memory_attn_mask,
                 tgt_attn_mask, memory_key_padding_mask, tgt_key_padding_mask)
    assert out.shape == (num_layers, num_query, batch_size, feat_channels)


def test_transformer(num_enc_layers=2,
                     num_dec_layers=3,
                     feat_channels=256,
                     num_heads=4,
                     num_query=100,
                     batch_size=2):
    module = Transformer(feat_channels, num_heads, num_enc_layers,
                         num_dec_layers)
    height, width = 80, 60
    x = torch.rand(batch_size, feat_channels, height, width)
    mask = torch.rand(batch_size, height, width) > 0.5
    query_embed = torch.rand(num_query, feat_channels)
    pos_embed = torch.rand(batch_size, feat_channels, height, width)
    hs, mem = module(x, mask, query_embed, pos_embed)
    assert hs.shape == (1, batch_size, num_query, feat_channels)
    assert mem.shape == (batch_size, feat_channels, height, width)

    # normalize_before
    module = Transformer(
        feat_channels,
        num_heads,
        num_enc_layers,
        num_dec_layers,
        normalize_before=True)
    hs, mem = module(x, mask, query_embed, pos_embed)
    assert hs.shape == (1, batch_size, num_query, feat_channels)
    assert mem.shape == (batch_size, feat_channels, height, width)

    # return_intermediate
    module = Transformer(
        feat_channels,
        num_heads,
        num_enc_layers,
        num_dec_layers,
        return_intermediate_dec=True)
    hs, mem = module(x, mask, query_embed, pos_embed)
    assert hs.shape == (num_dec_layers, batch_size, num_query, feat_channels)
    assert mem.shape == (batch_size, feat_channels, height, width)

    # normalize_before and return_intermediate
    module = Transformer(
        feat_channels,
        num_heads,
        num_enc_layers,
        num_dec_layers,
        normalize_before=True,
        return_intermediate_dec=True)
    hs, mem = module(x, mask, query_embed, pos_embed)
    assert hs.shape == (num_dec_layers, batch_size, num_query, feat_channels)
    assert mem.shape == (batch_size, feat_channels, height, width)


# test_multihead_attention()
# test_ffn()
# test_transformer_encoder_layer()
# test_transformer_decoder_layer()
# test_transformer_encoder()
# test_transformer_decoder()
# test_transformer()

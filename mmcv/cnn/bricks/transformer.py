import torch
import torch.nn as nn

from .activation import build_activation_layer
from .norm import build_norm_layer
from .wrappers import Linear


class MultiheadAttention(nn.Module):

    def __init__(self, feat_channels, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(feat_channels, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None):
        query = x
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            residual = x
        if key_pos is None:
            if query_pos is not None and key is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query += query_pos
        if key_pos is not None:
            key += key_pos
        out = self.attn(
            query,
            key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]
        return residual + self.dropout(out)


class FFN(nn.Module):

    def __init__(self,
                 feat_channels,
                 feedforward_channels=2048,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.1):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        # NOTE only when act_cfg['type'] not in ['Tanh', 'PReLU',
        # 'Sigmoid', 'HSigmoid', 'Swish'], can we set inplace in act_cfg.
        self.activate = build_activation_layer(act_cfg)

        layers = nn.ModuleList()
        in_channels = feat_channels
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, feat_channels))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = self.layers(x)
        return residual + self.dropout(x)


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 feat_channels,
                 num_heads=8,
                 feedforward_channels=2048,
                 dropout=0.1,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_ffn=2):
        super(TransformerEncoderLayer, self).__init__()
        self.order = order  # or order ('norm', 'selfattn', 'norm', 'ffn')
        self.normalize_before = order[0] == 'norm'
        self.self_attn = MultiheadAttention(feat_channels, num_heads, dropout)
        self.ffn = FFN(feat_channels, feedforward_channels, num_ffn, act_cfg,
                       dropout)
        self.norms = nn.ModuleList()
        self.norms.append(build_norm_layer(norm_cfg, feat_channels)[1])
        self.norms.append(build_norm_layer(norm_cfg, feat_channels)[1])

    def forward(self, x, pos=None, attn_mask=None, key_padding_mask=None):
        norm_cnt = 0
        inp_residual = x
        for layer in self.order:
            if layer == 'selfattn':
                # self attention
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
                x = self.norms[norm_cnt](x)
                norm_cnt += 1
            elif layer == 'ffn':
                x = self.ffn(x,
                             inp_residual if self.normalize_before else None)
            else:
                raise ValueError(f'Unsupported layer type {layer}.')
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 feat_channels,
                 num_heads=8,
                 feedforward_channels=2048,
                 dropout=0.1,
                 order=('selfattn', 'norm', 'multiheadattn', 'norm', 'ffn',
                        'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_ffn=2):
        super(TransformerDecoderLayer, self).__init__()
        # or order ('norm', 'selfattn', 'norm', 'multiheadattn', 'norm', 'ffn')
        self.order = order
        self.normalize_before = order[0] == 'norm'
        self.self_attn = MultiheadAttention(feat_channels, num_heads, dropout)
        self.multihead_attn = MultiheadAttention(feat_channels, num_heads,
                                                 dropout)
        self.ffn = FFN(feat_channels, feedforward_channels, num_ffn, act_cfg,
                       dropout)
        self.norms = nn.ModuleList()
        # 3 norm layers in official DETR's TransformerDecoderLayer
        for _ in range(3):
            self.norms.append(build_norm_layer(norm_cfg, feat_channels)[1])

    def forward(self,
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
                x = self.norms[norm_cnt](x)
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
                    key_padding_mask=memory_key_padding_mask)
                inp_residual = x
            elif layer == 'ffn':
                x = self.ffn(x,
                             inp_residual if self.normalize_before else None)
            else:
                raise ValueError(f'Unsupported layer type {layer}.')
        return x


class TransformerEncoder(nn.Module):

    def __init__(self,
                 num_layers,
                 feat_channels,
                 num_heads,
                 feedforward_channels=2048,
                 dropout=0.1,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_ffn=2):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.normalize_before = order[0] == 'norm'
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(feat_channels, num_heads,
                                        feedforward_channels, dropout, order,
                                        act_cfg, norm_cfg, num_ffn))
        self.norm = build_norm_layer(
            norm_cfg, feat_channels)[1] if self.normalize_before else None

    def forward(self, x, pos=None, attn_mask=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, pos, attn_mask, key_padding_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self,
                 num_layers,
                 feat_channels,
                 num_heads,
                 feedforward_channels=2048,
                 dropout=0.1,
                 order=('selfattn', 'norm', 'multiheadattn', 'norm', 'ffn',
                        'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_ffn=2,
                 return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerDecoderLayer(feat_channels, num_heads,
                                        feedforward_channels, dropout, order,
                                        act_cfg, norm_cfg, num_ffn))
        self.norm = build_norm_layer(norm_cfg, feat_channels)[1]

    def forward(self,
                x,
                memory,
                memory_pos=None,
                query_pos=None,
                memory_attn_mask=None,
                tgt_attn_mask=None,
                memory_key_padding_mask=None,
                tgt_key_padding_mask=None):
        intermediate = []
        for layer in self.layers:
            x = layer(x, memory, memory_pos, query_pos, memory_attn_mask,
                      tgt_attn_mask, memory_key_padding_mask,
                      tgt_key_padding_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(x))
        if self.norm is not None:
            x = self.norm(x)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(x)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return x.unsqueeze(0)


class Transformer(nn.Module):

    def __init__(self,
                 feat_channels=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 feedforward_channels=2048,
                 dropout=0.1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_ffn=2,
                 normalize_before=False,
                 return_intermediate_dec=False):
        super(Transformer, self).__init__()
        self.normalize_before = normalize_before
        if self.normalize_before:
            encoder_order = ('norm', 'selfattn', 'norm', 'ffn')
            decoder_order = ('norm', 'selfattn', 'norm', 'multiheadattn',
                             'norm', 'ffn')
        else:
            encoder_order = ('selfattn', 'norm', 'ffn', 'norm')
            decoder_order = ('selfattn', 'norm', 'multiheadattn', 'norm',
                             'ffn', 'norm')
        self.encoder = TransformerEncoder(num_encoder_layers, feat_channels,
                                          num_heads, feedforward_channels,
                                          dropout, encoder_order, act_cfg,
                                          norm_cfg, num_ffn)
        self.decoder = TransformerDecoder(num_decoder_layers, feat_channels,
                                          num_heads, feedforward_channels,
                                          dropout, decoder_order, act_cfg,
                                          norm_cfg, num_ffn,
                                          return_intermediate_dec)
        # TODO init parameters.

    def forward(self, x, mask, query_embed, pos_embed):
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [bs,c,h,w] -> [h*w,bs,c]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query,dim] -> [num_query,bs,dim]
        mask = mask.flatten(1)  # [bs, h,w] -> [bs, h*w]
        memory = self.encoder(
            x, pos=pos_embed, attn_mask=None, key_padding_mask=mask)
        tgt = torch.zeros_like(query_embed)
        # hs: [num_layers,num_query,bs,dim]
        hs = self.decoder(
            tgt,
            memory,
            memory_pos=pos_embed,
            query_pos=query_embed,
            memory_attn_mask=None,
            tgt_attn_mask=None,
            memory_key_padding_mask=mask,
            tgt_key_padding_mask=None)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

import torch.nn as nn

from .activation import build_activation_layer
from .norm import build_norm_layer
from .wrappers import Linear


class SelfAttentionModule(nn.Module):

    def __init__(self, dim_feat, nhead, dropout=0.1):
        super(SelfAttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(dim_feat, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self,
                x,
                residual=None,
                pos=None,
                attn_mask=None,
                key_padding_mask=None):
        if residual is None:
            residual = x
        q = k = x
        if pos is not None:
            q += pos
            k += pos
        out = self.attn(
            q,
            k,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]
        return residual + self.dropout1(out)


class FFN(nn.Module):

    def __init__(self,
                 dim_feat,
                 dim_feedforward=2048,
                 nrepeat=2,
                 act_cfg=dict(type='ReLU'),
                 dropout=0.1,
                 inplace=True):
        super(FFN, self).__init__()
        act_cfg_ = act_cfg.copy()
        if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
        ]:
            act_cfg_.setdefault('inplace', inplace)
        self.activate = build_activation_layer(act_cfg_)

        layers = nn.ModuleList()
        dim_inp = dim_feat
        for _ in range(nrepeat - 1):
            layers.append(
                nn.Sequential(
                    Linear(dim_inp, dim_feedforward), self.activate,
                    nn.Dropout(dropout)))
            dim_inp = dim_feedforward
        layers.append(Linear(dim_feedforward, dim_feat))
        self.layers = nn.Sequential(*layers)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = self.layers(x)
        return residual + self.dropout2(x)


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 dim_feat,
                 nhead=8,
                 dim_feedforward=2048,
                 num_ffn=2,
                 dropout=0.1,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 inplace=True,
                 order=('selfattn', 'norm', 'ffn', 'norm')):
        super(TransformerEncoderLayer, self).__init__()
        self.order = order  # or order ('norm', 'selfattn', 'norm', 'ffn')
        self.normalize_before = order[0] == 'norm'
        self.self_attn = SelfAttentionModule(dim_feat, nhead, dropout)
        self.ffn = FFN(dim_feat, dim_feedforward, num_ffn, act_cfg, dropout,
                       inplace)
        self.norms = nn.ModuleList()
        self.norms.append(build_norm_layer(norm_cfg, dim_feat)[1])
        self.norms.append(build_norm_layer(norm_cfg, dim_feat)[1])

    def forward(self, x, pos=None, attn_mask=None, key_padding_mask=None):
        norm_cnt = 0
        inp_residual = x
        for layer in self.order:
            if layer == 'selfattn':
                x = self.self_attn(
                    x, inp_residual if self.normalize_before else None, pos,
                    attn_mask, key_padding_mask)
                inp_residual = x
            elif layer == 'norm':
                x = self.norms[norm_cnt](x)
                norm_cnt += 1
            elif layer == 'ffn':
                x = self.ffn(x,
                             inp_residual if self.normalize_before else None)
        return x


# class Transformer(nn.Module):
#     def __init__(self, ):
#         super(Transformer, self).__init__()

#     def forward(self, hid_dim, nhead=8, num_encoder_laters=6,
#                   num_decoder_layers=6, dim_feedforward=2048,
#                   dropout=0.1, activation='relu'):
#         for layer in self.order:
#             if layer == 'selfattn':
#                 x = self.self_attn(x, None, pos, attn_mask, key_padding_mask)
#             elif layer == 'norm':
#                 x = self.norm1(x)
#             elif layer == 'ffn':
#                 pass
#             pass
#         pass

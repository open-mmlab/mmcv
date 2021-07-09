import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def NEG_INF_DIAG(n, device):
    """Returns a diagonal matrix of size [n, n].

    The diagonal are all "-inf". This is for avoiding calculating the
    overlapped element in the Criss-Cross twice.
    """
    return torch.diag(torch.tensor(float('-inf')).to(device).repeat(n), 0)


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        query = self.query_conv(x)
        query_H = rearrange(query, 'B C H W -> (B W) H C')
        query_W = rearrange(query, 'B C H W -> (B H) W C')
        key = self.key_conv(x)
        key_H = rearrange(key, 'B C H W -> (B W) C H')
        key_W = rearrange(key, 'B C H W -> (B H) C W')
        value = self.value_conv(x)
        value_H = rearrange(value, 'B C H W -> (B W) C H')
        value_W = rearrange(value, 'B C H W -> (B H) C W')

        energy_H = torch.bmm(query_H, key_H) + NEG_INF_DIAG(H, query_H.device)
        energy_H = rearrange(energy_H, '(B W) H H2 -> B H W H2', B=B)
        energy_W = torch.bmm(query_W, key_W)
        energy_W = rearrange(energy_W, '(B H) W W2 -> B H W W2', B=B)
        attn = F.softmax(
            torch.cat([energy_H, energy_W], dim=-1), dim=-1)  # [B,H,W,(H+W)]

        att_H = rearrange(attn[..., :H], 'B H W H2 -> (B W) H2 H')
        att_W = rearrange(attn[..., H:], 'B H W W2 -> (B H) W2 W')

        out = rearrange(torch.bmm(value_H, att_H), '(B W) C H -> B C H W', B=B)
        out += rearrange(
            torch.bmm(value_W, att_W), '(B H) C W -> B C H W', B=B)
        out *= self.gamma
        out += x

        return out

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels})'
        return s

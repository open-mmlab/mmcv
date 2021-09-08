# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


def coords_grid(batch, xx, yy):
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()

    return coords[None].repeat(batch, 1, 1, 1)  # shape(batch, 2, H, W)


def bilinear_sample(feat,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False,
                    scale=True):
    H, W = feat.shape[-2:]
    if grid.shape[-1] != 2:
        grid = grid.permute(0, 2, 3, 1)
    if scale:
        grid[:, :, :, 0] = grid[:, :, :, 0] * 2. / max(W - 1, 1) - 1.
        grid[:, :, :, 1] = grid[:, :, :, 1] * 2. / max(H - 1, 1) - 1.

    return F.grid_sample(feat, grid, mode, padding_mode, align_corners)


class CorrLookup(nn.Module):

    def __init__(self,
                 radius=4,
                 mode='bilinear',
                 padding_mode='zeros',
                 align_corners=False):
        super().__init__()
        self.r = radius
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, corr_pyramid, flow):
        B, _, H, W = flow.shape
        xx = torch.arange(0, W, device=flow.device)
        yy = torch.arange(0, H, device=flow.device)
        grid = coords_grid(B, xx, yy) + flow  # shape N, 2, H, W
        grid = grid.permute(0, 2, 3, 1)  # shape N, H, W, 2

        dx = torch.linspace(
            -self.r, self.r, 2 * self.r + 1, device=flow.device)
        dy = torch.linspace(
            -self.r, self.r, 2 * self.r + 1, device=flow.device)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
        delta_lvl = delta.view(1, 2 * self.r + 1, 2 * self.r + 1, 2)

        out_corr_pyramid = []
        for i, corr in enumerate(corr_pyramid):
            centroid_lvl = grid.reshape(B * H * W, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sample(corr, coords_lvl, self.mode,
                                   self.padding_mode, self.align_corners)
            corr = corr.view(B, H, W, -1)
            out_corr_pyramid.append(corr)

        out = torch.cat(out_corr_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

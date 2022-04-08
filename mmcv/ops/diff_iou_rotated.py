# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/lilanxiao/Rotated_IoU/blob/master/box_intersection_2d.py  # noqa
# Adapted from https://github.com/lilanxiao/Rotated_IoU/blob/master/oriented_iou_loss.py  # noqa
import torch
from torch.autograd import Function

from ..utils import ext_loader

EPSILON = 1e-8
ext_module = ext_loader.load_ext('_ext',
                                 ['diff_iou_rotated_sort_vertices_forward'])


class SortVertices(Function):

    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        idx = ext_module.diff_iou_rotated_sort_vertices_forward(
            vertices, mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, gradout):
        return ()


def box_intersection_th(corners1: torch.Tensor, corners2: torch.Tensor):
    """Find intersection points of rectangles.
    Convention: if two edges are collinear, there is no intersection point

    Args:
        corners1 (torch.Tensor): B, N, 4, 2
        corners2 (torch.Tensor): B, N, 4, 2

    Returns:
        intersections (torch.Tensor): B, N, 4, 4, 2
        mask (torch.Tensor) : B, N, 4, 4; bool
    """
    # build edges from corners
    # B, N, 4, 4: Batch, Box, edge, point
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3)
    line2_ext = line2.unsqueeze(2)
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = den_t / num
    t[num == .0] = -1.
    mask_t = (t > 0) & (t < 1)  # intersection on line segment 1
    den_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = -den_u / num
    u[num == .0] = -1.
    mask_u = (u > 0) & (u < 1)  # intersection on line segment 2
    mask = mask_t * mask_u
    # overwrite with EPSILON. otherwise numerically unstable
    t = den_t / (num + EPSILON)
    intersections = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)],
                                dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def box1_in_box2(corners1: torch.Tensor, corners2: torch.Tensor):
    """Check if corners of box1 lie in box2.
    Convention: if a corner is exactly on the edge of the other box,
    it's also a valid point.

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool
    """
    a = corners2[:, :, 0:1, :]  # (B, N, 1, 2)
    b = corners2[:, :, 1:2, :]  # (B, N, 1, 2)
    d = corners2[:, :, 3:4, :]  # (B, N, 1, 2)
    ab = b - a  # (B, N, 1, 2)
    am = corners1 - a  # (B, N, 4, 2)
    ad = d - a  # (B, N, 1, 2)
    p_ab = torch.sum(ab * am, dim=-1)  # (B, N, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)  # (B, N, 1)
    p_ad = torch.sum(ad * am, dim=-1)  # (B, N, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)  # (B, N, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes
    # are exactly the same also stable with different scale of bboxes
    cond1 = (p_ab / norm_ab > -1e-6) * (p_ab / norm_ab < 1 + 1e-6)  # (B, N, 4)
    cond2 = (p_ad / norm_ad > -1e-6) * (p_ad / norm_ad < 1 + 1e-6)  # (B, N, 4)
    return cond1 * cond2


def box_in_box_th(corners1: torch.Tensor, corners2: torch.Tensor):
    """Check if corners of two boxes lie in each other.

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool. i-th corner of box1 in box2
        c2_in_1: (B, N, 4) Bool. i-th corner of box2 in box1
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1


def build_vertices(corners1: torch.Tensor, corners2: torch.Tensor,
                   c1_in_2: torch.Tensor, c2_in_1: torch.Tensor,
                   inters: torch.Tensor, mask_inter: torch.Tensor):
    """Find vertices of intersection area.

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)
        c1_in_2 (torch.Tensor): Bool, (B, N, 4)
        c2_in_1 (torch.Tensor): Bool, (B, N, 4)
        inters (torch.Tensor): (B, N, 4, 4, 2)
        mask_inter (torch.Tensor): (B, N, 4, 4)

    Returns:
        vertices (torch.Tensor): (B, N, 24, 2) vertices of intersection area;
            only some elements are valid
        mask (torch.Tensor): (B, N, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient
    # (masked by multiplying with 0); can be used as trick
    B = corners1.size()[0]
    N = corners1.size()[1]
    # (B, N, 4 + 4 + 16, 2)
    vertices = torch.cat(
        [corners1, corners2, inters.view([B, N, -1, 2])], dim=2)
    # Bool (B, N, 4 + 4 + 16)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([B, N, -1])], dim=2)
    return vertices, mask


def sort_indices(vertices: torch.Tensor, mask: torch.Tensor):
    """Sort indices.

    Args:
        vertices (torch.Tensor): float (B, N, 24, 2)
        mask (torch.Tensor): bool (B, N, 24)

    Returns:
        sorted_index: bool (B, N, 9)

    Note:
        why 9? the polygon has maximal 8 vertices.
        +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X)
        and X indicates the index of arbitrary elements in the last
        16 (intersections not corners) with
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    num_valid = torch.sum(mask.int(), dim=2).int()  # (B, N)
    mean = torch.sum(
        vertices * mask.float().unsqueeze(-1), dim=2,
        keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean  # normalization makes sorting easier
    return SortVertices.apply(vertices_normalized, mask, num_valid).long()


def calculate_area(idx_sorted: torch.Tensor, vertices: torch.Tensor):
    """Calculate area of intersection.

    Args:
        idx_sorted (torch.Tensor): (B, N, 9)
        vertices (torch.Tensor): (B, N, 24, 2)

    return:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1, 1, 1, 2])
    selected = torch.gather(vertices, 2, idx_ext)
    total = selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1] \
        - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(corners1: torch.Tensor,
                                 corners2: torch.Tensor):
    """Calculate intersection area of 2d rectangles.

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, N, 4, 2)

    Returns:
        area: (B, N), area of intersection
        selected: (B, N, 9, 2), vertices of polygon with zero padding
    """
    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters,
                                    mask_inter)
    sorted_indices = sort_indices(vertices, mask)
    return calculate_area(sorted_indices, vertices)


def box2corners_th(box: torch.Tensor):
    """Convert box coordinate to corners.

    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5]  # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).to(box.device)
    x4 = x4 * w  # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).to(box.device)
    y4 = y4 * h  # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)  # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1, 4, 2]), rot_T.view([-1, 2, 2]))
    rotated = rotated.view([B, -1, 4, 2])  # (B * N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def diff_iou_rotated_2d(box1: torch.Tensor, box2: torch.Tensor):
    """Calculate differentiable iou of 2d boxes.
    Args:
        box1 (torch.Tensor): (B, N, 5)
        box2 (torch.Tensor): (B, N, 5)

    Returns:
        iou (torch.Tensor): (B, N)
    """
    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)  # (B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou


def diff_iou_rotated_3d(box3d1: torch.Tensor, box3d2: torch.Tensor):
    """Calculate differentiable iou of 3d boxes.

    Args:
        box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)

    Returns:
        iou (torch.Tensor): (B, N)
    """
    box1 = box3d1[..., [0, 1, 3, 4, 6]]  # 2d box
    box2 = box3d2[..., [0, 1, 3, 4, 6]]
    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)
    zmax1 = box3d1[..., 2] + box3d1[..., 5] * 0.5
    zmin1 = box3d1[..., 2] - box3d1[..., 5] * 0.5
    zmax2 = box3d2[..., 2] + box3d2[..., 5] * 0.5
    zmin2 = box3d2[..., 2] - box3d2[..., 5] * 0.5
    z_overlap = (torch.min(zmax1, zmax2) -
                 torch.max(zmin1, zmin2)).clamp_min(0.)
    intersection_3d = inter_area * z_overlap
    v1 = box3d1[..., 3] * box3d1[..., 4] * box3d1[..., 5]
    v2 = box3d2[..., 3] * box3d2[..., 4] * box3d2[..., 5]
    u3d = v1 + v2 - intersection_3d
    return intersection_3d / u3d

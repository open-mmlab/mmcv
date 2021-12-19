import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['points_in_polygons_forward'])


def points_in_polygons(points, polygons):
    assert points.shape[1] == 2, \
        'points dimension should be 2, ' \
        f'but got unexpected shape {points.shape[1]}'
    assert polygons.shape[1] == 8, \
        'polygons dimension should be 8, ' \
        f'but got unexpected shape {polygons.shape[1]}'
    output = torch.full([points.shape[0], polygons.shape[0]],
                        0.).cuda().float()
    ext_module.points_in_boxes_part_forward(points.contiguous(),
                                            polygons.contiguous(), output)
    return output

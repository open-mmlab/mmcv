import pytest
import torch

from mmcv.ops import convex_iou

# pointsets = torch.tensor([[1.0,1.0, 2.0,2.0, 1.0,2.0, 2.0,1.0,
#                         1.0,3.0, 3.0,1.0, 2.0,3.0, 3.0,2.0,
#                         1.5,1.5]], dtype=torch.float, device='cuda')

# polygons = torch.tensor([[1.0,1.0, 1.0,2.0, 2.0,2.0, 2.0,1.0],
#                        [1.0,1.0, 1.0,3.0, 3.0,3.0, 3.0,1.0]],
#                        dtype=torch.float, device='cuda')

# expected_iou = torch.tensor([[0.2857, 0.8750]], dtype=torch.float,
# device='cuda')

# assert torch.allclose(convex_iou(pointsets, polygons), expected_iou,
# atol=1e-3)

# print(convex_iou(pointsets, polygons))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_convex_iou():
    pointsets = torch.tensor([[
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 1.0, 2.0, 3.0,
        3.0, 2.0, 1.5, 1.5
    ]],
                             dtype=torch.float,
                             device='cuda')
    polygons = torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0],
                             [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0]],
                            dtype=torch.float,
                            device='cuda')
    expected_iou = torch.tensor([[0.2857, 0.8750]],
                                dtype=torch.float,
                                device='cuda')
    assert torch.allclose(
        convex_iou(pointsets, polygons), expected_iou, atol=1e-3)

import os
import sys

# Add the package directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    # Test one of our PyTorch implementations
    import torch
    from mmcv.ops.pure_pytorch_nms import nms_pytorch
    
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    
    keep = nms_pytorch(boxes, scores, 0.5)
    print(f'NMS results: {keep}')
    print('PyTorch-only implementation works!')
    
    # Test imported ops
    from mmcv.ops.nms import nms
    keep2 = nms(boxes, scores, 0.5)
    print(f'Imported NMS results: {keep2}')
    print('All implementations work!')
except Exception as e:
    print(f'Error: {e}')

# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.ops import filtered_lrelu

x = torch.randn((1, 3, 8, 8)).cuda()
y = filtered_lrelu(x)

print(y)

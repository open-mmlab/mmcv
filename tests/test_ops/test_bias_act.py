# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import bias_act

a = torch.randn((2, 2)).cuda()
b = torch.randn(2).cuda()

c = bias_act(a, b)

print(c)
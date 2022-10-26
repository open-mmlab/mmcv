# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.ops import bias_act

a = torch.randn((1, 3)).cuda()
b = torch.randn(3).cuda()

c = bias_act(a, b)
print(c)

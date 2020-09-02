import torch

from mmcv.cnn.bricks import Conv2dSamePadding


def test_conv2d_samepadding():
    # test Conv2dSamePadding with stride=1
    inputs = torch.rand((1, 3, 28, 28))
    conv = Conv2dSamePadding(3, 3, kernel_size=3, stride=1)
    output = conv(inputs)
    assert output.shape == inputs.shape

    inputs = torch.rand((1, 3, 13, 13))
    conv = Conv2dSamePadding(3, 3, kernel_size=3, stride=1)
    output = conv(inputs)
    assert output.shape == inputs.shape

    # test Conv2dSamePadding with stride=2
    inputs = torch.rand((1, 3, 28, 28))
    conv = Conv2dSamePadding(3, 3, kernel_size=3, stride=2)
    output = conv(inputs)
    assert output.shape == torch.Size([1, 3, 14, 14])

    inputs = torch.rand((1, 3, 13, 13))
    conv = Conv2dSamePadding(3, 3, kernel_size=3, stride=2)
    output = conv(inputs)
    assert output.shape == torch.Size([1, 3, 7, 7])

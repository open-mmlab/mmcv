import numpy as np
import torch
import torch.nn as nn


class Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        return torch.mean(input - target)


class TestCrissCrossAttention(object):

    def test_cc_attention(self):
        if not torch.cuda.is_available():
            return

        from mmcv.ops import CrissCrossAttention
        loss_func = Loss()

        input = np.fromfile(
            'tests/data/for_ccattention/ccattention_input.bin',
            dtype=np.float32)
        output = np.fromfile(
            'tests/data/for_ccattention/ccattention_output.bin',
            dtype=np.float32)
        input = input.reshape((1, 32, 45, 45))
        output = output.reshape((1, 32, 45, 45))
        label = torch.ones((1, 32, 45, 45))

        input = torch.FloatTensor(input)
        output = torch.FloatTensor(output)

        input.requires_grad = True

        shape = input.shape
        channel = shape[1]

        cca = CrissCrossAttention(channel)
        cca.cuda()
        input = input.cuda()
        label = label.cuda()
        cca.train()
        test_output = cca(input)
        test_loss = loss_func(test_output, label)
        test_loss.backward()
        test_output = test_output.detach().cpu().numpy()
        output = output.numpy()

        assert np.allclose(test_output, output)
        assert test_output.shape == shape

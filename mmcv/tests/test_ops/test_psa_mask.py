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


class TestPSAMask(object):

    def test_psa_mask_collect(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import PSAMask
        test_loss = Loss()

        input = np.fromfile(
            'tests/data/for_psa_mask/psa_input.bin', dtype=np.float32)
        output_collect = np.fromfile(
            'tests/data/for_psa_mask/psa_output_collect.bin', dtype=np.float32)

        input = input.reshape((4, 16, 8, 8))
        output_collect = output_collect.reshape((4, 64, 8, 8))
        label = torch.ones((4, 64, 8, 8))

        input = torch.FloatTensor(input)
        input.requires_grad = True

        psamask_collect = PSAMask('collect', (4, 4))

        # test collect cpu
        test_output = psamask_collect(input)
        loss = test_loss(test_output, label)
        loss.backward()
        test_output = test_output.detach().numpy()
        assert np.allclose(test_output, output_collect)
        assert test_output.shape == output_collect.shape

        psamask_collect.cuda()
        input = input.cuda()
        label = label.cuda()

        # test collect cuda
        test_output = psamask_collect(input)
        loss = test_loss(test_output, label)
        loss.backward()
        test_output = test_output.detach().cpu().numpy()
        assert np.allclose(test_output, output_collect)
        assert test_output.shape == output_collect.shape

    def test_psa_mask_distribute(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import PSAMask
        test_loss = Loss()

        input = np.fromfile(
            'tests/data/for_psa_mask/psa_input.bin', dtype=np.float32)
        output_distribute = np.fromfile(
            'tests/data/for_psa_mask/psa_output_distribute.bin',
            dtype=np.float32)

        input = input.reshape((4, 16, 8, 8))
        output_distribute = output_distribute.reshape((4, 64, 8, 8))
        label = torch.ones((4, 64, 8, 8))

        input = torch.FloatTensor(input)
        input.requires_grad = True

        psamask_distribute = PSAMask('distribute', (4, 4))

        # test distribute cpu
        test_output = psamask_distribute(input)
        loss = test_loss(test_output, label)
        loss.backward()
        test_output = test_output.detach().numpy()
        assert np.allclose(test_output, output_distribute)
        assert test_output.shape == output_distribute.shape

        psamask_distribute.cuda()
        input = input.cuda()
        label = label.cuda()

        # test distribute cuda
        test_output = psamask_distribute(input)
        loss = test_loss(test_output, label)
        loss.backward()
        test_output = test_output.detach().cpu().numpy()
        assert np.allclose(test_output, output_distribute)
        assert test_output.shape == output_distribute.shape

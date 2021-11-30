import numpy as np
import pytest
import torch

from mmcv.ops import (SigmoidFocalLoss, SoftmaxFocalLoss, sigmoid_focal_loss,
                      softmax_focal_loss)

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck
    _USING_PARROTS = False

# torch.set_printoptions(precision=8, threshold=100)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
class Testfocalloss(object):

    def setup_class(self):
        self.inputs = [
            ([[1., 0], [0, 1.]], [0, 1]),
            ([[1., 0, -1.], [0, 1., 2.]], [2, 1]),
            ([[1e-6, 2e-6, 3e-6], [4e-6, 5e-5, 6e-4], [7e-3, 8e-2,
                                                       9e-1]], [1, 2, 0]),
        ]
        self.softmax_outputs = [
            (0.00566451, [[-0.00657264, 0.00657264], [0.00657264,
                                                      -0.00657264]]),
            (0.34956908, [[0.10165970, 0.03739851, -0.13905823],
                          [0.01227554, -0.10298023, 0.09070466]]),
            (0.15754992, [[0.02590877, -0.05181759, 0.02590882],
                          [0.02589641, 0.02589760, -0.05179400],
                          [-0.07307514, 0.02234372, 0.05073142]])
        ]
        self.sigmoid_outputs = [
            (0.13562961, [[-0.00657264, 0.11185755], [0.11185755,
                                                      -0.00657264]]),
            (1.10251057, [[0.28808805, 0.11185755, -0.09602935],
                          [0.11185755, -0.00657264, 0.40376765]]),
            (0.42287254, [[0.07457182, -0.02485716, 0.07457201],
                          [0.07457211, 0.07457669, -0.02483728],
                          [-0.02462499, 0.08277918, 0.18050370]])
        ]
        self.outputs = {
            'softmax': self.softmax_outputs,
            'sigmoid': self.sigmoid_outputs
        }
        self.loss_func = {
            'softmax': softmax_focal_loss,
            'sigmoid': sigmoid_focal_loss
        }
        self.loss_class = {
            'softmax': SoftmaxFocalLoss,
            'sigmoid': SigmoidFocalLoss
        }
        self.alpha = 0.25
        self.gamma = 2.0

    def _test_softmax_and_sigmoid(self, mode, dtype):
        for case, output in zip(self.inputs, self.outputs[mode]):
            np_x = np.array(case[0])
            np_y = np.array(case[1])
            np_x_grad = np.array(output[1])

            x = torch.from_numpy(np_x).cuda().type(dtype)
            x.requires_grad_()
            y = torch.from_numpy(np_y).cuda().long()

            loss = self.loss_func[mode](x, y, self.gamma, self.alpha, None,
                                        'mean')
            loss.backward()

            assert np.allclose(loss.data.cpu().numpy(), output[0], 1e-2)
            assert np.allclose(x.grad.data.cpu(), np_x_grad, 1e-2)

    def _test_grad_softmax_and_sigmoid(self, mode, dtype):
        for case in self.inputs:
            np_x = np.array(case[0])
            np_y = np.array(case[1])

            x = torch.from_numpy(np_x).cuda().type(dtype)
            x.requires_grad_()
            y = torch.from_numpy(np_y).cuda().long()

            floss = self.loss_class[mode](self.gamma, self.alpha)
            if _USING_PARROTS:
                # gradcheck(floss, (x, y),
                #           no_grads=[y])
                pass
            else:
                gradcheck(floss, (x, y), eps=1e-2, atol=1e-2)

    @pytest.mark.parametrize('mode', ['softmax', 'sigmoid'])
    @pytest.mark.parametrize('dtype', [torch.float, torch.half])
    def test_softmax_and_sigmoid(self, mode, dtype):
        self._test_softmax_and_sigmoid(mode=mode, dtype=dtype)

    @pytest.mark.parametrize('mode', ['softmax', 'sigmoid'])
    @pytest.mark.parametrize('dtype', [torch.float, torch.half])
    def test_grad_softmax_and_sigmoid(self, mode, dtype):
        self._test_grad_softmax_and_sigmoid(mode=mode, dtype=dtype)

import os
import platform

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

if platform.system() == 'Windows':
    import regex as re
else:
    import re


class TestSyncBN(object):

    def dist_init(self):
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = str(os.environ['SLURM_NODELIST'])

        node_parts = re.findall('[0-9]+', node_list)
        os.environ['MASTER_ADDR'] = (f'{node_parts[1]}.{node_parts[2]}' +
                                     f'.{node_parts[3]}.{node_parts[4]}')
        os.environ['MASTER_PORT'] = '12341'
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)

        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)

    def _test_syncbn_train(self, size=1, half=False):

        if 'SLURM_NTASKS' not in os.environ or int(
                os.environ['SLURM_NTASKS']) != 4:
            print('must run with slurm has 4 processes!\n'
                  'srun -p test --gres=gpu:4 -n4')
            return
        else:
            print('Running syncbn test')
        from mmcv.ops import SyncBatchNorm

        assert size in (1, 2, 4)
        if not dist.is_initialized():
            self.dist_init()
        rank = dist.get_rank()

        torch.manual_seed(9)
        torch.cuda.manual_seed(9)

        self.x = torch.rand(16, 3, 2, 3).cuda()
        self.y_bp = torch.rand(16, 3, 2, 3).cuda()

        if half:
            self.x = self.x.half()
            self.y_bp = self.y_bp.half()
        dist.broadcast(self.x, src=0)
        dist.broadcast(self.y_bp, src=0)

        torch.cuda.synchronize()
        if size == 1:
            groups = [None, None, None, None]
            groups[0] = dist.new_group([0])
            groups[1] = dist.new_group([1])
            groups[2] = dist.new_group([2])
            groups[3] = dist.new_group([3])
            group = groups[rank]
        elif size == 2:
            groups = [None, None, None, None]
            groups[0] = groups[1] = dist.new_group([0, 1])
            groups[2] = groups[3] = dist.new_group([2, 3])
            group = groups[rank]
        elif size == 4:
            group = dist.group.WORLD
        syncbn = SyncBatchNorm(3, group=group).cuda()
        syncbn.weight.data[0] = 0.2
        syncbn.weight.data[1] = 0.5
        syncbn.weight.data[2] = 0.7
        syncbn.train()

        bn = nn.BatchNorm2d(3).cuda()
        bn.weight.data[0] = 0.2
        bn.weight.data[1] = 0.5
        bn.weight.data[2] = 0.7
        bn.train()

        sx = self.x[rank * 4:rank * 4 + 4]
        sx.requires_grad_()
        sy = syncbn(sx)
        sy.backward(self.y_bp[rank * 4:rank * 4 + 4])

        smean = syncbn.running_mean
        svar = syncbn.running_var
        sx_grad = sx.grad
        sw_grad = syncbn.weight.grad
        sb_grad = syncbn.bias.grad

        if size == 1:
            x = self.x[rank * 4:rank * 4 + 4]
            y_bp = self.y_bp[rank * 4:rank * 4 + 4]
        elif size == 2:
            x = self.x[rank // 2 * 8:rank // 2 * 8 + 8]
            y_bp = self.y_bp[rank // 2 * 8:rank // 2 * 8 + 8]
        elif size == 4:
            x = self.x
            y_bp = self.y_bp
        x.requires_grad_()
        y = bn(x)
        y.backward(y_bp)

        if size == 2:
            y = y[rank % 2 * 4:rank % 2 * 4 + 4]
        elif size == 4:
            y = y[rank * 4:rank * 4 + 4]

        mean = bn.running_mean
        var = bn.running_var
        if size == 1:
            x_grad = x.grad
            w_grad = bn.weight.grad
            b_grad = bn.bias.grad
        elif size == 2:
            x_grad = x.grad[rank % 2 * 4:rank % 2 * 4 + 4]
            w_grad = bn.weight.grad / 2
            b_grad = bn.bias.grad / 2
        elif size == 4:
            x_grad = x.grad[rank * 4:rank * 4 + 4]
            w_grad = bn.weight.grad / 4
            b_grad = bn.bias.grad / 4

        assert np.allclose(mean.data.cpu().numpy(),
                           smean.data.cpu().numpy(), 1e-3)
        assert np.allclose(var.data.cpu().numpy(),
                           svar.data.cpu().numpy(), 1e-3)
        assert np.allclose(y.data.cpu().numpy(), sy.data.cpu().numpy(), 1e-3)
        assert np.allclose(w_grad.data.cpu().numpy(),
                           sw_grad.data.cpu().numpy(), 1e-3)
        assert np.allclose(b_grad.data.cpu().numpy(),
                           sb_grad.data.cpu().numpy(), 1e-3)
        assert np.allclose(x_grad.data.cpu().numpy(),
                           sx_grad.data.cpu().numpy(), 1e-2)

    def test_syncbn_1(self):
        self._test_syncbn_train(size=1)

    def test_syncbn_2(self):
        self._test_syncbn_train(size=2)

    def test_syncbn_4(self):
        self._test_syncbn_train(size=4)

    def test_syncbn_1_half(self):
        self._test_syncbn_train(size=1, half=True)

    def test_syncbn_2_half(self):
        self._test_syncbn_train(size=2, half=True)

    def test_syncbn_4_half(self):
        self._test_syncbn_train(size=4, half=True)

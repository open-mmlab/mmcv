import os
import platform

import pytest
import torch
import torch.distributed as dist

from mmcv.cnn.bricks import ConvModule
from mmcv.cnn.utils import revert_sync_batchnorm

if platform.system() == 'Windows':
    import regex as re
else:
    import re


def test_revert_syncbn():
    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='SyncBN'))
    x = torch.randn(1, 3, 10, 10)
    with pytest.raises(ValueError):
        y = conv(x)
    conv = revert_sync_batchnorm(conv)
    y = conv(x)
    assert y.shape == (1, 8, 9, 9)


def test_revert_mmsyncbn():
    if 'SLURM_NTASKS' not in os.environ or int(
            os.environ['SLURM_NTASKS']) != 4:
        print('must run with slurm has 4 processes!\n'
              'srun -p test --gres=gpu:4 -n4')
        return
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
    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='MMSyncBN')).cuda()
    conv.eval()
    x = torch.randn(1, 3, 10, 10)
    y_mmsyncbn = conv(x).detach().cpu()
    conv = revert_sync_batchnorm(conv)
    conv = conv.to('cpu')
    y_bn = conv(x).detach()
    assert y_bn == y_mmsyncbn

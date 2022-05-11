# Copyright (c) OpenMMLab. All rights reserved.
import os
import platform

import numpy as np
import pytest
import torch
import torch.distributed as dist

from mmcv.cnn.bricks import ConvModule
from mmcv.cnn.utils import revert_sync_batchnorm

if platform.system() == 'Windows':
    import regex as re
else:
    import re


@pytest.mark.skipif(
    torch.__version__ == 'parrots', reason='not supported in parrots now')
def test_revert_syncbn():
    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='SyncBN'))
    x = torch.randn(1, 3, 10, 10)
    # Expect a ValueError prompting that SyncBN is not supported on CPU
    with pytest.raises(ValueError):
        y = conv(x)
    conv = revert_sync_batchnorm(conv)
    y = conv(x)
    assert y.shape == (1, 8, 9, 9)


def test_revert_mmsyncbn():
    if 'SLURM_NTASKS' not in os.environ or int(os.environ['SLURM_NTASKS']) < 2:
        print('Must run on slurm with more than 1 process!\n'
              'srun -p test --gres=gpu:2 -n2')
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
    x = torch.randn(1, 3, 10, 10).cuda()
    dist.broadcast(x, src=0)
    conv = ConvModule(3, 8, 2, norm_cfg=dict(type='MMSyncBN')).cuda()
    conv.eval()
    y_mmsyncbn = conv(x).detach().cpu().numpy()
    conv = revert_sync_batchnorm(conv)
    y_bn = conv(x).detach().cpu().numpy()
    assert np.all(np.isclose(y_bn, y_mmsyncbn, 1e-3))
    conv, x = conv.to('cpu'), x.to('cpu')
    y_bn_cpu = conv(x).detach().numpy()
    assert np.all(np.isclose(y_bn, y_bn_cpu, 1e-3))

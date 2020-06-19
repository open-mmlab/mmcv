import os
from unittest.mock import patch

import pytest

from mmcv.runner import init_dist


@patch('torch.distributed.init_process_group')
@patch('subprocess.getoutput', return_value='127.0.0.1')
def test_init_dist(mock_getoutput, mock_dist_init):
    with pytest.raises(ValueError):
        # launcher must be one of {'pytorch', 'mpi', 'slurm'}
        init_dist('invaliad_launcher')

    # test initialize with slurm launcher
    os.environ['SLURM_PROCID'] = '0'
    os.environ['SLURM_NTASKS'] = '1'
    os.environ['SLURM_NODELIST'] = '[0]'  # haven't check the correct form

    init_dist('slurm')
    assert os.environ['MASTER_PORT'] == '29500'
    assert os.environ['MASTER_ADDR'] == '127.0.0.1'
    assert os.environ['WORLD_SIZE'] == '1'
    assert os.environ['RANK'] == '0'
    mock_dist_init.assert_called_once_with(backend='nccl')
    mock_dist_init.reset_mock()

    init_dist('slurm', port=29505)
    assert os.environ['MASTER_PORT'] == '29505'
    assert os.environ['MASTER_ADDR'] == '127.0.0.1'
    assert os.environ['WORLD_SIZE'] == '1'
    assert os.environ['RANK'] == '0'
    mock_dist_init.assert_called_once_with(backend='nccl')
    mock_dist_init.reset_mock()

    init_dist('slurm')
    assert os.environ['MASTER_PORT'] == '29505'
    assert os.environ['MASTER_ADDR'] == '127.0.0.1'
    assert os.environ['WORLD_SIZE'] == '1'
    assert os.environ['RANK'] == '0'
    mock_dist_init.assert_called_once_with(backend='nccl')
    mock_dist_init.reset_mock()

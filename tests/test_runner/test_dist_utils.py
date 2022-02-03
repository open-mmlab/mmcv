import os
from unittest.mock import patch

import pytest

from mmcv.runner import init_dist


@patch('torch.cuda.device_count', return_value=1)
@patch('torch.cuda.set_device')
@patch('torch.distributed.init_process_group')
@patch('subprocess.getoutput', return_value='127.0.0.1')
def test_init_dist(mock_getoutput, mock_dist_init, mock_set_device,
                   mock_device_count):
    with pytest.raises(ValueError):
        # launcher must be one of {'pytorch', 'mpi', 'slurm'}
        init_dist('invaliad_launcher')

    # test initialize with slurm launcher
    os.environ['SLURM_PROCID'] = '0'
    os.environ['SLURM_NTASKS'] = '1'
    os.environ['SLURM_NODELIST'] = '[0]'  # haven't check the correct form

    init_dist('slurm')
    # no port is specified, use default port 29500
    assert os.environ['MASTER_PORT'] == '29500'
    assert os.environ['MASTER_ADDR'] == '127.0.0.1'
    assert os.environ['WORLD_SIZE'] == '1'
    assert os.environ['RANK'] == '0'
    mock_set_device.assert_called_with(0)
    mock_getoutput.assert_called_with('scontrol show hostname [0] | head -n1')
    mock_dist_init.assert_called_with(backend='nccl')

    init_dist('slurm', port=29505)
    # port is specified with argument 'port'
    assert os.environ['MASTER_PORT'] == '29505'
    assert os.environ['MASTER_ADDR'] == '127.0.0.1'
    assert os.environ['WORLD_SIZE'] == '1'
    assert os.environ['RANK'] == '0'
    mock_set_device.assert_called_with(0)
    mock_getoutput.assert_called_with('scontrol show hostname [0] | head -n1')
    mock_dist_init.assert_called_with(backend='nccl')

    init_dist('slurm')
    # port is specified by environment variable 'MASTER_PORT'
    assert os.environ['MASTER_PORT'] == '29505'
    assert os.environ['MASTER_ADDR'] == '127.0.0.1'
    assert os.environ['WORLD_SIZE'] == '1'
    assert os.environ['RANK'] == '0'
    mock_set_device.assert_called_with(0)
    mock_getoutput.assert_called_with('scontrol show hostname [0] | head -n1')
    mock_dist_init.assert_called_with(backend='nccl')

# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmcv.device.ipu import IPU_MODE
from torch.utils.data import Dataset
from mmcv.parallel.data_container import DataContainer
if IPU_MODE:
    from mmcv.device.ipu import IPUDataLoader, parse_ipu_options
    from mmcv.device.ipu.dataloader import collate

skip_no_ipu = pytest.mark.skipif(
    not IPU_MODE, reason='test case under ipu environment')


class ToyDataset(Dataset):
    def __getitem__(self, index):
        return 111

    def __len__(self,):
        return 3


@skip_no_ipu
def test_dataloader_ipu():

    dataloader = IPUDataLoader(None,
                               ToyDataset(),
                               batch_size=256,
                               num_workers=1,
                               mode='async')
    ipu_options = {'train_cfg': {}, 'eval_cfg': {}}
    ipu_options = parse_ipu_options(ipu_options)
    dataloader.init(ipu_options['training'])

@skip_no_ipu
def test_collate_ipu():
    with pytest.raises(TypeError, match='`batch` should be a sequence'):
        collate(123)

    with pytest.raises(TypeError, match='DataContainer is not supported'):
        collate([DataContainer(666)])

    batch = [[1, 2, 3], [2, 3, 4]]
    collate(batch)

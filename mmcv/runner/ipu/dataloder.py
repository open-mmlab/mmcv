# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping, Sequence

import poptorch
from functools import partial
from torch.utils.data.dataloader import default_collate

from mmcv.parallel.data_container import DataContainer


def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        raise TypeError('DataContainer is not supported in ipu data loader.')
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        _list = []
        for samples in transposed:
            if not isinstance(samples[0], DataContainer):
                _list.append(collate(samples, samples_per_gpu))
        return _list
    elif isinstance(batch[0], Mapping):
        _dic = {}
        for key in batch[0]:
            if not isinstance(batch[0][key], DataContainer):
                _dic[key] = collate([d[key] for d in batch])
        return _dic
    else:
        return default_collate(batch)


class IPUDataloader(poptorch.DataLoader):
    def __init__(self,
                 options,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0,
                 drop_last=True,
                 persistent_workers=True,
                 auto_distributed_partitioning=True,
                 mode=poptorch.DataLoaderMode.AsyncRebatched,
                 **kwargs):
        # lazy init
        self.kwargs = {'options': options,
                       'dataset': dataset,
                       'batch_size': batch_size,
                       'shuffle': shuffle,
                       'num_workers': num_workers,
                       'drop_last': drop_last,
                       'persistent_workers': persistent_workers,
                       'auto_distributed_partitioning':
                           auto_distributed_partitioning,
                       'mode': mode,
                       'collate_fn':
                       partial(collate, samples_per_gpu=batch_size),
                       'async_options': {
                           'load_indefinitely': True, 'buffer_size': 8},
                       'rebatched_worker_size': 128,
                       **kwargs}
        self.initialized = False

    def init(self, **kwargs):
        if not self.initialized:
            kwargs = {**self.kwargs, **kwargs}
            super().__init__(**kwargs)
            self.initialized = True

        return self

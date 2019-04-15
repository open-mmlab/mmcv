import collections

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from .data_container import DataContainer


def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, collections.Sequence):
        raise TypeError("{} is not supported.".format(batch.dtype))

    if isinstance(batch[0], DataContainer):
        assert len(batch) % samples_per_gpu == 0
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)
                if (batch[i].pad_dim == 'HW'):
                    ndim = batch[i].dim()
                    assert ndim > 2
                    h = batch[i].size(-2)
                    w = batch[i].size(-1)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim-2):
                            assert batch[i].size(dim) == sample.size(dim)
                        h = max(h, sample.size(-2))
                        w = max(w, sample.size(-1))
                    padded_samples = [
                        F.pad(
                            sample.data,
                            (0, w - sample.size(-1), 0, h - sample.size(-2)),
                            value=sample.padding_value)
                        for sample in batch[i:i + samples_per_gpu]
                    ]
                    stacked.append(default_collate(padded_samples))
                
                elif (batch[i].pad_dim == 'THW'):
                    ndim = batch[i].dim()
                    assert ndim > 3
                    t = batch[i].size(-3)
                    h = batch[i].size(-2)
                    w = batch[i].size(-1)
                    for sample in batch[i: i + samples_per_gpu]:
                        for dim in range(0, ndim-3):
                            assert batch[i].size(dim) == sample.size(dim)
                        t = max(t, sample.size(-3))
                        h = max(h, sample.size(-2))
                        w = max(w, sample.size(-1))
                    padded_samples = [
                        F.pad(
                            sample.data,
                            (0, w - sample.size(-1), 0, h - sample.size(-2), 0, t - sample.size(-3)),
                            value=sample.padding_value)
                        for sample in batch[i:i + samples_per_gpu]
                    ]
                    stacked.append(default_collate(padded_samples))
                elif (batch[i].pad_dim == None):
                    assert batch[i].dim() == 1
                    stacked.append(default_collate([sample.data for sample in batch[i: i+ samples_per_gpu]]))
                else:
                    raise ValueError("pad_dim should be None, 'HW', or 'THW'")

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], collections.Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)

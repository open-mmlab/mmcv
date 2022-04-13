# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch

from mmcv.parallel import DataContainer

# A customized None type for HierarchicalDataManager
HierarchicalDataNone = object()


class HierarchicalDataManager:
    """A class manage all the tensors in the complex data.

    At present, the input data structure accepted by IPU is limited,
    when the input data structure of mmcv varies.
    Here, an intermediate class is needed to get and update tensors
    from the original data.

    HierarchicalDataManager will record a complex input/output data in
    self._data. For example, we have an input data:
    {'img': tensorA, 'label': tensorB, 'img_metas': [tensorC, tensorD]}
    To enable IPU to use the input, HierarchicalDataManager will collect
    the torch tensors from self._data into a tuple like:
    (tensorA, tensorB, tensorC, tensorD).
    Meanwhile, the return of IPU is a tuple of tensors, HierarchicalDataManager
    also have a function named update_all_tensors to update tensors in
    self._data which is the output for upper calls.

    Args:
        logger (:obj:`logging.Logger`): Logger used during running.
             Defaults to None.
    """

    def __init__(self, logger=None):
        self.atomic_types = (int, str, float, np.ndarray, type(None))
        self.warning = warnings.warn if logger is None else logger.warning
        # enable or disable input data's shape and value check
        self.quick_mode = False
        self._data = None

    def quick(self):
        self.quick_mode = True

    def compare_atomic_type(self, a, b):
        """Compare data, supported datatypes are numpy array and python basic
        types."""
        if isinstance(a, np.ndarray):
            return np.all(a == b)
        else:
            return a == b

    def record_hierarchical_data(self, data):
        """Record a complex data."""
        if self._data is not None:
            if isinstance(data, torch.Tensor):
                assert isinstance(self._data, torch.Tensor), \
                    'original complex data is not torch.tensor'
                self._data = data
            else:
                self.update(data)
        else:
            self._data = data

    @property
    def data(self):
        return self._data

    def update(self,
               dataA,
               dataB=HierarchicalDataNone,
               strict=True,
               address='data'):
        """Update recorded complex data in-place.

        Args:
            dataA (list or dict or tuple): New complex data.
            dataB (list or dict or tuple): Complex data to update.
                if not specified, self.data will be updated then.
            strict (bool, optional): If true, an error will be reported
                when the following conditions occur:
                1. Non-torch.Tensor data changed.
                2. Torch.Tensor data shape changed.
            address (str): Record the address of current data to be updated.
                Default: 'data'.
        """
        if dataB is HierarchicalDataNone:
            dataB = self.data

        # Update with a da ta with the same structure
        # but different values(tensors and basic python data types)
        if isinstance(dataA, (tuple, list)):
            for idx, node in enumerate(dataA):
                new_address = ''
                if not self.quick_mode:
                    new_address = address + f'[{str(idx)}]'
                    assert isinstance(node, type(dataB[idx])),\
                        f'data structure changed: {new_address}'
                if isinstance(node, torch.Tensor):
                    dataB[idx] = node
                else:
                    self.update(node, dataB[idx], strict, address=new_address)
        elif isinstance(dataA, dict):
            for k, v in dataA.items():
                new_address = ''
                if not self.quick_mode:
                    new_address = address + f'[{str(k)}]'
                    assert isinstance(v, type(dataB[k])),\
                        f'data structure changed: {new_address}'
                if isinstance(v, torch.Tensor):
                    dataB[k] = v
                else:
                    self.update(v, dataB[k], strict, address=new_address)
        elif isinstance(dataA, self.atomic_types):
            if not self.quick_mode:
                is_equal = self.compare_atomic_type(dataA, dataB)
                if not is_equal:
                    if strict:
                        raise ValueError(
                            'all data except torch.Tensor should be same, '
                            f'but data({address}) is changed.')
                    else:
                        self.warning(
                            f'find a non-torch.Tensor data({type(dataA)}) '
                            f'changed, and the address is {address}')
        elif isinstance(dataA, DataContainer):
            if not self.quick_mode:
                assert isinstance(dataB, DataContainer)
                new_address = address + '.data'
                self.update(dataA.data, dataB.data, False, address=new_address)
        else:
            raise NotImplementedError(
                f'not supported datatype:{type(dataA)}, address is {address}')

    def get_all_tensors(self, hierarchical_data=None):
        """Collect torch.Tensor data from self.data to a list and return."""
        # get a list of tensor from self._data
        if hierarchical_data is None:
            hierarchical_data = self._data
        tensors = []
        if isinstance(hierarchical_data, torch.Tensor):
            tensors = [hierarchical_data]
        else:
            self._get_tensors(hierarchical_data, tensors)
        return tensors

    def _get_tensors(self, data, tensors):
        if isinstance(data, (tuple, list)):
            for node in data:
                if isinstance(node, torch.Tensor):
                    tensors.append(node)
                else:
                    self._get_tensors(node, tensors)
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    tensors.append(v)
                else:
                    self._get_tensors(v, tensors)
        elif isinstance(data, self.atomic_types):
            pass
        elif isinstance(data, DataContainer):
            self._get_tensors(data.data, tensors)
        else:
            raise NotImplementedError(f'not supported datatype:{type(data)}')

    def update_all_tensors(self, tensors):
        """Put tensors from tuple back to self.data."""
        if isinstance(self._data, torch.Tensor):
            print(tensors, len(tensors))
            assert len(tensors) == 1
            assert isinstance(tensors[0], torch.Tensor)
            self._data = tensors[0]
        else:
            # convert to list if tensors is tuple
            tensors = list(tensors)
            self._set_tensors(self._data, tensors)
        return self._data

    def _set_tensors(self, data, tensors):
        if isinstance(data, tuple):
            data = list(data)
            for idx in range(len(data)):
                if isinstance(data[idx], torch.Tensor):
                    data[idx] = tensors.pop(0)
                else:
                    self._set_tensors(data[idx], tensors)
            data = tuple(data)
        elif isinstance(data, list):
            for idx in range(len(data)):
                if isinstance(data[idx], torch.Tensor):
                    data[idx] = tensors.pop(0)
                else:
                    self._set_tensors(data[idx], tensors)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = tensors.pop(0)
                else:
                    self._set_tensors(v, tensors)
        elif isinstance(data, self.atomic_types):
            pass
        elif isinstance(data, DataContainer):
            self._set_tensors(data.data, tensors)
        else:
            raise NotImplementedError(f'not supported datatype:{type(data)}')

    def clean_all_tensors(self):
        """Delete tensors from self.data."""
        self._clean_tensors(self._data)

    def _clean_tensors(self, data):
        if isinstance(data, tuple):
            data = list(data)
            for idx in range(len(data)):
                if isinstance(data[idx], torch.Tensor):
                    data[idx] = None
                else:
                    self._clean_tensors(data[idx])
            data = tuple(data)
        elif isinstance(data, list):
            for idx in range(len(data)):
                if isinstance(data[idx], torch.Tensor):
                    data[idx] = None
                else:
                    self._clean_tensors(data[idx])
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = None
                else:
                    self._clean_tensors(v)
        elif isinstance(data, self.atomic_types):
            pass
        elif isinstance(data, DataContainer):
            self._clean_tensors(data.data)
        else:
            raise NotImplementedError(f'not supported datatype:{type(data)}')

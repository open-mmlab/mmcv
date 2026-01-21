# Copyright (c) OpenMMLab. All rights reserved.
import json
from collections import OrderedDict

import torch


class OpInputInfoLogger:

    def __init__(self, op):
        self.op = op
        self.op_name = self.op.__name__
        print(f'Wrap mmcv.ops.{self.op_name} with OpsInfoLogger')

    def _get_input_info(self, *args, **kwargs):
        input_info = OrderedDict()
        for i, arg in enumerate(args):
            input_info[f'arg_{i}'] = arg
        for name, value in kwargs.items():
            input_info[name] = value
        return input_info

    def _dump_input_info(self, input_info):
        info = dict()
        info[f'mmcv.ops.{self.op_name}'] = OrderedDict()
        for name, param in input_info.items():
            if isinstance(param, torch.Tensor):
                info[f'mmcv.ops.{self.op_name}'][name] = {
                    'shape': str(param.shape),
                    'dtype': str(param.dtype),
                }
            else:
                info[f'mmcv.ops.{self.op_name}'][name] = {
                    'value': str(param),
                    'type': str(type(param)),
                }
        with open('ops_input_info.jsonl', 'a') as f:
            f.write(json.dumps(info) + '\n')

    def __call__(self, *args, **kwargs):
        input_info = self._get_input_info(*args, **kwargs)
        self._dump_input_info(input_info)
        return self.op(*args, **kwargs)

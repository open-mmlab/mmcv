# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp


def func(x):
    return x

_base_ = ['./l1.py', './l2.yaml', './l3.json', './l4.py']
item3 = False
item4 = 'test'
item5 = osp.expanduser('~')

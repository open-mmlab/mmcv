# Copyright (c) OpenMMLab. All rights reserved.

# config now can have imported modules and defined functions for convenience
import os.path as osp
def func():
    return 'string with \"escape\" characters'

str_item_1 = osp.join(osp.expanduser('~'), 'folder') # with backslash in Windows
str_item_2 = func()
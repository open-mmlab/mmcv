# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['./u.py']
item21 = {{ _base_.item11 }}
item22 = item21
item23 = {{ _base_.item10 }}
item24 = item23
item25 = dict(
    a = dict( b = item24 ),
    b = [item24],
    c = [[dict(e = item22)],{{ _base_.item6 }}],
    e = item21
)

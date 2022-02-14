_base_ = ['./py_base.py']
item12 = _base_.item8
item13 = _base_.item9
item14 = _base_.item1
item15 = dict(
    a=dict(b=_base_.item2),
    b=[_base_.item3],
    c=[_base_.item4],
    d=[[dict(e=_base_.item5['a'])], _base_.item6],
    e=_base_.item1)

_base_ = ['./py_base.py']
item12 = item8
item13 = item9
item14 = item1
item15 = dict(
    a=dict(b=item2),
    b=[item3],
    c=[item4],
    d=[[dict(e=item5['a'])], item6],
    e=item1)

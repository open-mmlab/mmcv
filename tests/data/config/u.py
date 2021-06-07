_base_ = ['./t.py']
base = 'base.item8'
item11 = {{ base.item8 }}
item12 = {{ base.item9 }}
item13 = {{ base.item10 }}
item14 = {{ base.item1 }}
item15 = dict(
    a = dict( b = {{ base.item2 }} ),
    b = [{{ base.item3 }}],
    c = [{{ base.item4 }}],
    d = [[dict(e = {{ base.item5.a }})],{{ base.item6 }}],
    e = {{ base.item1 }}
)

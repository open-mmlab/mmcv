_base_ = ['./u.py']
base = 'base.item8'
item21 = {{ base.item11 }}
item22 = item21
item23 = {{ base.item10 }}
item24 = item23
item25 = dict(
    a = dict( b = item24 ),
    b = [item24],
    c = [[dict(e = item22)],{{ base.item6 }}],
    e = item21
)

_base_ = ['./l1.py', './l2.yaml', './l3.json', './l4.py']
item2 = dict(b=[5, 6])
item3 = False
item4 = 'test'
_base_.item6[0] = dict(c=0)
item8 = '{{fileBasename}}'
item9, item10, item11 = _base_.item7['b']['c']

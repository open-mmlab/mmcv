_base_ = ['./l1.py', './l2.yaml', './l3.json', './l4.py']
item3 = False
item4 = 'test'
item8 = '{{fileBasename}}'
item9 = {{ _base_.item2 }}
item10 = {{ _base_.item7.b.c }}

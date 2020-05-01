from mmcv import Config
cfg = Config.fromfile('./tests/data/config/a.py')
item5 = cfg.item1[0] + cfg.item2.a

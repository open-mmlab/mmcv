import os.path as osp

data_path = osp.join(osp.dirname(__file__), 'test_import_modules.txt')
with open(data_path, 'w') as f:
    f.write('test_import_modules')

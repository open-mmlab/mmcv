coverage run -p --source mmcv/runner  tests/test_ipu/test_complexdatamanager.py
coverage run -p --source mmcv/runner tests/test_ipu/test_hooks.py
coverage run -p --source mmcv/runner tests/test_ipu/test_ipu_model.py
coverage run -p --source mmcv/runner tests/test_ipu/test_ipu_runner.py
coverage run -p --source mmcv/runner tests/test_ipu/test_utils.py
coverage combine
coverage html && coverage report
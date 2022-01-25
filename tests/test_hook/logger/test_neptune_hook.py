import shutil
import sys
from unittest.mock import MagicMock

import torch
from tests.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner.hooks import NeptuneLoggerHook


def test_neptune_hook():
    sys.modules['neptune'] = MagicMock()
    sys.modules['neptune.new'] = MagicMock()
    runner = _build_demo_runner()
    hook = NeptuneLoggerHook()

    loader = DataLoader(torch.ones((5, 2)))

    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)

    hook.neptune.init.assert_called_with()
    hook.run['momentum'].log.assert_called_with(0.95, step=6)
    hook.run.stop.assert_called_with()

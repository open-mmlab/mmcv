import re
import tempfile

import torch
import torch.nn as nn

from .test_utils import _build_demo_runner


def test_runner_with_revise_keys():
    import os

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

    class PrefixModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.backbone = Model()

    pmodel = PrefixModel()
    model = Model()
    checkpoint_path = os.path.join(tempfile.gettempdir(), 'checkpoint.pth')

    # add prefix
    torch.save(model.state_dict(), checkpoint_path)
    runner = _build_demo_runner(runner_type='EpochBasedRunner')
    runner.model = pmodel
    state_dict = runner.load_checkpoint(
        checkpoint_path, revise_keys=[(r'^', 'backbone.')])
    for key in pmodel.backbone.state_dict().keys():
        assert torch.equal(pmodel.backbone.state_dict()[key], state_dict[key])
    # strip prefix
    torch.save(pmodel.state_dict(), checkpoint_path)
    runner.model = model
    state_dict = runner.load_checkpoint(
        checkpoint_path, revise_keys=[(r'^backbone\.', '')])
    for key in state_dict.keys():
        key_stripped = re.sub(r'^backbone\.', '', key)
        assert torch.equal(model.state_dict()[key_stripped], state_dict[key])
    os.remove(checkpoint_path)

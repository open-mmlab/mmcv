# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

import mmcv


def test_color():
    assert mmcv.color_val(mmcv.Color.blue) == (255, 0, 0)
    assert mmcv.color_val('green') == (0, 255, 0)
    assert mmcv.color_val((1, 2, 3)) == (1, 2, 3)
    assert mmcv.color_val(100) == (100, 100, 100)
    assert mmcv.color_val(np.zeros(3, dtype=int)) == (0, 0, 0)
    with pytest.raises(TypeError):
        mmcv.color_val([255, 255, 255])
    with pytest.raises(TypeError):
        mmcv.color_val(1.0)
    with pytest.raises(AssertionError):
        mmcv.color_val((0, 0, 500))

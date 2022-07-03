# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import mmcv


def write_to_json(dicts, filename: str):
    """save config to json file.

    Args:
        dicts (_type_): dict files
        filename (str): save path
    """

    with open(filename, 'w', encoding='utf-8') as f:
        mmcv.dump(dicts, f, file_format='json')


def expands_rate(d: int, config: dict) -> list:
    """expand dilation rate according to config.

    Args:
        d (int): _description_
        config (dict): config dict

    Returns:
        list: list of expanded dilation rates
    """
    exp_rate = config['exp_rate']

    return [
        np.clip(
            int(round((1 - exp_rate) * d)), config['mmin'], config['mmax']),
        np.clip(d, config['mmin'], config['mmax']),
        np.clip(
            int(round((1 + exp_rate) * d)), config['mmin'], config['mmax']),
    ]

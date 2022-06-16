# Copyright (c) OpenMMLab. All rights reserved.
import json


def load_structure(filename: str) -> dict:
    """load structure file.

    Args:
        filename (str): file path

    Returns:
        dict: model config
    """
    with open(filename, encoding='utf-8') as f:
        config = json.load(f)
        return config['model']


def write_to_json(dicts, filename: str):
    """save config to json file.

    Args:
        dicts (_type_): dict files
        filename (str): save path
    """

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dicts, f, indent=4)


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
        value_crop(
            int(round((1 - exp_rate) * d)), config['mmin'], config['mmax']),
        value_crop(d, config['mmin'], config['mmax']),
        value_crop(
            int(round((1 + exp_rate) * d)), config['mmin'], config['mmax']),
    ]


def value_crop(d: int, mind: int, maxd: int) -> int:
    """crop dilation value.

    Args:
        d (int): dilation rate
        mind (int): min dilation rate
        maxd (int): max dilation rate

    Returns:
        int: dilation rate
    """
    if d < mind:
        d = mind
    elif d > maxd:
        d = maxd
    return d

# Copyright (c) OpenMMLab. All rights reserved.
import json

# def init_config():
#     config = {}
#     config["search"] = {}
#     config["model"] = {}
#     config["search"]["step"] = 0
#     config["search"]["max_step"] = 11
#     config["search"]["search_interval"] = 1
#     config["search"]["exp_rate"] = 0.5
#     config["search"]["init_alphas"] = 0.01
#     config["search"]["normlize"] = "absavg"
#     config["search"]["mmin"] = 1
#     config["search"]["mmax"] = 24
#     config["search"]["finetune"] = False
#     return config


def load_structure(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        config = json.load(f)
        return config['model']


def write_to_json(dicts, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dicts, f, indent=4)


def expands_rate(d, config):
    exp_rate = config['exp_rate']

    return [
        value_crop(
            int(round((1 - exp_rate) * d)), config['mmin'], config['mmax']),
        value_crop(d, config['mmin'], config['mmax']),
        value_crop(
            int(round((1 + exp_rate) * d)), config['mmin'], config['mmax']),
    ]


def value_crop(d, mind, maxd):
    if d < mind:
        d = mind
    elif d > maxd:
        d = maxd
    return d

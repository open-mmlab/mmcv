import argparse
import json
import os
import os.path as osp
import time
import warnings
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

import mmcv
from mmcv.runner import build_runner
from mmcv.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir', default='./', help='the dir to save logs and models')
    parser.add_argument(
        '--num-iters', default=300, help='The number of iters per epoch')
    parser.add_argument(
        '--num-epochs', default=10, help='Only used in EpochBasedRinner')
    parser.add_argument(
        '--window-size',
        default='12*7',
        help='Size of the window to display images, in format of "$W*$H".')
    parser.add_argument(
        '--log-interval', default=1, help='The iterval of TextLoggerHook')
    args = parser.parse_args()
    return args


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def train_step(self, *args, **kwargs):
        return dict()

    def val_step(self, *args, **kwargs):
        return dict()


@patch('mmcv.runner.EpochBasedRunner.run_iter', Mock())
def run(cfg, logger):
    assert cfg.get('lr_config')
    model = SimpleModel()
    optimizer = SGD(model.parameters(), 0.1)
    cfg.work_dir = cfg.get('work_dir', './')
    workflow = [('train', 1)]

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.get('total_epochs', cfg.num_epochs)
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    fake_dataloader = DataLoader(list(range(cfg.num_iters)), batch_size=1)
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=None))
    lr_config = cfg.lr_config
    log_config = dict(
        interval=cfg.log_interval, hooks=[
            dict(type='TextLoggerHook'),
        ])
    runner.register_training_hooks(lr_config, log_config=log_config)
    runner.run([fake_dataloader], workflow)


def plot_lr_curve(json_file, cfg):
    lr_list = []
    assert os.path.isfile(json_file)
    with open(json_file, 'r') as f:
        for line in f:
            log = json.loads(line.strip())
            lr_list.append(log['lr'])
    wind_w, wind_h = [int(size) for size in cfg.window_size.split('*')]
    plt.figure(figsize=(wind_w, wind_h))
    # if legend is None, use {filename}_{key} as legend

    ax: plt.Axes = plt.subplot()

    ax.plot(lr_list, linewidth=1)
    if cfg.runner.type == 'EpochBasedRunner':
        ax.xaxis.tick_top()
        ax.set_xlabel('Iters')
        ax.xaxis.set_label_position('top')
        sec_ax = ax.secondary_xaxis(
            'bottom',
            functions=(lambda x: x / cfg.num_iters * cfg.log_interval,
                       lambda y: y * cfg.num_iters * cfg.log_interval))
        sec_ax.set_xlabel('Epochs')
    else:
        plt.xlabel('Iters')
    plt.ylabel('Learning Rate')
    title = cfg.lr_config.type
    plt.title(title)

    save_path = osp.join(cfg.work_dir, title)
    plt.savefig(save_path)
    print(f'The learning rate graph is saved at {save_path}')
    plt.show()


def main():
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg = mmcv.Config.fromfile(args.config)
    cfg['num_iters'] = args.num_iters
    cfg['num_epochs'] = args.num_epochs
    cfg['log_interval'] = args.log_interval
    cfg['window_size'] = args.window_size

    log_path = osp.join(cfg.get('work_dir', './'), f'{timestamp}.log')
    json_path = log_path + '.json'
    logger = get_logger('mmcv', log_path)

    run(cfg, logger)
    plot_lr_curve(json_path, cfg)


if __name__ == '__main__':
    main()

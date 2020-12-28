import argparse
import os
import os.path as osp
import random

import numpy as np
import torch

from ..runner import get_dist_info, init_dist
from ..utils import (Config, DictAction, import_modules_from_strings,
                     mkdir_or_exist)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def default_args_parser():
    """Default argument parser for OpenMMLab projects.

    This function is used as a default argument parser in OpenMMLab projects.
    To add customized arguments, users can create a new parser function which
    calls this functions first.

    Returns:
        :obj:`argparse.ArgumentParser`: Argument parser
    """
    parser = argparse.ArgumentParser(
        description='OpenMMLab Default Argument Parser')

    # common arguments for both training and testing
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results in multi-gpu testing.')

    # common arguments for training
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')

    # common arguments for testing
    parser.add_argument(
        '--test-only', action='store_true', help='whether to perform evaluate')
    parser.add_argument(
        '--checkpoint', help='checkpoint file used in evaluation')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    # TODO: decide whether to maintain two place for modifing eval options
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')

    return parser


def setup_cfg(args, cfg_args):
    """Set up config.

    Note:
        This function assumes the arguments are parsed from the parser of
        defined by :meth:`default_args_parser`, which contains necessary keys
        for distributed training including 'launcher', 'local_rank', etc.

    Arguments:
        args (:obj:`argparse.ArgumentParser`): arguments from entry point
        cfg_args (list[str]): list of key-value pairs that will be merged
            into cfgs.

    Returns:
        Config: config dict
    """
    cfg = Config.fromfile(args.config)
    # merge config from args.cfg_options
    if len(cfg_args) > 0:
        cfg.merge_from_arg_list(cfg_args)

    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # initialize some default but necessary options
    cfg.seed = cfg.get('seed', None)
    cfg.deterministic = cfg.get('deterministic', False)
    cfg.resume_from = cfg.get('resume_from', None)

    cfg.launcher = args.launcher
    cfg.local_rank = args.local_rank
    if args.launcher == 'none':
        cfg.distributed = False
    else:
        cfg.distributed = True

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])
    return cfg


def setup_envs(cfg, dump_cfg=True):
    """Setup running environments.

    This function initialize the running environment.
    It does the following things in order:

        1. Set local rank in the environment
        2. Set cudnn benchmark
        3. Initialize distributed function
        4. Create work dir anddump config file
        5. Set random seed

    Args:
        cfg (:obj:`Config`): Config object.
        dump_cfg: Whether to dump configs.
    """
    # set local rank
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(cfg.local_rank)

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', False)

    # init distributed env first, since logger depends on the dist info.
    if cfg.distributed:
        init_dist(cfg.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    if cfg.local_rank == 0 and dump_cfg:
        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(cfg.filename)))

    # set random seeds
    if cfg.seed is not None:
        set_random_seed(cfg.seed, deterministic=cfg.deterministic)


def gather_info(cfg, logger, env_info_dict):
    """Gather running information.

    This function do the following things in order:

        1. collect & log env info
        2. collect exp name, config

    Args:
        cfg (:obj:`Config`): Config object.
        logger (:obj:`logging.logger`): Logger.
        env_info_dict (dict): Environment information.
    """
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['seed'] = cfg.seed
    meta['exp_name'] = osp.basename(cfg.filename)

    # log some basic info
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    logger.info(f'Set random seed to {cfg.seed}, '
                f'deterministic: {cfg.deterministic}')
    logger.info(f'Distributed training: {cfg.distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    return meta

import subprocess
from typing import Union

import torch

from mmcv import dump, load
from mmcv.utils import Config


def update_model_zoo(name: str, config: str, details: dict,
                     checkpoint_path: str, json_path: str) -> None:
    r"""Update the model zoo information in the json file.

    The format of json file storing the model information should be as below:

    .. code-block:: json

    [
        {
            'name': str,
            'config': str,
            'details': {
                'metric_a': float,
                'metric_b': float
            },
            'path': str
        }
        ...
    ]

    Args:
        name (str): The name of the model to be updated. Usually, it is
            the name of the config.
        config (str): The path of the model's config.
            It should be a relative path of the repo that the model belongs to.
        details (dict[str, float]): The detail information of the model,
            including performance and latency.
        checkpoint_path (str): The path of the checkpoint. It a path that can
            be loaded by the load_checkpoint API implemented in MMCV.
        json_path (str): The JSON file path to be updated.
    """
    ori_json = load(json_path)
    # TODO: add check logics, including
    # 1. whether the name already exists
    # 2. the validity of path
    ori_json.append(
        dict(name=name, config=config, details=details, path=checkpoint_path))
    dump(json_path, ori_json)


def process_checkpoint(in_file: str, out_file: str) -> None:
    """Process the checkpoint to make it ready for publish.

    This function removes 'optimizer' in the checkpoint that is used for
    resuming training. Then it adds hash to the output file.

    Args:
        in_file (str): Path of the original checkpoint to be converted.
        out_file (str): Target path of the converted checkpoint.
    """
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file
    final_file = out_file_name + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])


def get_model_by_config(builder: callable,
                        config: Union[str, Config],
                        pretrained: bool = True) -> callable:
    """Obtain the model given builder and config name.

    Args:
        builder (callable): The build function that builds model based
            on the config.
        config (str, :obj:`Config`): The the path of config or a Config object.
        pretrained (bool, optional): Whether load the released pre-trained
            model. Defaults to True.

    Returns:
        tuple: (model, details)
    """

    # TODO: support to load the json for only once
    #


def find_model_by_name():
    pass


def get_builder_from_repo():
    pass


def load_model(
        repo_or_dir: str,
        name: str,
        pretrained: bool = True,
        override_config: Union[str, dict,
                               Config] = None) -> tuple[callable, dict]:
    """High level API that loads a model from a given github repo or a local
    directory.

    Args:
        repo_or_dir (str): Github/Gitlab repo or a local directory path.
        name (str): The name of the model.
        pretrained (bool): Whether to load the pretrained checkpoint.
            Defaults to True.
        override_config (Union[str, dict, :obj:`Config`], optional): The config
            used to override the original model config. If it is a str, it will
            be taken as an absolute path to load a :obj:`Config`.
            The config will be merged with the original config mapped by the
            name. Defaults to None.

    Returns:
        tuple[callable, dict]: Retuan the model and its corresponding detail
            information.
    """
    # obtain builder from the repo
    builder = get_builder_from_repo(repo_or_dir)

    # obtain info of the model by name
    model_info = find_model_by_name(name)
    config = model_info['config']

    # infer config from name
    if override_config is not None:
        if not isinstance(override_config, (dict, Config)):
            assert isinstance(override_config, str)
            override_config = Config.fromfile(override_config)
        config = Config.fromfile(config)
        config.merge_from_dict(override_config)

    model = get_model_by_config(builder, config, pretrained=pretrained)
    return model, model_info['details']

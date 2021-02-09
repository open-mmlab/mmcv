import subprocess

import torch

from mmcv import dump, load


def update_model_zoo(name: str, config: str, details: dict,
                     checkpoint_path: str, json_path: str) -> None:
    """Update the model zoo information in the json file.

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

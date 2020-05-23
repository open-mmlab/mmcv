## Model Zoo

The following types are supported for `filename` argument of `mmcv.load_checkpoint()`.
* filepath: The filepath of the checkpoint.
* link: The link to download the checkpoint. The `SHA256` prefix should be contained in the filename.
* `torchvison://xxx`: The model links in `torchvision.models`.Please refer to [torchvision](https://pytorch.org/docs/stable/torchvision/models.html) for details.
* `open-mmlab://yyy`: The model links provided in default and external json files.
* `local://zzz`: The model filepaths provided in external json files.

### Model URLs in JSON
The json file consists of key-value pair of model name and its url/path.
An example json file could be like:
```json
{
    `model_a`: `https://example.com/models/model_a_9e5bac.pth,
    `model_b`: `pretrain/model_b_ab3ef2c.pth`,
}
```
The default links of the pre-trained models hosted on Open-MMLab AWS could be found [here](../mmcv/urls/open_mmlab.json).

You may put additional json files under `MMCV_HOME`. If `MMCV_HOME` is not find in the environment, `~/.cache/mmcv` will be used by default. You may `export MMCV_HOME=/your/path` to use your own path.

The external json files will be merged into default one. If the same key presents in both external json and default json, the external one will be used.

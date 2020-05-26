## Model Zoo
Besides torchvision pre-trained models, we also provide pre-trained models of following CNN:
* VGG Caffe
* ResNet Caffe
* ResNeXt
* ResNet with Group Normalization
* ResNet with Group Normalization and Weight Standardization
* HRNetV2
* Res2Net
* RegNet

### Model URLs in JSON
The model zoo links in MMCV are managed by JSON files.
The json file consists of key-value pair of model name and its url or path.
An example json file could be like:
```json
{
    "model_a": "https://example.com/models/model_a_9e5bac.pth",
    "model_b": "pretrain/model_b_ab3ef2c.pth"
}
```
The default links of the pre-trained models hosted on Open-MMLab AWS could be found [here](../mmcv/model_zoo/open_mmlab.json).

You may override default links by putting `open-mmlab.json` under `MMCV_HOME`. If `MMCV_HOME` is not find in the environment, `~/.cache/mmcv` will be used by default. You may `export MMCV_HOME=/your/path` to use your own path.

The external json files will be merged into default one. If the same key presents in both external json and default json, the external one will be used.

### Load Checkpoint
The following types are supported for `filename` argument of `mmcv.load_checkpoint()`.
* filepath: The filepath of the checkpoint.
* `http://xxx` and `https://xxx`: The link to download the checkpoint. The `SHA256` postfix should be contained in the filename.
* `torchvison://xxx`: The model links in `torchvision.models`.Please refer to [torchvision](https://pytorch.org/docs/stable/torchvision/models.html) for details.
* `open-mmlab://xxx`: The model links or filepath provided in default and additional json files.

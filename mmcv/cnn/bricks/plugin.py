# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import platform
from typing import Dict, Tuple, Union

import torch.nn as nn
from mmengine.registry import MODELS

if platform.system() == 'Windows':
    import regex as re  # type: ignore
else:
    import re  # type: ignore


def infer_abbr(class_type: type) -> str:
    """Infer abbreviation from the class name.

    This method will infer the abbreviation to map class types to
    abbreviations.

    Rule 1: If the class has the property "abbr", return the property.
    Rule 2: Otherwise, the abbreviation falls back to snake case of class
    name, e.g. the abbreviation of ``FancyBlock`` will be ``fancy_block``.

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """

    def camel2snack(word):
        """Convert camel case word into snack case.

        Modified from `inflection lib
        <https://inflection.readthedocs.io/en/latest/#inflection.underscore>`_.

        Example::

            >>> camel2snack("FancyBlock")
            'fancy_block'
        """

        word = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', word)
        word = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', word)
        word = word.replace('-', '_')
        return word.lower()

    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_  # type: ignore
    else:
        return camel2snack(class_type.__name__)


def build_plugin_layer(cfg: Dict,
                       postfix: Union[int, str] = '',
                       **kwargs) -> Tuple[str, nn.Module]:
    """Build plugin layer.

    Args:
        cfg (dict): cfg should contain:

            - type (str): identify plugin layer type.
            - layer args: args needed to instantiate a plugin layer.
        postfix (int, str): appended into norm abbreviation to
            create named layer. Default: ''.

    Returns:
        tuple[str, nn.Module]: The first one is the concatenation of
        abbreviation and postfix. The second is the created plugin layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    # Switch registry to the target scope. If `plugin_layer` cannot be found
    # in the registry, fallback to search `plugin_layer` in the
    # mmengine.MODELS.
    with MODELS.switch_scope_and_registry(None) as registry:
        plugin_layer = registry.get(layer_type)
    if plugin_layer is None:
        raise KeyError(f'Cannot find {plugin_layer} in registry under scope '
                       f'name {registry.scope}')
    abbr = infer_abbr(plugin_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = plugin_layer(**kwargs, **cfg_)

    return name, layer

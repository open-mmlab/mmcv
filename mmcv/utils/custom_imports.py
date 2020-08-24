# Copyright (c) Open-MMLab. All rights reserved.

import importlib


def import_modules_from_strings(imports):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> runner, image = import_modules_from_strings(
        ...     ['mmcv.runner', 'mmcv.image'])
        >>> import mmcv.runner as runner2
        >>> import mmcv.image as image2
        >>> assert runner == runner2
        >>> assert image == image2
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
    imported = [importlib.import_module(imp) for imp in imports]
    if single_import:
        imported = imported[0]
    return imported

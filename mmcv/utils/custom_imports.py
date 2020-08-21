# Copyright (c) Open-MMLab. All rights reserved.

import importlib


def custom_imports(imports):
    """Import a module from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>>runner, image = custom_imports(['mmcv.runner', 'mmcv.image'])
        >>>import mmcv.runner as runner2
        >>>import mmcv.image as image2
        >>>assert runner == runner2
        >>>assert image == image2
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]

    assert isinstance(
        imports,
        list), (f'custom_imports must be a list but got {type(imports)}')
    imported = [importlib.import_module(imp) for imp in imports]
    if single_import:
        imported = imported[0]
    return imported

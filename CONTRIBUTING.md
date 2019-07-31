# Contributing to mmcv

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components

## Workflow

1. fork and pull the latest mmcv
2. checkout a new branch (do not use master branch for PRs)
3. commit your changes
4. create a PR

Note: If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.

## Code style

### Python
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:
- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [yapf](https://github.com/google/yapf): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

Style configurations of yapf and isort can be found in [.style.yapf](.style.yapf) and [.isort.cfg](.isort.cfg).

>Before you create a PR, make sure that your code lints and is formatted by yapf.

### C++ and CUDA
We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

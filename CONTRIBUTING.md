# Contributing to OpenMMLab

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components

## Workflow

1. fork and pull the latest OpenMMLab repository
2. checkout a new branch (do not use master branch for PRs)
3. commit your changes
4. create a PR

Note: If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.

## Code style

### Python

We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:

- [flake8](http://flake8.pycqa.org/en/latest/): A wrapper around some linter tools.
- [yapf](https://github.com/google/yapf): A formatter for Python files.
- [isort](https://github.com/timothycrosley/isort): A Python utility to sort imports.
- [markdownlint](https://github.com/markdownlint/markdownlint): A linter to check markdown files and flag style issues.
- [docformatter](https://github.com/myint/docformatter): A formatter to format docstring.

Style configurations of yapf and isort can be found in [setup.cfg](./setup.cfg).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `isort`, `trailing whitespaces`, `markdown files`,
fixes `end-of-files`, `double-quoted-strings`, `python-encoding-pragma`, `mixed-line-ending`, sorts `requirments.txt` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](./.pre-commit-config.yaml).

After you clone the repository, you will need to install initialize pre-commit hook.

```shell
pip install -U pre-commit
```

From the repository folder

```shell
pre-commit install
```

Try the following steps to install ruby when you encounter an issue on installing markdownlint

```shell
# install rvm
curl -L https://get.rvm.io | bash -s -- --autolibs=read-fail
[[ -s "$HOME/.rvm/scripts/rvm" ]] && source "$HOME/.rvm/scripts/rvm"
rvm autolibs disable

# install ruby
rvm install 2.7.1
```

Or refer to [this repo](https://github.com/innerlee/setup) and take [`zzruby.sh`](https://github.com/innerlee/setup/blob/master/zzruby.sh) according its instruction.

After this on every commit check code linters and formatter will be enforced.

>Before you create a PR, make sure that your code lints and is formatted by yapf.

### C++ and CUDA

We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

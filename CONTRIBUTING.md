## Contributing to OpenMMLab

All kinds of contributions are welcome, including but not limited to the following.

- Fix typo or bugs
- Add documentation or translate the documentation into other languages
- Add new features and components

```{note}
If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.
```

### Code style

#### Python

We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:

- [flake8](https://github.com/PyCQA/flake8): A wrapper around some linter tools.
- [isort](https://github.com/timothycrosley/isort): A Python utility to sort imports.
- [yapf](https://github.com/google/yapf): A formatter for Python files.
- [codespell](https://github.com/codespell-project/codespell): A Python utility to fix common misspellings in text files.
- [mdformat](https://github.com/executablebooks/mdformat): Mdformat is an opinionated Markdown formatter that can be used to enforce a consistent style in Markdown files.
- [docformatter](https://github.com/myint/docformatter): A formatter to format docstring.

Style configurations of yapf and isort can be found in [setup.cfg](./setup.cfg).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `isort`, `trailing whitespaces`, `markdown files`,
fixes `end-of-files`, `double-quoted-strings`, `python-encoding-pragma`, `mixed-line-ending`, sorts `requirments.txt` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](./.pre-commit-config.yaml).

#### C++ and CUDA

We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

### Workflow

1. fork and pull the latest OpenMMLab repository
   ![image](https://user-images.githubusercontent.com/57566630/167305749-43c7f4e9-449b-4e98-ade5-0c9276d5c9ce.png)

   ```shell
   git clone https://github.com/{username}/mmcv.git
   ```

2. Configure pre-commit for the first time

   After you clone the repository, you will need to install initialize pre-commit hook.

   ```shell
   pip install -U pre-commit
   pre-commit install
   ```

   After this on every commit check code linters and formatter will be enforced.

   > Before you create a PR, make sure that your code lints and is formatted by yapf.

   **pre-commit failed**(some code will be fixed automatically)

   ![image](https://user-images.githubusercontent.com/57566630/167306461-3cb3b5bf-d9b3-4d5a-9c0a-34cfded8dbbc.png)

   **pre-commit success**

   ![image](https://user-images.githubusercontent.com/57566630/167306496-d2b8daf7-d72c-4129-a0e8-175f8a32cc47.png)

   If you are bothered by pre-commit when commit temporally change, you can commit your change with `--no-verify`

   ```shell
   git commit -m "xxx" --no-verify
   ```

3. checkout a new branch (do not use master branch for PRs)

   It is recommended to name your branch with 'username/pr_name'

   ```shell
   git checkout -b username/refactor_contributing_doc
   ```

4. make changes and add unit test if necessary

   If your pr add some features or affect the logic of the previous implementation, a corresponding unit test should be added to the directory `tests`.

5. Pass corresponding unit test.

   Your changes should pass through the corresponding unit test at lest. For example, you make some changes in `mmcv/runner/epoch_based_runner.py`, then you should:

   ```shell
   pytest tests/test_runner/test_runner.py
   ```

6. commit your changes

   Make sure your changes pass through pre-commit.

7. create a PR

   If you create for the first time, CLA should be assigned

   ![image](https://user-images.githubusercontent.com/57566630/167307569-a794b967-6e28-4eac-a942-00deb657815f.png)

   Consider CI/CD will build mmcv in different platform, you can check your building status in:

   ![image](https://user-images.githubusercontent.com/57566630/167307490-f9ebf9fa-63c0-4d83-8ba1-081ea169eb3a.png)

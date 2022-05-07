## 贡献代码

欢迎任何类型的贡献，包括但不限于

- 修改拼写错误或代码错误
- 添加文档或将文档翻译成其他语言
- 添加新功能和新组件

### 工作流

| 详细工作流见 [拉取请求](pr.md)

1. 复刻并拉取最新的 OpenMMLab 算法库
2. 创建新的分支（不建议使用主分支提拉取请求）
3. 提交你的修改
4. 创建拉取请求

```{note}
如果你计划添加新功能并且该功能包含比较大的改动，建议先开 issue 讨论
```

### 代码风格

#### Python

[PEP8](https://www.python.org/dev/peps/pep-0008/) 作为 OpenMMLab 算法库首选的代码规范，我们使用以下工具检查和格式化代码

- [flake8](https://github.com/PyCQA/flake8): Python 官方发布的代码规范检查工具，是多个检查工具的封装
- [isort](https://github.com/timothycrosley/isort): 自动调整模块导入顺序的工具
- [yapf](https://github.com/google/yapf): Google 发布的代码规范检查工具
- [codespell](https://github.com/codespell-project/codespell): 检查单词拼写是否有误
- [mdformat](https://github.com/executablebooks/mdformat): 检查 markdown 文件的工具
- [docformatter](https://github.com/myint/docformatter): 格式化 docstring 的工具

yapf 和 isort 的配置可以在 [setup.cfg](./setup.cfg) 找到

通过配置 [pre-commit hook](https://pre-commit.com/) ，我们可以在提交代码时自动检查和格式化 `flake8`、`yapf`、`isort`、`trailing whitespaces`、`markdown files`，
修复 `end-of-files`、`double-quoted-strings`、`python-encoding-pragma`、`mixed-line-ending`，调整 `requirments.txt` 的包顺序。
pre-commit 钩子的配置可以在 [.pre-commit-config](./.pre-commit-config.yaml) 找到。

在克隆算法库后，你需要安装并初始化 pre-commit 钩子

```shell
pip install -U pre-commit
```

切换算法库根目录

```shell
pre-commit install
```

> 提交拉取请求前，请确保你的代码符合 yapf 的格式

#### C++ and CUDA

C++ 和 CUDA 的代码规范遵从 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

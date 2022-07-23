## 安装 MMCV

MMCV 有两个版本：

- **mmcv-full**: 完整版，包含所有的特性以及丰富的开箱即用的 CPU 和 CUDA 算子。注意完整版本可能需要更长时间来编译。
- **mmcv**: 精简版，不包含 CPU 和 CUDA 算子但包含其余所有特性和功能，类似 MMCV 1.0 之前的版本。如果你不需要使用 CUDA 算子的话，精简版可以作为一个考虑选项。

```{warning}
请不要在同一个环境中安装两个版本，否则可能会遇到类似 `ModuleNotFound` 的错误。在安装一个版本之前，需要先卸载另一个。`如果CUDA可用，强烈推荐安装mmcv-full`。
```

### 安装完整版 mmcv-full

```{important}
下述安装步骤仅适用于 Linux 和 Windows 平台，如需在 macOS 平台安装 mmcv-full，请参考[源码安装 mmcv-full](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html#macos-mmcv-full)。
```

在安装 mmcv-full 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://github.com/pytorch/pytorch#installation)。可使用以下命令验证

```bash
python -c 'import torch;print(torch.__version__)'
```

#### 使用 mim 安装（推荐）

```bash
pip install -U openmim
mim install mmcv-full
```

如需安装指定版本的 mmcv-full，例如安装 1.6.0 版本的 mmcv-full，可使用以下命令

```bash
mim install mmcv-full==1.6.0
```

:::{note}
如果你打算使用 `opencv-python-headless` 而不是 `opencv-python`，例如在一个很小的容器环境或者没有图形用户界面的服务器中，你可以先安装 `opencv-python-headless`，这样在安装 mmcv 依赖的过程中会跳过 `opencv-python`。

另外，如果安装依赖库的时间过长，可以指定 pypi 源，例如

```bash
mim install mmcv-full -i https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

安装完成后可以运行 [check_installation.py](https://github.com/open-mmlab/mmcv/.dev_scripts/check_installation.py) 脚本检查 mmcv-full 是否安装成功。

如果发现上述的安装命令没有使用 mmcv-full 预编译包安装，则表示没有提供对应 PyTorch 或者 CUDA 或者 mmcv-full 版本的预编译包，此时，请参考[源码安装 mmcv-full](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html)。

#### 使用 pip 安装

使用以下命令查看 CUDA 和 PyTorch 的版本

```bash
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'
```

**TODO**

<html>
<body>
<select id="platform" onChange="change()">
    <option selected="selected">Linux</option>
    <option>Windows</option>
</select>
<select id="pytorch" onChange="change()">
    <option selected="selected">torch1.10.x</option>
    <option>torch1.9.x</option>
</select>
<select id="cuda" onChange="change()">
    <option selected="selected">11.1</option>
    <option>10.2</option>
</select>
<select id="mmcv" onChange="change()">
    <option selected="selected">1.6.0</option>
    <option>1.5.3</option>
</select>

<script>
function change()
{
  var x = document.getElementById("platform");
  var y = document.getElementById("pytorch");
  y.options.length = 0;
  if(x.selectedIndex == 0)
  {
    y.options.add(new Option("1.10.x", "0"));
    y.options.add(new Option("1.9.x", "1", false, true));
  }
  if(x.selectedIndex == 1)
  {
    y.options.add(new Option("1.7.x", "0"));
    y.options.add(new Option("1.6.x", "1", false, true));
  }
}
</script>

</body>
</html>

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

:::{note}
PyTorch 在 1.x.0 和 1.x.1 之间通常是兼容的，故 mmcv-full 只提供 1.x.0 的编译包。如果你
的 PyTorch 版本是 1.x.1，你可以放心地安装在 1.x.0 版本编译的 mmcv-full。例如，如果你的
PyTorch 版本是 1.8.1，你可以放心选择 1.8.x。
:::

:::{note}
如果你打算使用 `opencv-python-headless` 而不是 `opencv-python`，例如在一个很小的容器环境或者没有图形用户界面的服务器中，你可以先安装 `opencv-python-headless`，这样在安装 mmcv 依赖的过程中会跳过 `opencv-python`。

另外，如果安装依赖库的时间过长，可以指定 pypi 源，例如

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html -i https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

如果上面的下拉框中没有找到对应的版本，则表示没有提供对应 PyTorch 或者 CUDA 或者 mmcv-full 版本的预编译包，此时，请参考[源码安装 mmcv-full](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html)。

#### 使用 docker 镜像

先将算法库克隆到本地再构建镜像

```bash
git clone https://github.com/open-mmlab/mmcv.git && cd mmcv
docker build -t mmcv -f docker/release/Dockerfile .
```

也可以直接使用下面的命令构建镜像

```bash
docker build -t mmcv https://github.com/open-mmlab/mmcv.git#master:docker/release
```

[Dockerfile](release/Dockerfile) 默认安装最新的 mmcv-full，如果你想要指定版本，可以使用下面的命令

```bash
docker image build -t mmcv -f docker/release/Dockerfile --build-arg MMCV=1.5.0 .
```

如果你想要使用其他版本的 PyTorch 和 CUDA，你可以在构建镜像是指定它们的版本。

例如指定 PyTorch 的版本是 1.11，CUDA 的版本是 11.3

```bash
docker build -t mmcv -f docker/release/Dockerfile \
    --build-arg PYTORCH=1.9.0 \
    --build-arg CUDA=11.1 \
    --build-arg CUDNN=8 \
    --build-arg MMCV=1.5.0 .
```

更多 PyTorch 和 CUDA 镜像可以点击 [dockerhub/pytorch](https://hub.docker.com/r/pytorch/pytorch/tags)。

### 安装精简版 mmcv

```python
pip install mmcv
```

### 安装完整版并且编译 onnxruntime 的自定义算子

详细的指南请查看 [这里](https://mmcv.readthedocs.io/zh_CN/latest/deployment/onnxruntime_custom_ops.html)。

如果想从源码编译 MMCV，请参考[该文档](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html)。

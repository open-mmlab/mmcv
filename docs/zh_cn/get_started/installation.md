## 安装 MMCV

MMCV 有两个版本：

- **mmcv-full**: 完整版，包含所有的特性以及丰富的开箱即用的 CUDA 算子。注意完整版本可能需要更长时间来编译。
- **mmcv**: 精简版，不包含 CUDA 算子但包含其余所有特性和功能，类似 MMCV 1.0 之前的版本。如果你不需要使用 CUDA 算子的话，精简版可以作为一个考虑选项。

```{warning}
请不要在同一个环境中安装两个版本，否则可能会遇到类似 `ModuleNotFound` 的错误。在安装一个版本之前，需要先卸载另一个。`如果CUDA可用，强烈推荐安装mmcv-full`。
```

a. 安装完整版

在安装 mmcv-full 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 PyTorch 官方[文档](https://pytorch.org/)。

我们提供了 **Linux 和 Windows 平台** PyTorch 和 CUDA 版本组合的 mmcv-full 预编译包，可以大大简化用户安装编译过程。强烈推荐通过预编译包来安装。另外，安装完成后可以运行 [check_installation.py](https://github.com/open-mmlab/mmcv/.dev_scripts/check_installation.py) 脚本检查 mmcv-full 是否安装成功。

i. 安装最新版本

如下是安装最新版 `mmcv-full` 的命令

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

请将链接中的 `{cu_version}` 和 `{torch_version}` 根据自身需求替换成实际的版本号，例如想安装和 `CUDA 11.1`、`PyTorch 1.9.0` 兼容的最新版 `mmcv-full`，使用如下替换过的命令

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

```{note}
PyTorch 在 1.x.0 和 1.x.1 之间通常是兼容的，故 mmcv-full 只提供 1.x.0 的编译包。如果你
的 PyTorch 版本是 1.x.1，你可以放心地安装在 1.x.0 版本编译的 mmcv-full。例如，如果你的
PyTorch 版本是 1.8.1、CUDA 版本是 11.1，你可以使用以下命令安装 mmcv-full。

`pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html`
```

如果想知道更多 CUDA 和 PyTorch 版本的命令，可以参考下面的表格，将链接中的 `=={mmcv_version}` 删去即可。

ii. 安装特定的版本

如下是安装特定版本 `mmcv-full` 的命令

```shell
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

首先请参考版本发布信息找到想要安装的版本号，将 `{mmcv_version}` 替换成该版本号，例如 `1.3.9`。
然后将链接中的 `{cu_version}` 和 `{torch_version}` 根据自身需求替换成实际的版本号，例如想安装和 `CUDA 11.1`、`PyTorch 1.9.0` 兼容的 `mmcv-full` 1.3.9 版本，使用如下替换过的命令

```shell
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

对于更多的 PyTorch 和 CUDA 版本组合，请参考下表：

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> CUDA </th>
      <th valign="bottom" align="left" width="120">torch 1.11</th>
      <th valign="bottom" align="left" width="120">torch 1.10</th>
      <th valign="bottom" align="left" width="120">torch 1.9</th>
      <th valign="bottom" align="left" width="120">torch 1.8</th>
      <th valign="bottom" align="left" width="120">torch 1.7</th>
      <th valign="bottom" align="left" width="120">torch 1.6</th>
      <th valign="bottom" align="left" width="120">torch 1.5</th>
    </tr>
    <tr>
      <td align="left">11.5</td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"></td>
      <td align="left"></code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">11.3</td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html</code></pre> </details> </td>
      <td align="left"></td>
      <td align="left"></code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">11.1</td>
      <td align="left"> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">11.0</td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.2</td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html</code></pre> </details></td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">10.1</td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">9.2</td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.5.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">cpu</td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.11.0/index.html</code></pre> </details></td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html</code></pre> </details> </td>
       <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> 安装 </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.5.0/index.html</code></pre> </details> </td>
    </tr>
  </tbody>
</table>

```{note}
以上提供的预编译包并不囊括所有的 mmcv-full 版本，我们可以点击对应链接查看支持的版本。例如，点击 [cu102-torch1.8.0](https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html)，可以看到 `cu102-torch1.8.0` 只提供了 1.3.0 及以上的 mmcv-full 版本。另外，从 `mmcv v1.3.17` 开始，我们不再提供`PyTorch 1.3 & 1.4` 对应的 mmcv-full 预编译包。你可以在 [这](./previous_versions.md) 找到 `PyTorch 1.3 & 1.4` 对应的预编包。虽然我们不再提供 `PyTorch 1.3 & 1.4` 对应的预编译包，但是我们依然在 CI 中保证对它们的兼容持续到下一年。
```

```{note}
mmcv-full 没有提供 Windows 平台 `cu102-torch1.8.0` 和 `cu92-torch*` 的预编译包。
```

除了使用预编译包之外，另一种方式是在本地进行编译，直接运行下述命令

```python
pip install mmcv-full
```

但注意本地编译可能会耗时 10 分钟以上。

b. 安装精简版

```python
pip install mmcv
```

c. 安装完整版并且编译 onnxruntime 的自定义算子

- 详细的指南请查看 [这里](https://mmcv.readthedocs.io/zh_CN/latest/deployment/onnxruntime_custom_ops.html)。

如果想从源码编译 MMCV，请参考[该文档](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html)。

## Installation

There are two versions of MMCV:

- **mmcv-full**: comprehensive, with full features and various CUDA ops out of box. It takes longer time to build.
- **mmcv**: lite, without CUDA ops but all other features, similar to mmcv<1.0.0. It is useful when you do not need those CUDA ops.

```{warning}
Do not install both versions in the same environment, otherwise you may encounter errors like `ModuleNotFound`. You need to uninstall one before installing the other. `Installing the full version is highly recommended if CUDA is avaliable`.
```

a. Install the full version.

Before installing mmcv-full, make sure that PyTorch has been successfully installed following the [official guide](https://pytorch.org/).

We provide pre-built mmcv packages (recommended) with different PyTorch and CUDA versions to simplify the building.

i. Install the latest version.

The rule for installing the latest ``mmcv-full`` is as follows:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired one. For example,
to install the latest ``mmcv-full`` with ``CUDA 11.1`` and ``PyTorch 1.9.0``, use the following command:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

For more details, please refer the the following tables and delete ``=={mmcv_version}``.

ii. Install a specified version.

The rule for installing a specified ``mmcv-full`` is as follows:

```shell
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

First of all, please refer to the Releases and replace ``{mmcv_version}`` a specified one. e.g. ``1.3.9``.
Then replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired versions. For example,
to install ``mmcv-full==1.3.9`` with ``CUDA 11.1`` and ``PyTorch 1.9.0``, use the following command:

```shell
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

```{tip}
mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility
usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you
can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well.
For example, if your PyTorch version is 1.8.1 and CUDA version is 11.1, you
can use the following command to install mmcv-full.

`pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html`
```

For more details, please refer the the following tables.

<table class="docutils">
  <tbody>
    <tr>
      <th width="80"> CUDA </th>
      <th valign="bottom" align="left" style="min-width: 100px">torch 1.9</th>
      <th valign="bottom" align="left" style="min-width: 100px">torch 1.8</th>
      <th valign="bottom" align="left" style="min-width: 100px">torch 1.7</th>
      <th valign="bottom" align="left" style="min-width: 100px">torch 1.6</th>
      <th valign="bottom" align="left" style="min-width: 100px">torch 1.5</th>
      <th valign="bottom" align="left" style="min-width: 100px">torch 1.4</th>
      <th valign="bottom" align="left" style="min-width: 100px">torch 1.3</th>
    </tr>
    <tr>
      <td align="left">11.1</td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">11.0</td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.2</td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"> </td>
      <td align="left"> </td>
    </tr>
    <tr>
      <td align="left">10.1</td>
      <td align="left"> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">9.2</td>
      <td align="left"> </td>
      <td align="left"> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
    <tr>
      <td align="left">cpu</td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.6.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code> pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.5.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.4.0/index.html</code></pre> </details> </td>
      <td align="left"><details><summary> install </summary><pre><code>pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.3.0/index.html</code></pre> </details> </td>
    </tr>
  </tbody>
</table>

Another way is to compile locally by running

```python
pip install mmcv-full
```

Note that the local compiling may take up to 10 mins.

b. Install the lite version.

```python
pip install mmcv
```

c. Install full version with custom operators for onnxruntime

- Check [here](https://mmcv.readthedocs.io/en/latest/deployment/onnxruntime_custom_ops.html) for detailed instruction.

If you would like to build MMCV from source, please refer to the [guide](https://mmcv.readthedocs.io/en/latest/get_started/build.html).

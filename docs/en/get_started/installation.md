## Installation

There are two versions of MMCV:

- **mmcv**: comprehensive, with full features and various CUDA ops out of box. It takes longer time to build.
- **mmcv-lite**: lite, without CUDA ops but all other features, similar to mmcv\<1.0.0. It is useful when you do not need those CUDA ops.

```{warning}
Do not install both versions in the same environment, otherwise you may encounter errors like `ModuleNotFound`. You need to uninstall one before installing the other. `Installing the full version is highly recommended if CUDA is avaliable`.
```

### Install mmcv

Before installing mmcv, make sure that PyTorch has been successfully installed following the [PyTorch official installation guide](https://pytorch.org/get-started/locally/#start-locally). This can be verified using the following command

```bash
python -c 'import torch;print(torch.__version__)'
```

If version information is output, then PyTorch is installed.

#### Install with mim (recommended)

[mim](https://github.com/open-mmlab/mim) is the package management tool for the OpenMMLab projects, which makes it easy to install mmcv

```bash
pip install -U openmim
mim install "mmcv>=2.0.0rc1"
```

If you find that the above installation command does not use a pre-built package ending with `.whl` but a source package ending with `.tar.gz`, you may not have a pre-build package corresponding to the PyTorch or CUDA or mmcv version, in which case you can [build mmcv from source](build.md).

<details>
<summary>Installation log using pre-built packages</summary>

Looking in links: https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
Collecting mmcv<br />
<b>Downloading https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/mmcv-2.0.0rc3-cp38-cp38-manylinux1_x86_64.whl</b>

</details>

<details>
<summary>Installation log using source packages</summary>

Looking in links: https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
Collecting mmcv==2.0.0rc3<br />
<b>Downloading mmcv-2.0.0rc3.tar.gz</b>

</details>

To install a specific version of mmcv, for example, mmcv version 2.0.0rc3, you can use the following command

```bash
mim install mmcv==2.0.0rc3
```

:::{note}
If you would like to use `opencv-python-headless` instead of `opencv-python`,
e.g., in a minimum container environment or servers without GUI,
you can first install it before installing MMCV to skip the installation of `opencv-python`.

Alternatively, if it takes too long to install a dependency library, you can specify the pypi source

```bash
mim install "mmcv>=2.0.0rc3" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

You can run [check_installation.py](https://github.com/open-mmlab/mmcv/blob/2.x/.dev_scripts/check_installation.py) to check the installation of mmcv-full after running the installation commands.

#### Install with pip

Use the following command to check the version of CUDA and PyTorch

```bash
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'
```

Select the appropriate installation command depending on the type of system, CUDA version, PyTorch version, and MMCV version

<html>
<body>
    <style>
      select {
          z-index: 1000;
          position: absolute;
          top: 10px;
          width: 6.7rem;
      }
      #select-container {
          position: relative;
          height: 30px;
      }
      #select-cmd {
          background-color: #f5f6f7;
          font-size: 14px;
          margin-top: 20px;
      }
      /* 让每一个都间隔1.3rem */
      #select-os {
          /* left: 1.375rem; */
          left: 0;
      }
      #select-cuda {
          /* left: 9.375rem;    9.375 = 1.375 + 6.7 + 1.3 */
          left: 8rem;
      }
      #select-torch {
          /* left: 17.375rem;    17.375 = 9.375 + 6.7 + 1.3 */
          left: 16rem;
      }
      #select-mmcv {
          /* left: 25.375rem;    25.375 = 17.375 + 6.7 + 1.3 */
          left: 24rem;
      }
    </style>
    <div id="select-container">
        <select
            onmousedown="handleSelectMouseDown(this.id)"
            onblur="handleSelectBlur(this.id)"
            onchange="changeOS(this.value)"
            id="select-os">
        </select>
        <select
            onmousedown="handleSelectMouseDown(this.id)"
            onblur="handleSelectBlur(this.id)"
            onchange="changeCUDA(this.value)"
            id="select-cuda">
        </select>
        <select
            onmousedown="handleSelectMouseDown(this.id)"
            onblur="handleSelectBlur(this.id)"
            onchange="changeTorch(this.value)"
            id="select-torch">
        </select>
        <select
            onmousedown="handleSelectMouseDown(this.id)"
            onblur="handleSelectBlur(this.id)"
            onchange="changeMMCV(this.value)"
            id="select-mmcv">
        </select>
    </div>
    <pre id="select-cmd"></pre>
</body>
<script>
    let osVal, cudaVal, torchVal, mmcvVal;
    function changeMMCV(val) {
        mmcvVal = val;
        change("select-mmcv");
    }
    function changeTorch(val) {
        torchVal = val;
        change("select-torch");
    }
    function changeCUDA(val) {
        cudaVal = val;
        change("select-cuda");
    }
    function changeOS(val) {
        osVal = val;
        change("select-os");
    }
    function handleSelectMouseDown(id) {
        const dom = document.getElementById(id);
        if (!dom) return;
        const len = dom?.options?.length;
        if (len >= 9) {
            dom.size = 10;
            dom.style.zIndex = 100;
        }
    }
    function handleSelectClick() {
        const selects = Array.from(document.getElementsByTagName("select"));
        selects.forEach(select => {
            select.size = 1;
        });
    }
    function handleSelectBlur(id) {
        const dom = document.getElementById(id);
        if (!dom) {
            handleSelectClick();
            return;
        }
        dom.size = 1;
        dom.style.zIndex = 1;
    }
    function changeCmd() {
        const cmd = document.getElementById("select-cmd");
        let cmdString = "pip install mmcv=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html";
        // e.g: pip install mmcv==2.0.0rc1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
        let cudaVersion;
        if (cudaVal === "cpu" || cudaVal === "mps") {
            cudaVersion = "cpu";
        } else {
            cudaVersion = `cu${cudaVal.split(".").join("")}`;
        }
        const torchVersion = `torch${torchVal.substring(0, torchVal.length - 2)}`;
        cmdString = cmdString.replace("{cu_version}", cudaVersion).replace("{mmcv_version}", mmcvVal).replace("{torch_version}", torchVersion);
        cmd.textContent = cmdString;
    }
    function unique(arr) {
        if (!arr || !Array.isArray(arr)) return [];
        return [...new Set(arr)];
    }
    function genOptionFragment(data, id) {
        const name = id.includes("-")? id.split("-")[1] : id;
        const fragment = new DocumentFragment();
        data.forEach(option => {
            const ele = document.createElement("option");
            let text = `${name} ${option}`;
            if (name === "os" || option.toUpperCase() === "CPU" || option.toUpperCase() === "MPS") {
                text = `${option}`;
            }
            ele.textContent = text;
            ele.value = option;
            ele.addEventListener('click', handleSelectClick);
            fragment.appendChild(ele);
        });
        return fragment;
    }
    function findAndAppend(data, id) {
        const fragment = genOptionFragment(data, id);
        const dom = document.getElementById(id);
        if (dom) dom.replaceChildren(fragment);
    }
    function change(id) {
        const order = ["select-mmcv", "select-torch", "select-cuda", "select-os"];
        const idx = order.indexOf(id);
        if (idx === -1) return;
        const versionDetail = version[osVal];
        if (idx >= 3) {
            let cuda = [];
            versionDetail.forEach(v => {
                cuda.push(v.cuda);
            });
            cuda = unique(cuda);
            cudaVal = cuda[0];
            findAndAppend(cuda, "select-cuda");
        }
        if (idx >= 2) {
            const torch = [];
            versionDetail.forEach(v => {
                if (v.cuda === cudaVal) torch.push(v.torch);
            });
            torchVal = torch[0];
            findAndAppend(torch, "select-torch");
        }
        if (idx >= 1) {
            let mmcv = [];
            versionDetail.forEach(v => {
                if (v.cuda === cudaVal && v.torch === torchVal) mmcv = v.mmcv;
            });
            mmcvVal = mmcv[0];
            findAndAppend(mmcv, "select-mmcv");
        }
        changeCmd();
    }
    function init() {
        document.addEventListener("click", handleSelectBlur);
        const version = window.version;
        const os = Object.keys(version);
        osVal = os[0];
        findAndAppend(os, "select-os");
        change("select-os");
        changeCmd();
    }
    window.onload = function () {
        const url = "../_static/version.json"
        const request = new XMLHttpRequest();
        request.open("get", url);
        request.send(null);
        request.onload = function () {
            if (request.status !== 200) return;
            const data = JSON.parse(request.responseText);
            window.version = data;
            init();
        }
    }
</script>
</html>

If you do not find a corresponding version in the dropdown box above, you probably do not have a pre-built package corresponding to the PyTorch or CUDA or mmcv version, at which point you can [build mmcv from source](build.md).

:::{note}
mmcv is only compiled on PyTorch 1.x.0 because the compatibility
usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you
can install mmcv compiled with PyTorch 1.x.0 and it usually works well.
For example, if your PyTorch version is 1.8.1, you can feel free to choose 1.8.x.
:::

:::{note}
If you would like to use `opencv-python-headless` instead of `opencv-python`,
e.g., in a minimum container environment or servers without GUI,
you can first install it before installing MMCV to skip the installation of `opencv-python`.

Alternatively, if it takes too long to install a dependency library, you can specify the pypi source

```bash
mim install "mmcv>=2.0.0rc1" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

You can run [check_installation.py](https://github.com/open-mmlab/mmcv/blob/2.x/.dev_scripts/check_installation.py) to check the installation of mmcv after running the installation commands.

#### Using mmcv with Docker

Build with local repository

```bash
git clone https://github.com/open-mmlab/mmcv.git && cd mmcv
docker build -t mmcv -f docker/release/Dockerfile .
```

Or build with remote repository

```bash
docker build -t mmcv https://github.com/open-mmlab/mmcv.git#2.x:docker/release
```

The [Dockerfile](release/Dockerfile) installs latest released version of mmcv-full by default, but you can specify mmcv versions to install expected versions.

```bash
docker image build -t mmcv -f docker/release/Dockerfile --build-arg MMCV=2.0.0rc1 .
```

If you also want to use other versions of PyTorch and CUDA, you can also pass them when building docker images.

An example to build an image with PyTorch 1.11 and CUDA 11.3.

```bash
docker build -t mmcv -f docker/release/Dockerfile \
    --build-arg PYTORCH=1.11.0 \
    --build-arg CUDA=11.3 \
    --build-arg CUDNN=8 \
    --build-arg MMCV=2.0.0rc1 .
```

More available versions of PyTorch and CUDA can be found at [dockerhub/pytorch](https://hub.docker.com/r/pytorch/pytorch/tags).

### Install mmcv-lite

If you need to use PyTorch-related modules, make sure PyTorch has been successfully installed in your environment by referring to the [PyTorch official installation guide](https://github.com/pytorch/pytorch#installation).

```python
pip install mmcv-lite
```

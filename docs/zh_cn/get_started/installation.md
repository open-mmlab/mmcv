## 安装 MMCV

MMCV 有两个版本：

- **mmcv**: 完整版，包含所有的特性以及丰富的开箱即用的 CPU 和 CUDA 算子。注意，完整版本可能需要更长时间来编译。
- **mmcv-lite**: 精简版，不包含 CPU 和 CUDA 算子但包含其余所有特性和功能，类似 MMCV 1.0 之前的版本。如果你不需要使用算子的话，精简版可以作为一个考虑选项。

```{warning}
请不要在同一个环境中安装两个版本，否则可能会遇到类似 `ModuleNotFound` 的错误。在安装一个版本之前，需要先卸载另一个。`如果 CUDA 可用，强烈推荐安装 mmcv`。
```

### 安装 mmcv

在安装 mmcv 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://pytorch.org/get-started/locally/#start-locally)。可使用以下命令验证

```bash
python -c 'import torch;print(torch.__version__)'
```

如果输出版本信息，则表示 PyTorch 已安装。

#### 使用 mim 安装（推荐）

[mim](https://github.com/open-mmlab/mim) 是 OpenMMLab 项目的包管理工具，使用它可以很方便地安装 mmcv。

```bash
pip install -U openmim
mim install "mmcv>=2.0.0rc1"
```

如果发现上述的安装命令没有使用预编译包（以 `.whl` 结尾）而是使用源码包（以 `.tar.gz` 结尾）安装，则有可能是我们没有提供和当前环境的 PyTorch 版本、CUDA 版本相匹配的 mmcv 预编译包，此时，你可以[源码安装 mmcv](build.md)。

<details>
<summary>使用预编译包的安装日志</summary>

Looking in links: https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
Collecting mmcv<br />
<b>Downloading https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/mmcv-2.0.0rc3-cp38-cp38-manylinux1_x86_64.whl</b>

</details>

<details>
<summary>使用源码包的安装日志</summary>

Looking in links: https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
Collecting mmcv==2.0.0rc3<br />
<b>Downloading mmcv-2.0.0rc3.tar.gz</b>

</details>

如需安装指定版本的 mmcv，例如安装 2.0.0rc3 版本的 mmcv，可使用以下命令

```bash
mim install mmcv==2.0.0rc3
```

:::{note}
如果你打算使用 `opencv-python-headless` 而不是 `opencv-python`，例如在一个很小的容器环境或者没有图形用户界面的服务器中，你可以先安装 `opencv-python-headless`，这样在安装 mmcv 依赖的过程中会跳过 `opencv-python`。

另外，如果安装依赖库的时间过长，可以指定 pypi 源

```bash
mim install "mmcv>=2.0.0rc1" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

安装完成后可以运行 [check_installation.py](https://github.com/open-mmlab/mmcv/blob/2.x/.dev_scripts/check_installation.py) 脚本检查 mmcv 是否安装成功。

#### 使用 pip 安装

使用以下命令查看 CUDA 和 PyTorch 的版本

```bash
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'
```

根据系统的类型、CUDA 版本、PyTorch 版本以及 MMCV 版本选择相应的安装命令

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
    // 各个select当前的值
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
    // 控制size大小相关的几个方法
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
            // 如果没有指定特定的id，那就直接把所有的select都设置成size = 1
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
    // string数组去重
    function unique(arr) {
        if (!arr || !Array.isArray(arr)) return [];
        return [...new Set(arr)];
    }
    // 根据string数组生成option的DocumentFragment
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
            // 添加value属性，方便下拉框选择时直接读到数据
            ele.value = option;
            // 添加点击事件监听
            ele.addEventListener('click', handleSelectClick);
            fragment.appendChild(ele);
        });
        return fragment;
    }
    // 在dom树中找到id对应的dom（select元素），并将生成的options添加到元素内
    function findAndAppend(data, id) {
        const fragment = genOptionFragment(data, id);
        const dom = document.getElementById(id);
        if (dom) dom.replaceChildren(fragment);
    }
    /**
     * change方法的重点在于
     * 1. 各个下拉框数据的联动
     *      OS ==> cuda ==> torch ==> mmcv
     * 2. 命令行的修改
    */
    function change(id) {
        const order = ["select-mmcv", "select-torch", "select-cuda", "select-os"];
        const idx = order.indexOf(id);
        if (idx === -1) return;
        const versionDetail = version[osVal];
        if (idx >= 3) {
            // 根据os修改cuda
            let cuda = [];
            versionDetail.forEach(v => {
                cuda.push(v.cuda);
            });
            cuda = unique(cuda);
            cudaVal = cuda[0];
            findAndAppend(cuda, "select-cuda");
        }
        if (idx >= 2) {
            // 根据cuda修改torch
            const torch = [];
            versionDetail.forEach(v => {
                if (v.cuda === cudaVal) torch.push(v.torch);
            });
            torchVal = torch[0];
            findAndAppend(torch, "select-torch");
        }
        if (idx >= 1) {
            // 根据torch修改mmcv
            let mmcv = [];
            versionDetail.forEach(v => {
                if (v.cuda === cudaVal && v.torch === torchVal) mmcv = v.mmcv;
            });
            mmcvVal = mmcv[0];
            findAndAppend(mmcv, "select-mmcv");
        }
        changeCmd();
    }
    // 初始化，处理version数据，并调用findAndAppend
    function init() {
        // 增加一个全局的click事件监听，作为select onBlur事件失效的兜底
        document.addEventListener("click", handleSelectBlur);
        const version = window.version;
        // OS
        const os = Object.keys(version);
        osVal = os[0];
        findAndAppend(os, "select-os");
        change("select-os");
        changeCmd();
    }
    // 利用xhr获取本地version数据，如果作为html直接浏览的话需要使用本地服务器打开，否则会有跨域问题
    window.onload = function () {
        const url = "../_static/version.json"
        // 申明一个XMLHttpRequest
        const request = new XMLHttpRequest();
        // 设置请求方法与路径
        request.open("get", url);
        // 不发送数据到服务器
        request.send(null);
        //XHR对象获取到返回信息后执行
        request.onload = function () {
            // 返回状态为200，即为数据获取成功
            if (request.status !== 200) return;
            const data = JSON.parse(request.responseText);
            window.version = data;
            init();
        }
    }
</script>
</html>

如果在上面的下拉框中没有找到对应的版本，则可能是没有对应 PyTorch 或者 CUDA 或者 mmcv 版本的预编译包，此时，你可以[源码安装 mmcv](build.md)。

:::{note}
PyTorch 在 1.x.0 和 1.x.1 之间通常是兼容的，故 mmcv 只提供 1.x.0 的编译包。如果你
的 PyTorch 版本是 1.x.1，你可以放心地安装在 1.x.0 版本编译的 mmcv。例如，如果你的
PyTorch 版本是 1.8.1，你可以放心选择 1.8.x。
:::

:::{note}
如果你打算使用 `opencv-python-headless` 而不是 `opencv-python`，例如在一个很小的容器环境或者没有图形用户界面的服务器中，你可以先安装 `opencv-python-headless`，这样在安装 mmcv 依赖的过程中会跳过 `opencv-python`。

另外，如果安装依赖库的时间过长，可以指定 pypi 源

```bash
pip install "mmcv>=2.0.0rc1" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html -i https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

安装完成后可以运行 [check_installation.py](https://github.com/open-mmlab/mmcv/blob/2.x/.dev_scripts/check_installation.py) 脚本检查 mmcv 是否安装成功。

#### 使用 docker 镜像

先将算法库克隆到本地再构建镜像

```bash
git clone https://github.com/open-mmlab/mmcv.git && cd mmcv
docker build -t mmcv -f docker/release/Dockerfile .
```

也可以直接使用下面的命令构建镜像

```bash
docker build -t mmcv https://github.com/open-mmlab/mmcv.git#2.x:docker/release
```

[Dockerfile](release/Dockerfile) 默认安装最新的 mmcv，如果你想要指定版本，可以使用下面的命令

```bash
docker image build -t mmcv -f docker/release/Dockerfile --build-arg MMCV=2.0.0rc1 .
```

如果你想要使用其他版本的 PyTorch 和 CUDA，你可以在构建镜像时指定它们的版本。

例如指定 PyTorch 的版本是 1.11，CUDA 的版本是 11.3

```bash
docker build -t mmcv -f docker/release/Dockerfile \
    --build-arg PYTORCH=1.11.0 \
    --build-arg CUDA=11.3 \
    --build-arg CUDNN=8 \
    --build-arg MMCV=2.0.0rc1 .
```

更多 PyTorch 和 CUDA 镜像可以点击 [dockerhub/pytorch](https://hub.docker.com/r/pytorch/pytorch/tags) 查看。

### 安装 mmcv-lite

如果你需要使用和 PyTorch 相关的模块，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://pytorch.org/get-started/locally/#start-locally)。

```python
pip install mmcv-lite
```

def docker_images = ["registry.cn-hangzhou.aliyuncs.com/sensetime/openmmlab:cuda10.1-cudnn7-devel-ubuntu18.04-py37-pt1.3",
                     "registry.cn-hangzhou.aliyuncs.com/sensetime/openmmlab:cuda10.2-cudnn7-devel-ubuntu18.04-py37-pt1.5"]
def torch_versions = ["1.3.0", "1.5.0"]
def torchvision_versions = ["0.4.2", "0.6.0"]


def get_stages(docker_image, folder) {
    def pip_mirror = "-i https://mirrors.aliyun.com/pypi/simple"
    stages = {
        docker.image(docker_image).inside('-u root --gpus all --net host') {
            sh "rm -rf ${env.WORKSPACE}-${folder} ${env.WORKSPACE}-${folder}@tmp"
            sh "cp -r ${env.WORKSPACE} ${env.WORKSPACE}-${folder}"
            try {
                dir("${env.WORKSPACE}-${folder}") {
                    stage("before_install") {
                        sh "apt-get update && apt-get install -y ninja-build"
                    }
                    stage("dependencies") {
                        // torch and torchvision are pre-installed in dockers
                        sh "pip list | grep torch"
                        sh "apt-get install -y ffmpeg libturbojpeg"
                        sh "pip install pytest coverage lmdb PyTurboJPEG Cython ${pip_mirror}"
                    }
                    stage("build") {
                        sh "MMCV_WITH_OPS=1 pip install -e . ${pip_mirror}"
                    }
                    stage("test") {
                        sh "coverage run --branch --source=mmcv -m pytest tests/"
                        sh "coverage xml"
                        sh "coverage report -m"
                    }
                }
            } finally {
                sh "rm -rf ${env.WORKSPACE}-${folder} ${env.WORKSPACE}-${folder}@tmp"
            }
        }
    }
    return stages
}


node('master') {
    // fetch latest change from SCM (Source Control Management)
    checkout scm

    def stages = [:]
    for (int i = 0; i < docker_images.size(); i++) {
        def docker_image = docker_images[i]
        def torch = torch_versions[i]
        def torchvision = torchvision_versions[i]
        def tag = docker_image + '_' + torch + '_' + torchvision
        def folder = "${i}"
        stages[tag] = get_stages(docker_image, folder)
    }
    parallel stages
}

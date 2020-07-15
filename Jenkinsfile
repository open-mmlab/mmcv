def docker_images = ["hejm37/python-envs:cuda10.1-cudnn7-devel-ubuntu18.04-py37"]
def torch_versions = ["1.3.0", "1.5.0"]
def torchvision_versions = ["0.4.2", "0.6.0"]


def get_stages(docker_image, torch, torchvision, folder) {
    def pip_mirror = "-i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com"
    stages = {
        docker.image(docker_image).inside('-u root --gpus all') {
            sh "cp -r ${env.WORKSPACE} ${env.WORKSPACE}-${folder}"
            dir("${env.WORKSPACE}-${folder}") {
                stage("before_install") {
                    sh "apt-get update && apt-get install -y ninja-build"
                }
                stage("dependencies") {
                    if (torchvision == "0.4.2") {
                        sh "pip install Pillow==6.2.2 ${pip_mirror}"
                    }
                    sh "pip install pip install torch==${torch} torchvision==${torchvision} ${pip_mirror}"
                    sh "apt-get install -y ffmpeg libturbojpeg"
                    sh "pip install pytest coverage lmdb PyTurboJPEG ${pip_mirror}"
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
        for (int j = 0; j < torch_versions.size(); j++) {
            def torch = torch_versions[j]
            def torchvision = torchvision_versions[j]
            def tag = docker_image + '_' + torch + '_' + torchvision
            def folder = "${i}-${j}"
            stages[tag] = get_stages(docker_image, torch, torchvision, folder)
        }
    }
    parallel stages
}

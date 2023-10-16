# adv_rl

For dev work:

```bash
pre-commit install
```

# docker

### dev container

Download the devcontainers vscode extension: https://code.visualstudio.com/docs/devcontainers/containers (or just in vscode extensions pane)

This is setup to run with devcontainers. Upon opening the window with this folder, you should get prompted to launch with devcontainers.
Click ok and the image build will start (takes a while on the first build). Once that completes, you will be attached to the docker container.
Dev workflow is the same as if you were coding locally. Make sure to push your changes to a branch before killing though, as it is ephemeral.

Note: make sure you are not using docker desktop context. `docker context use default` before running devcontainers due to incompatibility with nvidia cuda and devcontainers.


### official docker way
```bash
sudo docker build -t adv_rl -f Dockerfile .
sudo docker run --gpus all adv_rl nvidia-smi
```

So you just need to attach a script to the run command for training, e.g., assume we have `train.py` with arg passthrough:
```bash
sudo docker run --gpus all adv_rl python train.py --epochs 10 --batch-size 32
```

You can see this has gpu acceleration. May need to download nvidia container toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

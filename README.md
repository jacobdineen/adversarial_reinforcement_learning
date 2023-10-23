# adv_rl

For dev work:

```bash
pre-commit install
```

Conda local dev:

```python
conda create -n adv_rl python=3.10
conda activate adv_rl
pip install -r docker_requirements.txt
pip install -e .  # install this module in editable mode
```


# Common commands
```python
# train resnet on cifar 10
python src/Resnet_18_train.py

# testing inference on trained resnet model
python src/Resnet_18_inference.py

# dry run of env
python src/env.py


# train rl agent
python src/train.py


```



# docker

```bash
sudo docker build -t adv_rl -f Dockerfile .
# if on a gpu machine, you can run this
# although the devcontainer should work fine if you have that capability
sudo docker run --gpus all adv_rl nvidia-smi

# else you can run this and then attach to the running container
docker run -it adv_rl /bin/bash
```

So you just need to attach a script to the run command for training, e.g., assume we have `train.py` with arg passthrough:
```bash
sudo docker run --gpus all adv_rl python train.py --epochs 10 --batch-size 32
```

### using dev containers
You can spawn a container like this: `docker run -it adv_rl /bin/bash`
and then attach vscode instance to the running container and dev that way



#### Nvidia container toolkit
You can see this has gpu acceleration. May need to download nvidia container toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    vim \
    zsh \
    curl \
    git \
    sudo \
    python3-pip \
    python3-dev

# Copy the requirements file into the container
COPY docker_requirements.txt .

# Install Python3 and Pip
RUN apt-get install -y python3-pip

# Install the packages from the requirements file
RUN pip3 install --no-cache-dir -r docker_requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Copy the rest of the application code into the container
COPY . .

# Install ZSH plugins
RUN sh -c "$(curl -L https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p https://github.com/zsh-users/zsh-completions

# Set the command to run your application
# eventually want entrpoint
CMD ["zsh"]

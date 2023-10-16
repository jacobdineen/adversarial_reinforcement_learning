FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# Install basic utilities and software
RUN apt-get update && apt-get install -y \
    vim \
    zsh \
    curl \
    git \
    python3.10 \
    python3-pip

# RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
#   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
#     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
#   && \
#     sudo apt-get update

# RUN sudo apt-get install -y nvidia-container-toolkit

# Copy the requirements file into the container
COPY docker_requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r docker_requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Install ZSH plugins
RUN sh -c "$(curl -L https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p https://github.com/zsh-users/zsh-completions

# Set the command to run ZSH by default
CMD ["zsh"]

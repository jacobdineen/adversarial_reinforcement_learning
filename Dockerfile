FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    vim \
    zsh \
    curl \
    git


# Copy the requirements file into the container
COPY docker_requirements.txt .

# Install the packages from the requirements file
RUN pip3 install --no-cache-dir -r docker_requirements.txt
#&& pip3 install thefuck --user

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
CMD ["zsh"]

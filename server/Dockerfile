FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for pyenv
RUN apt-get update && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    git

# Install pyenv
RUN curl https://pyenv.run | bash

# Add pyenv to PATH
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install Python 3.10.14 and set as default
RUN pyenv install 3.10.14 && \
    pyenv global 3.10.14

# Verify Python version
RUN python --version

WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python -m pip install -U --pre oneflow -f https://github.com/siliconflow/oneflow_releases/releases/expanded_assets/community_cu121
RUN python -m pip install --pre onediff

EXPOSE 3000
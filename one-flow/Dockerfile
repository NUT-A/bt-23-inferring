FROM nvcr.io/nvidia/pytorch:24.06-py3 AS base

RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    nasm \
    g++ \
    gcc \
    python3-pip \
    cmake \
    autoconf \
    libtool \
    git \
    binutils-dev \
    libunwind-dev \
    ninja-build \
    libxml2-dev \
    libedit-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install OneFlow
RUN git clone https://github.com/Oneflow-Inc/oneflow.git /opt/oneflow

RUN mkdir /opt/oneflow/build
WORKDIR /opt/oneflow/build

ENV CMAKE_CXX_STANDARD=17

RUN cmake .. -C ../cmake/caches/international/cuda.cmake -DCMAKE_CUDA_ARCHITECTURES=89 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCUDNN_ROOT_DIR=/usr/include -DCMAKE_CXX_STANDARD=17 -DONEFLOW_CURRENT_VERSION=1.0.0 -DRPC_BACKEND=LOCAL -DUSE_CLANG_FORMAT=ON -DCUDA_DRIVER_LIBRARY=/usr/lib/x86_64-linux-gnu/libcuda.so -DCUDA_RUNTIME_LIBRARY=/usr/lib/x86_64-linux-gnu/libcuda.so
# RUN make VERBOSE=1 -j$(nproc)
# RUN make install

# ENV PYTHONPATH=/opt/oneflow/python:${PYTHONPATH}

# # Install OneFlow Python
# RUN pip3 install -e /opt/oneflow/python

# RUN mkdir /workspace
# WORKDIR /workspace

# COPY requirements.txt .
# RUN pip3 install -r requirements.txt
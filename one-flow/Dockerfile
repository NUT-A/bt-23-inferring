FROM nvcr.io/nvidia/pytorch:23.07-py3 AS base

RUN apt-get update && apt-get install -y libopenblas-dev nasm g++ gcc python3-pip cmake autoconf libtool git binutils-dev libxml2-dev libedit-dev clang-format
# RUN apt-get update && apt-get install -y libopenblas-dev nasm g++ gcc python3-pip cmake autoconf libtool git binutils-dev libxml2-dev libedit-dev libc++-dev libc++abi-dev clang-format

# Install OneFlow
RUN git clone -b release/v0.9.0 https://github.com/Oneflow-Inc/oneflow.git /opt/oneflow

WORKDIR /opt/oneflow

# Replace xorg-macros download URL
RUN sed -i 's|https://github.com/freedesktop/xorg-macros/archive/refs/tags/util-macros-1.19.1.tar.gz|https://github.com/tech4me/xorg-macros/archive/refs/tags/util-macros-1.19.1.tar.gz|g' \
    $(find /opt/oneflow -type f)

# Replace xorg-libpciaccess download URL
RUN sed -i 's|https://github.com/freedesktop/xorg-libpciaccess/archive/refs/tags/libpciaccess-0.16.tar.gz|https://github.com/gitlab-freedesktop-mirrors/libpciaccess/archive/refs/tags/libpciaccess-0.16.tar.gz|g' \
    $(find /opt/oneflow -type f)

# Replace hash in third_party/re2/CMakeLists.txt
RUN sed -i 's/92e2b604e294a9160bc977c000507340/93554c189796c27dfc72af17a367a0b4/g' \
    $(find /opt/oneflow -type f)

RUN find /opt/oneflow -type f \( -name "*.cmake" -o -name "CMakeLists.txt" \) -print0 | xargs -0 sed -i 's/-Werror //g'

RUN mkdir build
WORKDIR /opt/oneflow/build

ENV CMAKE_CXX_STANDARD=14
ENV CXXFLAGS="-fPIC"

RUN cmake .. -C ../cmake/caches/international/cuda.cmake -DCMAKE_CUDA_ARCHITECTURES=89 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCUDNN_ROOT_DIR=/usr/include -DCMAKE_CXX_STANDARD=17 -DONEFLOW_CURRENT_VERSION=0.9.1 -DRPC_BACKEND=LOCAL -DCUDA_DRIVER_LIBRARY=/usr/lib/x86_64-linux-gnu/libcuda.so -DCUDA_RUNTIME_LIBRARY=/usr/lib/x86_64-linux-gnu/libcuda.so -DTREAT_WARNINGS_AS_ERRORS=OFF -DWITH_MLIR=ON -DWITH_MLIR_CUDA_CODEGEN=ON -DCXXFLAGS="-fPIC"
RUN chmod +x .cache/bin/clang-format || true

# Remove -Werror flag from all files
RUN find /opt/oneflow/build -type f -print0 | xargs -0 sed -i 's/-Werror //g'


# RUN make VERBOSE=1 -j$(nproc)

# ENV PYTHONPATH=/opt/oneflow/python:${PYTHONPATH}

# # # Install OneFlow Python
# RUN pip3 install -e /opt/oneflow/python

# # RUN mkdir /workspace
# WORKDIR /workspace

# COPY requirements.txt .
# RUN pip3 install -r requirements.txt
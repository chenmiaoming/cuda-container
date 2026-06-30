FROM docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TVM_REF=v0.25.0
ARG PYTORCH_CUDA=cu130
ARG VLLM_VERSION=0.23.0
ARG SGLANG_VERSION=0.5.13.post1
ARG PYTHON_VERSION=3.12

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_INDEX_URL=https://download.pytorch.org/whl/${PYTORCH_CUDA}
ENV TVM_HOME=/opt/tvm
ENV TVM_LIBRARY_PATH=/opt/tvm/build/lib
ENV LLVM_CONFIG=/opt/llvm/bin/llvm-config
ENV LD_LIBRARY_PATH=/opt/tvm/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=${CONDA_DIR}/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ---------- system packages (no python) ----------
RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
    openssh-server \
    gdb \
    lldb \
    wget \
    bzip2 \
    ca-certificates \
    gnupg \
    git \
    tini \
    build-essential \
    cmake \
    ninja-build \
    libedit-dev \
    libtinfo-dev \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libopenblas-dev \
    libxml2-dev \
    pkg-config \
    skopeo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- nsight systems ----------
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget gnupg && \
    . /etc/os-release && \
    UBUNTU_VER="$(echo "${VERSION_ID}" | tr -d '.')" && \
    ARCH="$(dpkg --print-architecture)" && \
    wget -qO- "https://developer.download.nvidia.com/devtools/repos/ubuntu${UBUNTU_VER}/${ARCH}/nvidia.pub" | gpg --dearmor > /usr/share/keyrings/nvidia-devtools.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu${UBUNTU_VER}/${ARCH}/ /" > /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-get update && \
    apt-cache policy nsight-systems-cli nsight-systems && \
    apt-get install -y --no-install-recommends nsight-systems-cli && \
    rm -rf /var/lib/apt/lists/*

# ---------- miniforge (base env with python for build tools like LLVM) ----------
RUN wget --no-hsts --quiet https://mirrors.ustc.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    printf "channels:\n  - conda-forge\nmirrored_channels:\n  conda-forge:\n    - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge\n" > ${CONDA_DIR}/.condarc && \
    conda install -y python=${PYTHON_VERSION} && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes

# ---------- LLVM (shared across envs, used by TVM) ----------
RUN git clone --branch release/21.x --depth 1 https://github.com/llvm/llvm-project.git /opt/llvm-project && \
    cmake -S /opt/llvm-project/llvm -B /opt/llvm-project/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DLLVM_ENABLE_ZLIB=ON \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm && \
    cmake --build /opt/llvm-project/build -j $(nproc) && \
    cmake --install /opt/llvm-project/build && \
    rm -rf /opt/llvm-project

# ---------- TVM source + build ----------
RUN git clone --branch ${TVM_REF} --depth 1 https://github.com/apache/tvm.git ${TVM_HOME} && \
    cd ${TVM_HOME} && \
    git submodule update --init --recursive

RUN cmake -S ${TVM_HOME} -B ${TVM_HOME}/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DUSE_CUDNN=ON \
    -DUSE_CUBLAS=ON \
    -DUSE_LLVM=${LLVM_CONFIG} \
    -DUSE_RPC=ON && \
    cmake --build ${TVM_HOME}/build -j $(nproc) && \
    test -f ${TVM_HOME}/build/lib/libtvm_compiler.so && \
    test -f ${TVM_HOME}/build/lib/libtvm_runtime.so && \
    test -f ${TVM_HOME}/build/lib/libtvm_runtime_cuda.so && \
    test -f ${TVM_HOME}/build/lib/libtvm_runtime_extra.so

# ---------- conda env: tvm ----------
RUN conda create -y -n tvm python=${PYTHON_VERSION} pip && \
    conda run -n tvm --no-capture-output pip install --upgrade pip setuptools wheel && \
    conda run -n tvm --no-capture-output pip install \
    numpy cython tornado psutil 'xgboost>=1.1.0' cloudpickle && \
    conda run -n tvm --no-capture-output pip install --index-url ${PYTORCH_INDEX_URL} \
    torch torchvision torchaudio && \
    conda run -n tvm --no-capture-output pip install ${TVM_HOME}/3rdparty/tvm-ffi && \
    conda run -n tvm --no-capture-output pip install -e ${TVM_HOME} && \
    conda clean --all --yes

# ---------- conda env: vllm ----------
RUN conda create -y -n vllm python=${PYTHON_VERSION} pip && \
    conda run -n vllm --no-capture-output pip install --upgrade pip setuptools wheel uv && \
    conda run -n vllm --no-capture-output uv pip install \
    --python ${CONDA_DIR}/envs/vllm/bin/python \
    --torch-backend=cu130 \
    vllm==${VLLM_VERSION} && \
    conda run -n vllm --no-capture-output python -c "import torch, vllm; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('vllm:', vllm.__version__)" && \
    conda clean --all --yes


# ---------- conda env: sglang ----------
RUN conda create -y -n sglang python=${PYTHON_VERSION} pip && \
    conda run -n sglang --no-capture-output pip install --upgrade pip setuptools wheel uv && \
    conda run -n sglang --no-capture-output uv pip install \
    --python ${CONDA_DIR}/envs/sglang/bin/python \
    --torch-backend=cu130 \
    --prerelease=allow \
    sglang==${SGLANG_VERSION} && \
    conda run -n sglang --no-capture-output python -c "import torch, sglang; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('sglang:', sglang.__version__)" && \
    conda clean --all --yes

# ---------- SSH + shell setup ----------
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc && \
    echo "export PATH=${PATH}" >> /etc/profile && \
    mkdir -p /etc/ssh/ssh_config.d /etc/ssh/sshd_config.d && \
    printf "Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n" > /etc/ssh/ssh_config.d/99-aistation.conf && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    mkdir -p /var/run/sshd

# ---------- /etc/environment for SSH sessions ----------
RUN printf '%s\n' \
    'CUDA_VERSION=13.0.2' \
    'NVIDIA_DRIVER_CAPABILITIES=compute,utility' \
    'NVIDIA_PRODUCT_NAME=CUDA' \
    'NVARCH=x86_64' \
    'TVM_HOME=/opt/tvm' \
    'TVM_LIBRARY_PATH=/opt/tvm/build/lib' \
    'CONDA_DIR=/opt/conda' \
    'LLVM_CONFIG=/opt/llvm/bin/llvm-config' \
    'LD_LIBRARY_PATH=/opt/tvm/build/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib' \
    'PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
    > /etc/environment

EXPOSE 22

ENTRYPOINT ["tini", "--"]
CMD ["/usr/sbin/sshd", "-D", "-e"]

WORKDIR ${TVM_HOME}
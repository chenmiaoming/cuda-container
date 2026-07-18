FROM docker.io/nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_COMPAT_PACKAGE=cuda-compat-13-3
ARG PYTORCH_CUDA=cu130
ARG VLLM_VERSION=0.24.0
ARG SGLANG_VERSION=0.5.14
ARG PYTHON_VERSION=3.12
ARG FRP_VERSION=0.69.0

ENV CONDA_DIR=/opt/conda
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDA_COMPAT_PATH=/usr/local/cuda/compat

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV FLASHINFER_USE_CUDA_NORM=1

# cuda-compat must precede the host-injected NVIDIA driver libraries.
ENV LD_LIBRARY_PATH=${CUDA_COMPAT_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${CUDA_HOME}/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib
ENV PATH=${CONDA_DIR}/bin:/usr/local/nvidia/bin:${CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin


# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ${CUDA_COMPAT_PACKAGE} \
    openssh-server \
    gdb \
    lldb \
    wget \
    bzip2 \
    ca-certificates \
    gnupg \
    git \
    tini \
    tar \
    screen \
    build-essential \
    cmake \
    ninja-build \
    skopeo \
    libnuma1 \
    libnuma-dev \
    numactl \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------------------------
# CUDA forward-compatibility package validation
#
# This only checks files provided by the installed Debian package.
# It does not initialize CUDA and does not require a GPU.
# ---------------------------------------------------------------------------

RUN test -e ${CUDA_COMPAT_PATH}/libcuda.so.1 && \
    test -e ${CUDA_COMPAT_PATH}/libnvidia-ptxjitcompiler.so.1 && \
    dpkg-query -W -f='${Package} ${Version}\n' \
    ${CUDA_COMPAT_PACKAGE}


# ---------------------------------------------------------------------------
# frpc
#
# Install only. frpc is not started automatically.
#
# Manual example:
#   screen -dmS frpc frpc -c /etc/frp/frpc.toml
# ---------------------------------------------------------------------------

RUN set -eux; \
    archive="frp_${FRP_VERSION}_linux_amd64.tar.gz"; \
    directory="frp_${FRP_VERSION}_linux_amd64"; \
    wget --no-hsts \
    "https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/${archive}" \
    -O "/tmp/${archive}"; \
    tar -xzf "/tmp/${archive}" -C /tmp; \
    install -m 0755 \
    "/tmp/${directory}/frpc" \
    /usr/local/bin/frpc; \
    mkdir -p /etc/frp; \
    rm -rf \
    "/tmp/${archive}" \
    "/tmp/${directory}"; \
    frpc --version


# ---------------------------------------------------------------------------
# Nsight Systems
# ---------------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg && \
    . /etc/os-release && \
    UBUNTU_VER="$(echo "${VERSION_ID}" | tr -d '.')" && \
    ARCH="$(dpkg --print-architecture)" && \
    wget -qO- \
    "https://developer.download.nvidia.com/devtools/repos/ubuntu${UBUNTU_VER}/${ARCH}/nvidia.pub" \
    | gpg --dearmor \
    > /usr/share/keyrings/nvidia-devtools.gpg && \
    echo \
    "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu${UBUNTU_VER}/${ARCH}/ /" \
    > /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    nsight-systems-cli && \
    rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------------------------
# Miniforge
# ---------------------------------------------------------------------------

RUN wget --no-hsts --quiet \
    https://mirrors.ustc.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh \
    -b \
    -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    printf '%s\n' \
    'channels:' \
    '  - conda-forge' \
    'mirrored_channels:' \
    '  conda-forge:' \
    '    - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge' \
    > ${CONDA_DIR}/.condarc && \
    conda install -y \
    python=${PYTHON_VERSION} \
    pip && \
    conda clean \
    --tarballs \
    --index-cache \
    --packages \
    --yes && \
    find ${CONDA_DIR} \
    -follow \
    -type f \
    -name '*.a' \
    -delete && \
    find ${CONDA_DIR} \
    -follow \
    -type f \
    -name '*.pyc' \
    -delete && \
    conda clean \
    --force-pkgs-dirs \
    --all \
    --yes


# ---------------------------------------------------------------------------
# Conda environment: vLLM
#
# The build-time version check reads package metadata only.
# It does not import vLLM, initialize CUDA, or execute a GPU kernel.
# ---------------------------------------------------------------------------

RUN conda create -y \
    -n vllm \
    python=${PYTHON_VERSION} \
    pip && \
    conda run \
    -n vllm \
    --no-capture-output \
    pip install \
    --upgrade \
    pip \
    setuptools \
    wheel \
    uv && \
    conda run \
    -n vllm \
    --no-capture-output \
    uv pip install \
    --python ${CONDA_DIR}/envs/vllm/bin/python \
    --torch-backend=${PYTORCH_CUDA} \
    vllm==${VLLM_VERSION} && \
    conda run \
    -n vllm \
    --no-capture-output \
    python -c \
    "from importlib.metadata import version; print('torch:', version('torch')); print('vllm:', version('vllm'))" && \
    conda clean --all --yes


# ---------------------------------------------------------------------------
# Conda environment: SGLang
#
# The build-time version check reads package metadata only.
# It does not import SGLang, initialize CUDA, or execute a GPU kernel.
# ---------------------------------------------------------------------------

RUN conda create -y \
    -n sglang \
    python=${PYTHON_VERSION} \
    pip && \
    conda run \
    -n sglang \
    --no-capture-output \
    pip install \
    --upgrade \
    pip \
    setuptools \
    wheel \
    uv && \
    conda run \
    -n sglang \
    --no-capture-output \
    uv pip install \
    --python ${CONDA_DIR}/envs/sglang/bin/python \
    --torch-backend=${PYTORCH_CUDA} \
    --prerelease=allow \
    --upgrade \
    sglang==${SGLANG_VERSION} && \
    conda run \
    -n sglang \
    --no-capture-output \
    python -c \
    "from importlib.metadata import version; print('torch:', version('torch')); print('sglang:', version('sglang'))" && \
    conda clean --all --yes


# ---------------------------------------------------------------------------
# Shell environment
#
# Docker ENV applies to normal container processes.
# Interactive SSH shells source this file through .bashrc.
# /etc/environment and sshd SetEnv are not used.
# ---------------------------------------------------------------------------

RUN printf '%s\n' \
    'export CUDA_HOME=/usr/local/cuda' \
    'export CUDA_PATH=/usr/local/cuda' \
    'export CUDA_COMPAT_PATH=/usr/local/cuda/compat' \
    'export CONDA_DIR=/opt/conda' \
    'export LANG=C.UTF-8' \
    'export LC_ALL=C.UTF-8' \
    'export PIP_NO_CACHE_DIR=1' \
    'export PYTHONUNBUFFERED=1' \
    'export FLASHINFER_USE_CUDA_NORM=1' \
    'export PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}' \
    'export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' \
    > /etc/profile.d/aistation.sh && \
    chmod 0644 /etc/profile.d/aistation.sh && \
    printf '%s\n' \
    'if [ -f /etc/profile.d/aistation.sh ]; then' \
    '    . /etc/profile.d/aistation.sh' \
    'fi' \
    'if [ -f /opt/conda/etc/profile.d/conda.sh ]; then' \
    '    . /opt/conda/etc/profile.d/conda.sh' \
    '    conda activate base >/dev/null 2>&1 || true' \
    'fi' \
    >> /root/.bashrc && \
    printf '%s\n' \
    'if [ -f /etc/profile.d/aistation.sh ]; then' \
    '    . /etc/profile.d/aistation.sh' \
    'fi' \
    'if [ -f /opt/conda/etc/profile.d/conda.sh ]; then' \
    '    . /opt/conda/etc/profile.d/conda.sh' \
    '    conda activate base >/dev/null 2>&1 || true' \
    'fi' \
    >> /etc/skel/.bashrc


# ---------------------------------------------------------------------------
# SSH server
#
# 00 prefix ensures these values are read before other sshd_config snippets.
# The image does not initialize or overwrite the root password.
# ---------------------------------------------------------------------------

RUN mkdir -p \
    /etc/ssh/sshd_config.d \
    /run/sshd && \
    printf '%s\n' \
    'PermitRootLogin yes' \
    'PasswordAuthentication yes' \
    'PubkeyAuthentication yes' \
    'KbdInteractiveAuthentication yes' \
    'UsePAM yes' \
    > /etc/ssh/sshd_config.d/00-aistation.conf && 


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

EXPOSE 22

WORKDIR /workspace

ENTRYPOINT ["tini", "-g", "--"]

CMD ["/usr/sbin/sshd -D -e"]
FROM docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG FLAGTREE_VERSION=0.6.0
ARG PYTORCH_VERSION=2.10.0
ARG PYTORCH_CUDA=cu130

ENV VENV_DIR=/opt/flagtree-venv
ENV FLAGTREE_HOME=/opt/flagtree
ENV CUDA_HOME=/usr/local/cuda
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PATH=${VENV_DIR}/bin:${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# FlagTree is a Triton fork. PyTorch is installed first so that its matching
# runtime dependencies are retained; the upstream Triton wheel is then removed
# before FlagTree is built and installed in its place.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv ${VENV_DIR} && \
    python -m pip install --upgrade \
      pip \
      setuptools \
      wheel \
      uv \
      'pybind11>=2.13.1'

RUN uv pip install \
      --python ${VENV_DIR}/bin/python \
      --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA} \
      torch==${PYTORCH_VERSION} && \
    python -c "from importlib.metadata import version; assert version('triton') == '3.6.0', version('triton'); print('torch:', version('torch')); print('bundled triton:', version('triton'))" && \
    python -m pip uninstall -y triton

RUN git clone \
      --branch ${FLAGTREE_VERSION} \
      --depth 1 \
      --recurse-submodules \
      --shallow-submodules \
      https://github.com/flagos-ai/FlagTree.git \
      ${FLAGTREE_HOME} && \
    git -C ${FLAGTREE_HOME} switch -c release/${FLAGTREE_VERSION}

RUN cd ${FLAGTREE_HOME} && \
    MAX_JOBS=$(nproc) \
    TRITON_BUILD_PROTON=OFF \
    python -m pip install --no-build-isolation . && \
    python -c "from importlib.metadata import version; assert version('flagtree') == '${FLAGTREE_VERSION}', version('flagtree'); assert version('torch').startswith('${PYTORCH_VERSION}'), version('torch'); print('flagtree:', version('flagtree')); print('torch:', version('torch'))" && \
    ! python -m pip show triton >/dev/null 2>&1

WORKDIR /workspace

CMD ["/bin/bash"]

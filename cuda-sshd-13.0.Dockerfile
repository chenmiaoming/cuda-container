FROM docker.io/nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get dist-upgrade -y  && apt-get install -y \
    openssh-server gdb lldb python3 wget bzip2 ca-certificates git tini   && apt-get autoremove -y && apt-get clean  && rm -rf /var/lib/apt/lists/*

RUN wget --no-hsts --quiet https://mirrors.ustc.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh
RUN printf "channels:\n  - conda-forge\nmirrored_channels:\n  conda-forge:\n    - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge\n" > ${CONDA_DIR}/.condarc && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes

RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc && \
    echo "export PATH=${PATH}" >> /etc/profile


 
RUN echo "root:password" | chpasswd     
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    mkdir -p /var/run/sshd
COPY cuda13.0.env /etc/environment

EXPOSE 22
ENTRYPOINT ["tini", "--"]
CMD ["/usr/sbin/sshd","-D"]

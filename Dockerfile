# using only the opengl image, pytorch fails to detect cuda. 
# But accroding to https://discuss.pytorch.org/t/how-to-check-which-cuda-version-my-pytorch-is-using/116622, 
# it doesn't matter which system cuda you are using.

FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

# System packages.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libgl1-mesa-dev \
  libglew2.1 \
  libglfw3-dev \
  wget \
  build-essential \
  swig \
  libgl1-mesa-glx \
  libosmesa6 \
  libosmesa6-dev \
  patchelf \
  git \
  && apt-get clean

CMD ["/bin/bash"]
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ARG CONDA_VERSION=py310_23.3.1-0
RUN CONDA_VERSION=py310_23.3.1-0 /bin/sh -c set -x &&     UNAME_M="$(uname -m)" &&     if [ "${UNAME_M}" = "x86_64" ]; then         MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh";         SHA256SUM="aef279d6baea7f67940f16aad17ebe5f6aac97487c7c03466ff01f4819e5a651";     elif [ "${UNAME_M}" = "s390x" ]; then         MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-s390x.sh";         SHA256SUM="ed4f51afc967e921ff5721151f567a4c43c4288ac93ec2393c6238b8c4891de8";     elif [ "${UNAME_M}" = "aarch64" ]; then         MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-aarch64.sh";         SHA256SUM="6950c7b1f4f65ce9b87ee1a2d684837771ae7b2e6044e0da9e915d1dee6c924c";     elif [ "${UNAME_M}" = "ppc64le" ]; then         MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-ppc64le.sh";         SHA256SUM="b3de538cd542bc4f5a2f2d2a79386288d6e04f0e1459755f3cefe64763e51d16";     fi &&     wget "${MINICONDA_URL}" -O miniconda.sh -q &&     echo "${SHA256SUM} miniconda.sh" > shasum &&     if [ "${CONDA_VERSION}" != "latest" ]; then sha256sum --check --status shasum; fi &&     mkdir -p /opt &&     bash miniconda.sh -b -p /opt/conda &&     rm miniconda.sh shasum &&     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&     echo "conda activate base" >> ~/.bashrc &&     find /opt/conda/ -follow -type f -name '*.a' -delete &&     find /opt/conda/ -follow -type f -name '*.js.map' -delete &&     /opt/conda/bin/conda clean -afy # buildkit

ENV MUJOCO_GL=egl
WORKDIR /root/

# Python packages.
RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117 --no-cache-dir
RUN pip install xformers==0.0.22 --no-cache-dir
RUN pip install pip --upgrade
RUN pip install "cython<3"
COPY requirements.txt /root/
RUN wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
RUN tar -xzvf mujoco210-linux-x86_64.tar.gz
ENV MUJOCO_PY_MUJOCO_PATH=/root/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/mujoco210/bin
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install gdown --upgrade --no-cache-dir
COPY metaworld.patch /root/
RUN git clone https://github.com/Farama-Foundation/Metaworld && \
  cd Metaworld && \
  git checkout 04be337a12305e393c0caf0cbf5ec7755c7c8feb && \
  git apply ../metaworld.patch && \
  pip install -e . --no-cache-dir
RUN pip install zoopt --no-cache-dir

# fix for mujoco_py with gymnasium
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

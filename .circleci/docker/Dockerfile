# This requires cuda & cudnn packages pre-installed in the base image.
# Other available cuda images are listed at https://hub.docker.com/r/nvidia/cuda
ARG base_image="nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04"
FROM "${base_image}"

ARG python_version="3.7"
ARG cuda="1"
ARG cuda_compute="5.2,7.5"
ARG cc="clang-8"
ARG cxx="clang++-8"
ARG cxx_abi="1"
ARG tpuvm=""

# Disable CUDA for PyTorch
ENV USE_CUDA "0"

# Enable CUDA for XLA
ENV XLA_CUDA "${cuda}"
ENV TF_CUDA_COMPUTE_CAPABILITIES "${cuda_compute}"
ENV TF_CUDA_PATHS "/usr/local/cuda,/usr/include,/usr"

# Whether to build torch and torch_xla libraries with CXX ABI
ENV CC "${cc}"
ENV CXX "${cxx}"
ENV CXX_ABI "${cxx_abi}"
ENV CFLAGS "${CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=${CXX_ABI}"
ENV CXXFLAGS="${CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=${CXX_ABI}"

# Whether to build for TPUVM mode
ENV TPUVM_MODE "${tpuvm}"

# AWS specific CUDA build guidance
ENV TORCH_CUDA_ARCH_LIST Maxwell
ENV TORCH_NVCC_FLAGS "-Xfatbin -compress-all"
ENV CUDA_PATH /usr/local/cuda


# Install base system packages
RUN apt-get update
RUN apt-get install -y python-pip python3-pip git curl libopenblas-dev vim \
  apt-transport-https ca-certificates wget procps openssl libssl-dev sudo

# Install clang & llvm
ADD ./install_llvm_clang.sh install_llvm_clang.sh
RUN bash ./install_llvm_clang.sh

# Sets up jenkins user.
RUN useradd jenkins && \
    mkdir /home/jenkins && \
    chown jenkins /home/jenkins
RUN echo 'jenkins ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p /opt/conda /opt/cargo /opt/rustup /workspace /var/lib/jenkins && \
    chown jenkins /opt/conda /opt/cargo /opt/rustup /workspace /var/lib/jenkins
USER jenkins
WORKDIR /workspace

# Install openmpi for CUDA
run sudo apt-get install -y ssh
run sudo apt-get install -y --allow-downgrades --allow-change-held-packages openmpi-bin libopenmpi-dev

# Builds and configure sccache
ENV OPENSSL_INCLUDE_DIR /usr/include/openssl
ENV OPENSSL_LIB_DIR /usr/lib/x86_64-linux-gnu

ENV CARGO_HOME /opt/cargo
ENV RUSTUP_HOME /opt/rustup

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN . $CARGO_HOME/env && \
    git clone --recursive https://github.com/mozilla/sccache.git && \
    cd sccache && \
    cargo install sccache && \
    cd .. && \
    rm -rf sccache

ENV PATH $CARGO_HOME/bin:$PATH

# Installs and configures Conda.
ADD ./install_conda.sh install_conda.sh
RUN sudo chown jenkins ./install_conda.sh
RUN bash ./install_conda.sh "${python_version}" /opt/conda

RUN echo "conda activate pytorch" >> ~/.bashrc
RUN echo "export TF_CPP_LOG_THREAD_ID=1" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH

RUN bash -c "source ~/.bashrc"
CMD ["bash"]

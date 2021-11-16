# Use nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 or later for CUDA.
# Warning: All ARGs placed before FROM will only be scoped up unitl FROM statement.
# https://github.com/docker/cli/blob/3c7ede6a68941f64c3a154c9a753eb7a9b1c2c3e/docs/reference/builder.md#understand-how-arg-and-from-interact
ARG base_image="debian:buster"
FROM "${base_image}"

ARG python_version="3.6"
ARG cloud_build="false"
ARG release_version="nightly"
ARG cuda="0"
ARG cuda_compute="7.0,7.5"
ARG cxx_abi="0"
ARG tpuvm=""
ARG debian9="debian9"

RUN apt-get update
RUN apt-get install -y git sudo python-pip python3-pip

RUN mkdir debian9
RUN cd debian9 && git clone https://github.com/pytorch/pytorch

# Disable CUDA for PyTorch
ENV USE_CUDA "0"

# Enable CUDA for XLA
ENV XLA_CUDA "${cuda}"
ENV TF_CUDA_COMPUTE_CAPABILITIES "${cuda_compute}"

# Whether the build is running on GCB
ENV CLOUD_BUILD "${cloud_build}"

# Whether to build torch and torch_xla libraries with CXX ABI
ENV CXX_ABI "${cxx_abi}"

# Whether to build for TPUVM mode
ENV TPUVM_MODE "${tpuvm}"

# HACK: To create Debian 9 image build by explicit installation of gcc-8.3 which
# resolves build error on debian 9.
RUN sudo apt-get install -y unzip
RUN sudo apt-get install -y wget
RUN wget http://ftp.mirrorservice.org/sites/sourceware.org/pub/gcc/releases/gcc-8.3.0/gcc-8.3.0.tar.gz
RUN tar zxf gcc-8.3.0.tar.gz
RUN cd gcc-8.3.0

RUN cd gcc-8.3.0 && sudo ./contrib/download_prerequisites
RUN sudo apt -y install build-essential
RUN cd gcc-8.3.0 && sudo ./configure --disable-multilib
RUN cd gcc-8.3.0 && sudo make -j 4
RUN cd gcc-8.3.0 && sudo make install
ENV LD_LIBRARY_PATH /usr/local/lib64/:$LD_LIBRARY_PATH

RUN cd /usr/bin && sudo rm gcc gcc-ar gcc-nm gcc-ranlib
RUN sudo ln -s /usr/local/bin/gcc /usr/bin/gcc
RUN sudo ln -s /usr/local/bin/gcc-ar /usr/bin/gcc-ar
RUN sudo ln -s /usr/local/bin/gcc-nm /usr/bin/gcc-nm
RUN sudo ln -s /usr/local/bin/gcc-ranlib /usr/bin/gcc-ranlib

# To get around issue of Cloud Build with recursive submodule update
# clone recursively from pytorch/xla if building docker image with
# cloud build. Otherwise, just use local.
# https://github.com/GoogleCloudPlatform/cloud-builders/issues/435
COPY . /debian9/pytorch/xla
RUN if [ "${cloud_build}" = true ]; then github_branch="${release_version}" && \
  if [ "${release_version}" = "nightly" ]; then github_branch="master"; fi && \
  cd /debian9/pytorch && \
  rm -rf xla && \
  git clone -b amangu_build --recursive https://github.com/pytorch/xla && \
  sed -i 's/c++14/c++17/g' ./xla/third_party/tensorflow/.bazelrc && \
  cd / && \
  git clone -b "${github_branch}" --recursive https://github.com/pytorch-tpu/examples tpu-examples; fi

RUN cd /debian9/pytorch && bash xla/scripts/build_torch_wheels.sh ${python_version} ${release_version}

# Use conda environment on startup or when running scripts.
RUN echo "conda activate pytorch" >> ~/.bashrc
RUN echo "export TF_CPP_LOG_THREAD_ID=1" >> ~/.bashrc
ENV PATH /root/anaconda3/envs/pytorch/bin/:/root/bin:$PATH

# Define entrypoint and cmd
COPY docker/docker-entrypoint.sh /usr/local/bin
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]

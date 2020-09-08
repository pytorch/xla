# Use nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 or later for CUDA.
# Warning: All ARGs placed before FROM will only be scoped up unitl FROM statement.
# https://github.com/docker/cli/blob/3c7ede6a68941f64c3a154c9a753eb7a9b1c2c3e/docs/reference/builder.md#understand-how-arg-and-from-interact
ARG base_image="debian:stretch"
FROM "${base_image}"

ARG python_version="3.6"
ARG cloud_build="false"
ARG release_version="nightly"
ARG cuda="0"
ARG cuda_compute="7.0,7.5"
ARG cxx_abi="0"
# Aug 3 VM's latest commit
#ARG xla_commit="354b2429bdcc87d0ad3a516b6d7644015f8c58ad"

# Aug 4 VM's latest commit
#ARG pytorch_commit="8850fd1952c3983793dcac4022fdc8e3913dad96"

# Aug 4 VM's latest commit
#ARG xla_commit="18fa011c3eba79b93ea38cf5b0946920e45b2053"

# Using 1 of 2 commits from Aug 3.
#ARG xla_commit="7a2f0493986803a277392ca565a883535f990a47"

# Commit on Aug 7
ARG xla_commit="2c732f20b5343dd1a694cd69cd15dd7f76ee1028"
ARG pytoch_commit="b7a9bc0802b68cff07958937cfda2271155689e5"

# Aug 3 VM's latest commit
#ARG pytorch_commit="bfa94487b968ccb570ef8cd9547029b967e76ed0"

RUN apt-get update
RUN apt-get install -y git sudo python-pip python3-pip
RUN git clone https://github.com/pytorch/pytorch

# Disable CUDA for PyTorch
ENV USE_CUDA "0"

# Enable CUDA for XLA
ENV XLA_CUDA "${cuda}"
ENV TF_CUDA_COMPUTE_CAPABILITIES "${cuda_compute}"

# Whether the build is running on GCB
ENV CLOUD_BUILD "${cloud_build}"

# Whether to build torch and torch_xla libraries with CXX ABI
ENV CXX_ABI "${cxx_abi}"

# To get around issue of Cloud Build with recursive submodule update
# clone recursively from pytorch/xla if building docker image with
# cloud build. Otherwise, just use local.
# https://github.com/GoogleCloudPlatform/cloud-builders/issues/435
COPY . /pytorch/xla
RUN if [ "${cloud_build}" = true ]; then github_branch="${release_version}" && \
  if [ "${release_version}" = "nightly" ]; then github_branch="master"; fi && \
  cd /pytorch && \
  rm -rf xla && \
  git clone -b "${github_branch}" --recursive https://github.com/pytorch/xla && \
  cd / && \
  git clone -b "${github_branch}" --recursive https://github.com/pytorch-tpu/examples tpu-examples; fi

RUN if [ -n "$xla_commit" ]; then cd /pytorch && rm -rf xla && \
  git clone https://github.com/pytorch/xla --recursive && \
  cd xla && \
  git fetch origin $xla_commit && \
  #git checkout $xla_commit; fi
  git checkout $xla_commit && \
  git revert 18fa011c3eba79b93ea38cf5b0946920e45b2053 --no-edit; fi

RUN if [ -n "$pytorch_commit" ]; then echo "$pytorch_commit" >> /pytorch/xla/.torch_pin; fi

RUN cd /pytorch && bash xla/scripts/build_torch_wheels.sh ${python_version} ${release_version}

# Use conda environment on startup or when running scripts.
RUN echo "conda activate pytorch" >> ~/.bashrc
RUN echo "export TF_CPP_LOG_THREAD_ID=1" >> ~/.bashrc
ENV PATH /root/anaconda3/envs/pytorch/bin/:/root/bin:$PATH

# Define entrypoint and cmd
COPY docker/docker-entrypoint.sh /usr/local/bin
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]

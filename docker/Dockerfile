FROM debian:stretch

ARG python_version="3.6"
ARG cloud_build="false"
ARG release_version="nightly"

RUN apt-get update
RUN apt-get install -y git sudo python-pip python3-pip
RUN git clone https://github.com/pytorch/pytorch

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

RUN cd pytorch && bash xla/scripts/build_torch_wheels.sh ${python_version} ${release_version}

# Use conda environment on startup or when running scripts.
RUN echo "conda activate pytorch" >> ~/.bashrc
ENV PATH /root/anaconda3/envs/pytorch/bin/:$PATH

# Define entrypoint and cmd
COPY docker/docker-entrypoint.sh /usr/local/bin
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]

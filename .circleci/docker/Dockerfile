FROM debian:bullseye

# Install base system packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
         python3 \
         build-essential \
         wget \
         cmake \
         git \
         curl \
         sudo \
         ca-certificates \
         openssl \
         libssl-dev \
         libopenblas-dev \
         libjpeg-dev \
         libpng-dev && \
         rm -rf /var/lib/apt/lists/*

# Set Python 3 as default.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Sets up jenkins user.
RUN useradd jenkins && \
    mkdir /home/jenkins && \
    chown jenkins /home/jenkins
RUN echo 'jenkins ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p /opt/conda /opt/cargo /opt/rustup /workspace /var/lib/jenkins && \
    chown jenkins /opt/conda /opt/cargo /opt/rustup /workspace /var/lib/jenkins
USER jenkins
WORKDIR /workspace

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
RUN wget -O miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x miniconda.sh && \
     ./miniconda.sh -b -u -p /opt/conda && \
     rm miniconda.sh && \
     /opt/conda/bin/conda install -y numpy pyyaml scipy ipython mkl mkl-include ninja cython typing
ENV PATH /opt/conda/bin:$PATH
RUN conda init bash

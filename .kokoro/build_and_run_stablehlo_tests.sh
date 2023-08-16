#!/bin/bash
set -ex
DEBIAN_FRONTEND=noninteractive
XLA_DIR="${KOKORO_ARTIFACTS_DIR}/github/xla"
cd $XLA_DIR
docker build . -f=.kokoro/Dockerfile -t xladocker
docker run -it xladocker

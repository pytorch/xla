#!/bin/bash
set -ex
DEBIAN_FRONTEND=noninteractive
XLA_DIR="${KOKORO_ARTIFACTS_DIR}/github/xla"
cd $XLA_DIR
cp $KOKORO_KEYSTORE_DIR ./keystore_content -r
docker build . -f=.kokoro/Dockerfile -t xladocker
docker -v "$KOKORO_KEYSTORE_DIR:/tmp/keystore" run xladocker

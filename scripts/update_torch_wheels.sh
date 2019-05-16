#!/bin/bash
gsutil cp gs://pytorch-tpu-releases/wheels/torch-1.1.0a0+stable-cp35-cp35m-linux_x86_64.whl /tmp/.
pip3 install /tmp/torch-1.1.0a0+stable-cp35-cp35m-linux_x86_64.whl
gsutil cp gs://pytorch-tpu-releases/wheels/torch_xla-0.1+nightly-cp35-cp35m-linux_x86_64.whl /tmp/.
pip3 install /tmp/torch_xla-0.1+nightly-cp35-cp35m-linux_x86_64.whl

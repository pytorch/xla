#!/bin/bash

# Explicitly source bashrc even when running commands directly.
# Since commands run as a separate subshell, we need to source manually.
# ex. docker run -it gcr.io/tpu-pytorch/xla:nightly bash ...
# The above will not source bashrc without entrypoint.
source ~/.bashrc

# Activate pytorch conda env at entry by default.
# TODO: This should not be needed as it is already sourced from the .bashrc above.
conda activate pytorch

exec "$@"

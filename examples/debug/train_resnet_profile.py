import os
import sys
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_resnet_base import TrainResNetBase

import torch_xla.debug.profiler as xp

# check https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#environment-variables
os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

if __name__ == '__main__':
  base = TrainResNetBase()
  profile_port = 9012
  # you can also set profile_logdir to a gs bucket, for example
  # profile_logdir = "gs://your_gs_bucket/profile"
  profile_logdir = "/tmp/profile/"
  duration_ms = 30000
  assert profile_logdir.startswith('gs://') or os.path.exists(profile_logdir)
  server = xp.start_server(profile_port)
  # Ideally you want to start the profile tracing after the initial compilation, for example
  # at step 5.
  xp.trace_detached(
      f'localhost:{profile_port}', profile_logdir, duration_ms=duration_ms)
  base.start_training()
  # You can view the profile at tensorboard by
  # 1. pip install tensorflow-cpu tensorboard-plugin-profile
  # 2. tensorboard --logdir /tmp/profile/ --port 6006
  # For more detail plase take a look at https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm

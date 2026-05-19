import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_resnet_base import TrainResNetBase

import itertools
import time

import torch_xla
import torch_xla.core.xla_model as xm


# This example aims to provide a simple way to benchmark torch_xla. Ideally device execution
# time should be greater than the tracing time so tracing time can be overlapped perfectlly.
# If that's not the case try to increase the batch size which will increase the device execution
# time and keep tracing time the same.
class TrainResNetBenchmark(TrainResNetBase):

  def train_loop_fn(self, loader, epoch):
    self.model.train()
    loader = itertools.islice(loader, self.num_steps)
    for step, (data, target) in enumerate(loader):
      tracing_start_time = time.time()
      self.optimizer.zero_grad()
      output = self.model(data)
      loss = self.loss_fn(output, target)
      loss.backward()
      self.run_optimizer()
      tracing_end_time = time.time()
      # for releases < 2.3 uses `xm.mark_step()`.
      # Couple things to note
      # 1. sync itself is not blocking, it will schedule a device execution and return.
      # 2. In TrainResNetBase we uses MpDeviceLoader which calls `torch_xla.sync()` for every batch.
      #    We don't have to manually call `sync` here if we don't want to wait for execution to finish.
      torch_xla.sync()
      # Do not call this function every step unless you are benchmarking. It will block the main process
      # and torch_xla won't be able to overlap the tracing of the next step with the device
      # execution of the current step. For e2e benchmarking, call `wait_device_ops` once at the end.
      xm.wait_device_ops()
      device_execution_end_time = time.time()
      print(
          f'Step: {step}, Tracing time: {tracing_end_time - tracing_start_time}s, E2E time: {device_execution_end_time - tracing_start_time}s'
      )


if __name__ == '__main__':
  benchmark = TrainResNetBenchmark()
  benchmark.start_training()

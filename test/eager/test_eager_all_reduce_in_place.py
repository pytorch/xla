import torch
import torch_xla

import torch_xla.core.xla_model as xm
import torch_xla.debug
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.metrics as met


def _mp_fn(index):
  import torch_xla
  torch_xla.experimental.eager_mode(True)

  device = torch_xla.device()

  if xm.xla_device_hw(device) not in ('TPU', 'CUDA', 'NEURON'):
    return

  ordinal_tensor_1 = torch.tensor([index], dtype=torch.float).to(device)
  ordinal_tensor_2 = torch.tensor([index], dtype=torch.int32).to(device)
  xm.wait_device_ops()
  met.clear_all()

  # all_reduce with list of tensor as input will be a inplace op. This is
  # used by the optimizer_step.
  xm.all_reduce(xm.REDUCE_SUM, [ordinal_tensor_1, ordinal_tensor_2])

  xm.wait_device_ops()
  assert met.metric_data("EagerOpExecuteTime")[0] == 1

  num_device = torch_xla.runtime.global_runtime_device_count()
  expected_sum = (num_device - 1) * num_device / 2
  expected_1 = torch.tensor([(expected_sum)], dtype=torch.float)
  expected_2 = torch.tensor([(expected_sum)], dtype=torch.int32)
  assert torch.allclose(expected_1, ordinal_tensor_1.cpu())
  assert torch.allclose(expected_2, ordinal_tensor_2.cpu())


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())

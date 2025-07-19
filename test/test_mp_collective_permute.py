import sys
import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm


def _test_single_tensor_collective_permute(device, world_size, ordinal, pairs):
  value = torch.tensor([ordinal] * 100, dtype=torch.int32, device=device)
  result_tensor = xm.collective_permute(value, pairs)

  result = result_tensor.cpu().tolist()
  expected = [ordinal - 1] * 100 if ordinal != 0 else [world_size - 1] * 100

  if result != expected:
    print(f"Wrong result from core {ordinal}: {result}", file=sys.stderr)
    return False
  return True


def _test_multi_tensor_collective_permute(device, world_size, ordinal, pairs):
  tensor1 = torch.tensor([ordinal] * 50, dtype=torch.int32, device=device)
  tensor2 = torch.tensor([ordinal + 100] * 75, dtype=torch.int32, device=device)
  tensor3 = torch.tensor(
      [ordinal + 200] * 25, dtype=torch.float32, device=device)

  result_list = xm.collective_permute([tensor1, tensor2, tensor3], pairs)
  expected_ordinal = ordinal - 1 if ordinal != 0 else world_size - 1

  result1 = result_list[0].cpu().tolist()
  expected1 = [expected_ordinal] * 50
  if result1 != expected1:
    print(f"Wrong result from core {ordinal}: {result1}", file=sys.stderr)
    return False

  result2 = result_list[1].cpu().tolist()
  expected2 = [expected_ordinal + 100] * 75
  if result2 != expected2:
    print(f"Wrong result from core {ordinal}: {result2}", file=sys.stderr)
    return False

  result3 = result_list[2].cpu().tolist()
  expected3 = [expected_ordinal + 200.0] * 25
  if result3 != expected3:
    print(f"Wrong result from core {ordinal}: {result3}", file=sys.stderr)
    return False

  return True


def _mp_fn(index):
  device = torch_xla.device()
  if xm.xla_device_hw(device) in ['TPU', 'NEURON']:
    world_size = xr.world_size()
    ordinal = xr.global_ordinal()
    pairs = []
    for i in range(1, world_size):
      pairs.append([i - 1, i])
    pairs.append([world_size - 1, 0])
    if not _test_single_tensor_collective_permute(device, world_size, ordinal,
                                                  pairs):
      sys.exit(1)
    if not _test_multi_tensor_collective_permute(device, world_size, ordinal,
                                                 pairs):
      sys.exit(1)
  else:
    print(
        f"Device {device} is not a supported device for this test",
        file=sys.stderr)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())

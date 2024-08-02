import sys
import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.functions as xf
import torch_xla.core.xla_model as xm


def _mp_fn(index):
  device = xm.xla_device()

  if xm.xla_device_hw(device) in ('TPU', 'CUDA'):
    world_size = xr.world_size()
    torch_xla._XLAC._xla_set_mat_mul_precision('highest')
    torch.manual_seed(11)
    xm.set_rng_state(11)

    N = 3
    M = 4
    KO = 2
    wsize = KO * world_size
    wg = torch.randn(N, wsize, device=device, requires_grad=True)
    w = torch.narrow(wg, 1, index * KO, KO)
    x = torch.randn(wsize, M, device=device)

    mm = wg @ x
    bmm = xf.distributed_mm(w, x, split=2)

    mm_cpu = mm.cpu()
    bmm_cpu = bmm.cpu()
    if not mm_cpu.allclose(bmm_cpu, rtol=1e-04, atol=1e-04):
      print('distributed_mm() produced wrong result', file=sys.stderr)
      print('[{}]\n{}\n{}'.format(index, mm_cpu, bmm_cpu), file=sys.stderr)
      sys.exit(1)
  else:
    print(
        'Default device {} is not a TPU or GPU device'.format(device),
        file=sys.stderr)


if __name__ == '__main__':
  torch_xla.launch(_mp_fn)

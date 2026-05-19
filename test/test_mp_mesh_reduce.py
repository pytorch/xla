import torch
import torch_xla
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm


def _test_scalar():

  def reduce_add(vlist):
    return sum(vlist)

  svalue = 1.25
  rvalue = xm.mesh_reduce('test_mp_mesh_reduce._test_scalar', svalue,
                          reduce_add)
  assert rvalue == svalue * xr.world_size()


def _test_tensor():

  def reduce_add(vlist):
    return torch.stack(vlist).sum(dim=0)

  tvalue = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
  rvalue = xm.mesh_reduce('test_mp_mesh_reduce._test_tensor', tvalue,
                          reduce_add)
  assert rvalue.allclose(tvalue * xr.world_size())


def _mp_fn(index):
  _test_scalar()
  _test_tensor()


if __name__ == '__main__':
  torch_xla.launch(_mp_fn, args=())

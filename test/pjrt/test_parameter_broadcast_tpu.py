from numpy.testing import assert_array_equal, assert_raises
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt
from absl.testing import absltest, parameterized


def broadcast(sync):
  torch.manual_seed(xm.get_ordinal())
  device = xm.xla_device()
  model = nn.Linear(5, 5).to(device)
  if sync:
    pjrt.broadcast_master_param(model)
  return next(model.parameters()).detach().cpu().numpy()


class TestBroadcastParametersPjrt(parameterized.TestCase):

  @parameterized.named_parameters(('synchronized_parameters', True),
                                  ('unsynchronized_parameters', False))
  def test_broadcast_parameter_sync(self, sync):
    torch.set_default_tensor_type('torch.FloatTensor')
    results = pjrt.run_multiprocess(broadcast, sync)
    master_params = results[0][0]
    for process_key in results:
      worker_params = results[process_key][0]
      if sync:
        assert_array_equal(master_params, worker_params)
      elif process_key != 0:
        assert_raises(AssertionError, assert_array_equal, master_params,
                      worker_params)


if __name__ == '__main__':
  absltest.main()

import itertools
import os
from typing import List, Optional
from absl.testing import absltest, parameterized

import torch.distributed as dist
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt


class PjRtMeshServiceTest(parameterized.TestCase):

  @staticmethod
  def _rendezvous_default(xrt_mesh_addr: Optional[str], replicas: List[int]):
    if xrt_mesh_addr:
      os.environ['XRT_MESH_SERVICE_ADDRESS'] = xrt_mesh_addr

    payload = b'message %d' % xm.get_ordinal()
    return xm.rendezvous("test rendezvous", payload, replicas)

  @parameterized.named_parameters(
      ('defaults', None, []), ('xrt_address', 'localhost:9477', []),
      ('four_replicas', None, [0, 1, 2, 3]), ('two_replicas', None, [0, 1]))
  def test_rendezvous(self, xrt_mesh_addr, replicas):
    results = pjrt._run_multiprocess(self._rendezvous_default, xrt_mesh_addr,
                                     replicas)
    replicas = replicas or list(range(len(results)))

    for ordinal, value in results.items():
      if ordinal in replicas or not replicas:
        self.assertEqual(value, [b'message %d' % r for r in replicas])

  @staticmethod
  def _mesh_reduce():
    return xm.mesh_reduce('test mesh reduce', xm.get_ordinal(), sum)

  def test_mesh_reduce(self):
    results = pjrt._run_multiprocess(self._mesh_reduce)
    values = list(results.values())

    expected = sum(range(len(values)))
    for v in values:
      self.assertEqual(v, expected)


if __name__ == "__main__":
  absltest.main()

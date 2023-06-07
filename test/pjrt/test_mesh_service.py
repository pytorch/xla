import functools
from absl.testing import absltest, parameterized

import torch_xla.debug.metrics as met
import torch_xla.core.xla_model as xm
from torch_xla import runtime as xr
from torch_xla._internal import pjrt


class PjRtMeshServiceTest(parameterized.TestCase):

  @staticmethod
  def _rendezvous_static_size():
    payload = b'message %d' % xm.get_ordinal()
    return xm.rendezvous("test rendezvous", payload)

  def test_rendezvous_static_size(self):
    results = pjrt.run_multiprocess(self._rendezvous_static_size)

    expected = sorted([b'message %d' % r for r in results])
    self.assertDictEqual(results, {r: expected for r in results})

  @staticmethod
  def _rendezvous_dynamic_size():
    payload = b'message' * xm.get_ordinal()
    return xm.rendezvous("test rendezvous", payload)

  def test_rendezvous_dynamic_size(self):
    results = pjrt.run_multiprocess(self._rendezvous_dynamic_size)

    expected = sorted([b'message' * r for r in results])
    self.assertDictEqual(results, {r: expected for r in results})

  @staticmethod
  def _rendezvous_replica_groups():
    replicas = list(range(xr.global_device_count()))
    return xm.rendezvous("test rendezvous", b'message', replicas)

  def test_rendezvous_replica_groups(self):
    results = pjrt.run_multiprocess(self._rendezvous_replica_groups)

    expected = [b'message'] * len(results)
    self.assertDictEqual(results, {r: expected for r in results})

  def test_rendezvous_empty_payload(self):
    test_fn = functools.partial(xm.rendezvous, 'test rendezvous', b'')
    results = pjrt.run_multiprocess(test_fn)

    expected = [b''] * len(results)
    self.assertDictEqual(results, {r: expected for r in results})

  @staticmethod
  def rendezvous_default_payload_cpu_transfers():
    xm.rendezvous('test rendezvous')

    return met.counter_value('xla::_to_cpu')

  def test_rendezvous_default_payload_cpu_transfers(self):
    results = pjrt.run_multiprocess(
        self.rendezvous_default_payload_cpu_transfers)

    # Expect one CPU transfer: the max size of all payloads
    for val in results.values():
      self.assertEqual(val, 1)

  def test_rendezvous_string_payload(self):
    test_fn = functools.partial(xm.rendezvous, 'test rendezvous', "")

    with self.assertRaises(TypeError):
      pjrt.run_multiprocess(test_fn)

  @staticmethod
  def _mesh_reduce():
    return xm.mesh_reduce('test mesh reduce', xm.get_ordinal(), sum)

  def test_mesh_reduce(self):
    results = pjrt.run_multiprocess(self._mesh_reduce)
    values = list(results.values())

    expected = sum(range(len(values)))
    for v in values:
      self.assertEqual(v, expected)


if __name__ == "__main__":
  absltest.main()

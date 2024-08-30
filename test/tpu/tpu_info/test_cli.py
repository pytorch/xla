import threading
from absl.testing import absltest, parameterized
import contextlib
import multiprocessing
import os
from typing import Dict, Optional
import torch_xla as xla
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp
from tpu_info import cli, device, metrics


class TpuInfoCliTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    xr.set_device_type("TPU")
    chip_type, num_chips = device.get_local_chips()
    assert chip_type is not None
    cls.chip_type = chip_type
    cls.num_chips = num_chips

  @staticmethod
  def _init_tpu_and_wait(
      # accept index as first arg to make xmp.spawn happy
      index: int,
      q: multiprocessing.Queue,
      done: multiprocessing.Event,
      env: Optional[Dict[str, str]] = None,
  ):
    if env:
      os.environ.update(**env)
    xla.device()
    q.put(os.getpid())
    done.wait()

  @contextlib.contextmanager
  def _torch_xla_process(self, env: Dict[str, str]):
    with multiprocessing.Manager() as m:
      q = m.Queue()
      done = m.Event()
      p = multiprocessing.Process(
          target=self._init_tpu_and_wait, args=(0, q, done, env))
      p.start()
      pid = q.get(timeout=20.0)
      with contextlib.ExitStack() as e:
        e.callback(done.set)
        yield pid
      # Wait for process to exit before next test
      p.join()

  @parameterized.named_parameters([
      ("all_chips", {}),
      ("one_chip", {
          "TPU_VISIBLE_CHIPS": "0",
          "TPU_PROCESS_BOUNDS": "1,1,1",
          "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1"
      }),
  ])
  def test_single_process_e2e(self, extra_env):
    with self._torch_xla_process(extra_env) as subprocess_pid:
      owners = device.get_chip_owners()
      for _, pid in owners.items():
        self.assertEqual(pid, subprocess_pid)
      usages = metrics.get_chip_usage(self.chip_type)
      for u in usages:
        self.assertGreater(u.total_memory, 0)
        self.assertEqual(u.duty_cycle_pct, 0.0)
        one_gb = 1 << 30
        self.assertLess(u.memory_usage, one_gb)
      # TODO: check output
      cli.print_chip_info()

  @contextlib.contextmanager
  def _torch_xla_spawn(self):
    with multiprocessing.Manager() as m:
      q = m.Queue()
      done = m.Event()
      # HACK: run xmp.spawn in a thread because `join` arg is not implemented
      t = threading.Thread(
          target=xmp.spawn,
          args=(self._init_tpu_and_wait,),
          kwargs={'args': (q, done)})
      t.start()

      # v2 and v3 may have duplicates due to multithreading
      child_pids = set()
      for _ in range(self.chip_type.value.devices_per_chip *
                     self.num_chips):
        child_pids.add(q.get(timeout=20.0))
      with contextlib.ExitStack() as e:
        e.callback(done.set)
        yield child_pids

      t.join()

  def test_multiprocessing_e2e(self):
    with self._torch_xla_spawn() as subprocess_pids:
      owners = device.get_chip_owners()
      self.assertSetEqual(
          set(pid for _, pid in owners.items()), subprocess_pids)
      usages = metrics.get_chip_usage(self.chip_type)
      for u in usages:
        self.assertGreater(u.total_memory, 0)
        self.assertEqual(u.duty_cycle_pct, 0.0)
        one_gb = 1 << 30
        self.assertLess(u.memory_usage, one_gb)
      # TODO: check output
      cli.print_chip_info()


if __name__ == "__main__":
  absltest.main()

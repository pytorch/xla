import contextlib
import multiprocessing
import os
import threading
from typing import Dict, Optional
import pytest
import torch_xla as xla
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp
from tpu_info import cli, device, metrics


@pytest.fixture
def local_tpus():
  chip_type, num_chips = device.get_local_chips()
  assert chip_type, "Expected local TPU chip"
  yield chip_type, num_chips


@pytest.fixture(scope="module", autouse=True)
def use_tpu():
  xr.set_device_type("TPU")


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


@pytest.fixture(params=[
    {},
    {
        "TPU_VISIBLE_CHIPS": "0",
        "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
    },
])
def torch_xla_process(request: Dict[str, str]):
  extra_env = request.param
  with multiprocessing.Manager() as m:
    q = m.Queue()
    done = m.Event()
    env = {
        "TPU_RUNTIME_METRICS_PORTS": "8431",
    }
    env.update(extra_env)
    p = multiprocessing.Process(
        target=_init_tpu_and_wait, args=(0, q, done, env))
    p.start()
    pid = q.get(timeout=10.0)
    with contextlib.ExitStack() as e:
      e.callback(done.set)
      yield pid
    # Wait for process to exit before next test
    p.join()


def test_single_process_e2e(torch_xla_process, local_tpus):
  chip_type, _ = local_tpus
  owners = device.get_chip_owners()
  for _, pid in owners.items():
    assert pid == torch_xla_process
  usages = metrics.get_chip_usage(chip_type)
  for u in usages:
    assert u.total_memory > 0
    assert u.duty_cycle_pct == 0.0
    # There's a small amount of memory usage when libtpu is initialized
    one_gb = 1 << 30
    assert u.memory_usage < one_gb
  # TODO: check output
  cli.print_chip_info()


@pytest.fixture
def torch_xla_spawn(local_tpus):
  chip_type, num_chips = local_tpus
  with multiprocessing.Manager() as m:
    q = m.Queue()
    done = m.Event()
    # TODO: This should get set automatically by libtpu
    os.environ["TPU_RUNTIME_METRICS_PORTS"] = ",".join(
        str(i) for i in range(8431, 8431 + num_chips))
    # HACK: run xmp.spawn in a thread because `join` arg is not implemented
    t = threading.Thread(
        target=lambda: xmp.spawn(_init_tpu_and_wait, args=(q, done)))
    t.start()
    child_pids = set()
    for _ in range(chip_type.value.accelerators_per_chip * num_chips):
      child_pids.add(q.get(timeout=10.0))
    with contextlib.ExitStack() as e:
      e.callback(done.set)
      yield child_pids
    os.environ.pop("TPU_RUNTIME_METRICS_PORTS")
    t.join()


def test_multiprocessing_e2e(torch_xla_spawn, local_tpus):
  chip_type, _ = local_tpus
  child_pids = torch_xla_spawn
  owners = device.get_chip_owners()
  for _, pid in owners.items():
    assert pid in child_pids
  usages = metrics.get_chip_usage(chip_type)
  for u in usages:
    assert u.total_memory > 0
    assert u.duty_cycle_pct == 0.0
    one_gb = 1 << 30
    assert u.memory_usage < one_gb
  # TODO: check output
  cli.print_chip_info()

import os
import subprocess
import pathlib


def test_local_torchrun_rt_init():
  # This test launches a allreduce using torchrun launcher, uses native xla_model CCop
  ci_dir = pathlib.Path(__file__).parent.resolve()
  cmd = f'torchrun --nproc_per_node=2  --master_addr=127.0.0.1 --master_port=2020 {ci_dir}/allreduce_torchrun.py'
  proc = subprocess.Popen(cmd, shell=True)
  return_code = proc.wait()
  assert return_code == 0


def test_local_torchrun_xla_backend():
  # This test launches a allreduce using torchrun launcher, uses xla backend
  ci_dir = pathlib.Path(__file__).parent.resolve()
  cmd = f'torchrun --nproc_per_node=2  --master_addr=127.0.0.1 --master_port=2020 {ci_dir}/allreduce_torchrun.py --use_xla_backend'
  proc = subprocess.Popen(cmd, shell=True)
  return_code = proc.wait()
  assert return_code == 0


if __name__ == '__main__':
  test_local_torchrun_rt_init()
  test_local_torchrun_xla_backend()

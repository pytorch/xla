"""
This script is for starting the xrt_server. It also polls the PID and
checks if it exist. It would kill the server, when the process whose
PID it was tracking dies.
NOTE: This script should be used only by xrt_init.py and not anyone else.
"""
import os
import argparse
import psutil
import time
import signal
import multiprocessing
import torch_xla


def _polling(pid_to_track):

  def is_pid_alive(pid):
    # The idea behind this is: if the process doesn't exist,
    # getting a process status should throw an error.
    # If the process exist, then we check if it hasn't gone
    # into zombie state. This can happen when we run torchrun
    # from neuron_parallel_compile.
    try:
      return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except:
      return False

  while is_pid_alive(pid_to_track):
    time.sleep(10)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", required=True)
  parser.add_argument("--pid_to_track", default=None)
  args = parser.parse_args()
  polling_process = multiprocessing.Process(
      target=_polling, args=(int(args.pid_to_track),))
  server_process = multiprocessing.Process(
      target=torch_xla._XLAC._run_xrt_local_service, args=(int(args.port),))
  polling_process.start()
  server_process.start()
  polling_process.join()
  os.kill(server_process.pid, signal.SIGKILL)

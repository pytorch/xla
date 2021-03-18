import subprocess
import sys

if __name__ == '__main__':
  assert len(sys.argv) == 2, 'Need to provide the local service port'
  subprocess.Popen(
      ["python", "-m", "torch_xla.core._xrt_run_server", sys.argv[1]],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      start_new_session=True)

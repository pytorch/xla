import subprocess

if __name__ == '__main__':
  subprocess.Popen(["python", "-m", "torch_xla.core._xrt_run_server"],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   start_new_session=True)

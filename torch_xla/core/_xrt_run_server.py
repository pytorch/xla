import torch_xla
import sys

if __name__ == '__main__':
  assert len(sys.argv) == 2, 'Need to provide the local service port'
  torch_xla._XLAC._run_pjrt_local_service(int(sys.argv[1]))

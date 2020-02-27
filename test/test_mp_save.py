import io
import os
import tempfile
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _create_state_dict(device):
  a = torch.randn(3, 4, device=device)
  b = torch.randn(3, 4, device=device)
  c = torch.randn(3, 4, device=device)
  d = dict()
  d['a'] = a
  d['b'] = b
  d['c'] = c
  d['a+b'] = a + b
  d['a+c'] = a + c
  d['b*c'] = b * c
  d['a,b,c'] = [a, b, c]
  d['b,c,a*b'] = tuple([b, c, a * b])
  return d


def _get_temp_file():
  fd, path = tempfile.mkstemp()
  os.close(fd)
  return path


def _get_data_str(data):
  bio = io.BytesIO()
  torch.save(data, bio)
  return bio.getvalue()


def _mp_fn(index, temp_file):
  device = xm.xla_device()
  dd = _create_state_dict(device)
  xm.save(dd, temp_file)
  ldd = torch.load(temp_file)
  pdd = _get_data_str(ldd)
  data = xm.rendezvous('xm_save_test', pdd)
  if xm.get_local_ordinal() == 0:
    os.remove(temp_file)
  for i in range(1, len(data)):
    bio = io.BytesIO(data[i])
    ildd = torch.load(bio)
    for k, v in ldd.items():
      if isinstance(v, torch.Tensor):
        assert v.allclose(ildd[k])
      elif isinstance(v, (list, tuple)):
        iv = ildd[k]
        for a, b in zip(v, iv):
          assert a.allclose(b)
      else:
        raise RuntimeError('Invalid data type')


if __name__ == '__main__':
  temp_file = _get_temp_file()
  xmp.spawn(_mp_fn, args=(temp_file,))

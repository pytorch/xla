import re
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _get_replica_group(index):
  world_size = xm.xrt_world_size()
  split = world_size // 2
  gid = index // split if split > 0 else 0
  return list(range(0, split)) if index < split else list(
      range(split, world_size)), gid


def _mp_fn(index):
  ordinal = xm.get_ordinal()
  print('Core {} waiting for rendezvous ...'.format(ordinal))
  replicas, gid = _get_replica_group(index)
  data = xm.rendezvous(
      'rendezvous_test.{}'.format(gid),
      'ORD={}'.format(ordinal).encode('utf-8'),
      replicas=replicas)
  print('Core {} got rendezvous!'.format(ordinal))
  for i in range(0, len(data)):
    idata = data[i].decode('utf-8')
    m = re.match(r'ORD=(\d+)', idata)
    assert m, 'Bad payload format: {}'.format(idata)
    xordinal = int(m.group(1))
    assert replicas[i] == xordinal, 'Payload {} got ordinal {}'.format(
        replicas[i], xordinal)
  xm.rendezvous('_mp_fn.exit')


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())

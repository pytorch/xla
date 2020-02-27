import re
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(index):
  ordinal = xm.get_ordinal()
  print('Core {} waiting for rendezvous ...'.format(ordinal))
  data = xm.rendezvous('rendezvous_test',
                       'ORD={}'.format(ordinal).encode('utf-8'))
  print('Core {} got rendezvous!'.format(ordinal))
  for i in range(0, len(data)):
    idata = data[i].decode('utf-8')
    m = re.match(r'ORD=(\d+)', idata)
    assert m, 'Bad payload format: {}'.format(idata)
    xordinal = int(m.group(1))
    assert i == xordinal, 'Payload {} got ordinal {}'.format(i, xordinal)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())

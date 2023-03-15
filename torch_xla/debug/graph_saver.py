import collections
import os
import threading
import torch_xla

_SAVE_GRAPH_LOCK = threading.Lock()
_SAVE_GRAPH_IDS = collections.defaultdict(dict)


def save_tensors_graph(save_dir, name, tensors):
  fmt = os.environ.get('SAVE_GRAPH_FMT', 'text')
  if fmt == 'text':
    graph = torch_xla._XLAC._get_xla_tensors_text(tensors)
  elif fmt == 'dot':
    graph = torch_xla._XLAC._get_xla_tensors_dot(tensors)
  elif fmt == 'hlo':
    graph = torch_xla._XLAC._get_xla_tensors_hlo(tensors)
  else:
    raise RuntimeError('Invalid save graph format: {}'.format(fmt))
  tid = threading.current_thread().ident
  with _SAVE_GRAPH_LOCK:
    tdict = _SAVE_GRAPH_IDS[tid]
    id = tdict.get(name, 0)
    tdict[name] = id + 1
  fname = '{}-{}-{}.{}'.format(name, tid, id, fmt)
  with open(os.path.join(save_dir, fname), 'w') as fd:
    fd.write(graph)

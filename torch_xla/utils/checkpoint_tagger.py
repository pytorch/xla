from __future__ import division
from __future__ import print_function

from collections import defaultdict
import json

class CheckpointTagger(object):

  def __init__(self, remover=None):
    self._tags = dict()
    self._refcount = defaultdict(int)
    remover = (lambda x: None) if remover is None else remover
    assert callable(remover)
    self._remover = remover

  def tag(self, name, path):
    self._refcount[path] += 1
    old_path = self._tags.get(name)
    if old_path is not None:
      self._refcount[old_path] -= 1
      if self._refcount[old_path] == 0:
        self._refcount.pop(old_path)
        self._remover(old_path)
    self._tags[name] = path

  def dump(self):
    return json.dumps(self._tags)

  @classmethod
  def load(cls, dat, remover=None):
    instance = cls(remover=remover)
    if isinstance(dat, bytes):
      dat = dat.decode()
    if isinstance(dat, str):
      try:
        dat = json.loads(dat)
      except json.decoder.JSONDecodeError as e:
        raise type(e)(e.message + '\n\twas a filename passed?')
    for name, path in dat.items():
      instance.tag(name, path)
    return instance

from __future__ import division
from __future__ import print_function

import io
import os
import re
import torch
import torch_xla
import torch_xla.utils.gcsfs as gcs


def _index_split(index, split_size, split_count):
  parts = []
  while True:
    if parts:
      part = str(index % split_size)
    else:
      part = '{}.pt'.format(index)
    parts.append(part)
    index = index // split_size
    if index == 0:
      break
  while len(parts) < split_count:
    parts.append('0')
  parts.reverse()
  return parts


def _infer_sample_count(path):
  files = set()
  maxid = -1
  for fpath in gcs.generic_glob(os.path.join(path, '**'), recursive=True):
    fname = os.path.basename(fpath)
    m = re.match(r'(\d+)\.pt$', fname)
    if m:
      files.add(fpath)
      maxid = max(maxid, int(m.group(1)))
  assert maxid + 1 == len(files)
  return maxid + 1


class CachedDataset(torch.utils.data.Dataset):
  """Wraps an existing `torch.utils.data.Dataset` by providing file caching.

  The `CachedDataset` can be used to trade the CPU/RAM resources required to
  process a raw dataset, with storage/network resources.
  Example::

    train_dataset = datasets.MNIST(
        FLAGS.datadir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]))
    train_dataset = CachedDataset(train_dataset, FLAGS.dscache_dir)

  Args:
    data_set (torch.utils.data.Dataset): The raw `torch.utils.data.Dataset` to be
      cached. It can be set to `None` in case all the input samples are stored
      within the `path` folder.
    path (string): The path where the dataset samples should be stored/loaded.
      The `path` needs to be writeable, unless all the samples are already stored.
      The `path` can be a GCS path (prefixed with `gs://`).
    max_files_per_folder (int): The maximum amount of files to be stored within a
      single folder.
      Default: 1000
    compress (bool): Whether the saved samples should be compressed. Compression
      saves space at the expense of CPU required to compress/decompress.
      Default: True
  """

  def __init__(self, data_set, path, max_files_per_folder=1000, compress=True):
    super(CachedDataset, self).__init__()
    self._data_set = data_set
    self._path = path
    self._max_files_per_folder = max_files_per_folder
    self._compress = compress
    self._count = len(
        data_set) if data_set is not None else _infer_sample_count(path)
    self._split_count = len(_index_split(self._count, max_files_per_folder, 0))

  def _index_path(self, index):
    return os.path.join(
        self._path,
        *_index_split(index, self._max_files_per_folder, self._split_count))

  def _save_sample(self, data, path):
    bio = io.BytesIO()
    torch.save(data, bio, _use_new_zipfile_serialization=self._compress)
    gcs.generic_write(bio.getvalue(), path, makedirs=True)

  def _load_sample(self, path):
    try:
      data = gcs.generic_read(path)
      return torch.load(io.BytesIO(data))
    except:
      pass

  def warmup(self):
    for index in range(0, self._count):
      self.__getitem__(index)

  def __len__(self):
    return self._count

  def __getitem__(self, index):
    path = self._index_path(index)
    data = self._load_sample(path)
    if data is None:
      if self._data_set is None:
        raise RuntimeError(
            'Source dataset not provided and sample {} is missing from cache folder {}'
            .format(index, self._path))
      data = self._data_set[index]
      self._save_sample(data, path)
    return data

from __future__ import division
from __future__ import print_function

import os
import re
import torch

_SAVE_DIR = None
_TENSOR_IDS = {}
_STEP = None
_MAX_DIFFS = 25


def _index_of(sizes, lindex):
  index = []
  for size in reversed(sizes):
    index.append(lindex % size)
    lindex = lindex // size
  return list(reversed(index))


def _compare_tensors(tensor1, tensor2, rtol=1e-05, atol=1e-08, max_diffs=25):
  sizes1 = list(tensor1.size())
  sizes2 = list(tensor2.size())
  if sizes1 != sizes2:
    return 'Tensors have different shape: {} vs. {}\n'.format(sizes1, sizes2)

  report = ''
  diffno = 0
  values1 = tensor1.flatten().tolist()
  values2 = tensor2.flatten().tolist()
  for i in range(0, len(values1)):
    v1 = values1[i]
    v2 = values2[i]
    if abs(v1 - v2) > atol + rtol * abs(v2):
      if diffno >= max_diffs:
        report += '... aborting after {} differences\n'.format(diffno)
        break
      report += '{}: {} vs. {}\n'.format(_index_of(sizes1, i), v1, v2)
      diffno += 1
  return report


def _collect_saved_tensors(path):
  files = []
  for root, dirnames, filenames in os.walk(path):
    for fname in filenames:
      if re.match(r'.*\.\d+$', fname):
        files.append(fname)
  return set(files)


def configure(save_dir, max_diffs=25):
  global _SAVE_DIR, _MAX_DIFFS, _TENSOR_IDS, _STEP
  _SAVE_DIR = save_dir
  _MAX_DIFFS = max_diffs
  _TENSOR_IDS = {}
  _STEP = None


def save(name, tensor, step=None):
  global _TENSOR_IDS, _STEP
  if step is not None:
    path = os.path.join(_SAVE_DIR, 'step-{}'.format(step))
    if not os.path.isdir(path):
      os.mkdir(path)
    if step != _STEP:
      _STEP = step
      _TENSOR_IDS = {}
  else:
    path = _SAVE_DIR
  id = _TENSOR_IDS.get(name, 0)
  _TENSOR_IDS[name] = id + 1
  path = os.path.join(path, '{}.{}'.format(name, id))
  torch.save(tensor.data.cpu(), path)
  return tensor


def _parse_path(path):
  fname = os.path.basename(path)
  rpath = os.path.dirname(path)
  stepname = os.path.basename(rpath)
  step = None
  m = re.match(r'step-(\d+)$', stepname)
  if m:
    step = int(m.group(1))
    rpath = os.path.dirname(rpath)
  id = None
  m = re.match(r'(.*)\.(\d+)$', fname)
  assert m, fname
  return m.group(1), int(m.group(2)), step, rpath


def _explain_differences(path1, tensor1, path2, tensor2):
  name, id, step, _ = _parse_path(path1)
  report = 'Mismatch: {}.{} differs{}\n'.format(
      name, id, ' in step={}'.format(step) if step is not None else '')
  report += '{}\nvs.\n{}\n'.format(str(tensor1), str(tensor2))
  return report


def tensor_compare(path1, path2, rtol=1e-05, atol=1e-08):
  tensor1 = torch.load(path1)
  tensor2 = torch.load(path2)
  return _compare_tensors(
      tensor1, tensor2, rtol=rtol, atol=atol, max_diffs=_MAX_DIFFS)


def compare(save_dir1, save_dir2, rtol=1e-05, atol=1e-08):
  files1 = _collect_saved_tensors(save_dir1)
  files2 = _collect_saved_tensors(save_dir2)
  report = ''
  for path1 in files1:
    if path1 not in files2:
      report += 'Mismatch: {} not in {}\n'.format(path1, save_dir2)
    else:
      report += tensor_compare(
          os.path.join(save_dir1, path1),
          os.path.join(save_dir2, path1),
          rtol=rtol,
          atol=atol)
  for path2 in files2:
    if path2 not in files1:
      report += 'Mismatch: {} not in {}\n'.format(path2, save_dir1)
  return report

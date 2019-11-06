from __future__ import division
from __future__ import print_function

import torch_xla


class TfRecordReader(object):
  """Reads TfRecords or TfExamples.

  Args:
    path (string): The path to the file containing TfRecords.
    compression (string, optional): The compression type. The empty string for
      no compression, otherwise ``ZLIB`` or ``GZIP``.
      Default: No compression.
    buffer_size (int, optional): The size of the buffer to be used to read
      TfRecords.
      Default: 16 * 1024 * 1024
    transforms (dict, optional): A dictionary with the key matching the
      TfExample label name, and value which is either a callable which will be
      called to tranform the matching tensor data, or ``STR`` for string
      conversion.
  """

  def __init__(self,
               path,
               compression='',
               buffer_size=16 * 1024 * 1024,
               transforms=None):
    self._reader = torch_xla._XLAC._xla_create_tfrecord_reader(
        path, compression=compression, buffer_size=buffer_size)
    self._transforms = transforms

  def read_record(self):
    """Reads a TfRecord and returns the raw bytes.

    Returns:
      The raw bytes of the record, or ``None`` in case of EOF.
    """
    return torch_xla._XLAC._xla_tfrecord_read(self._reader)

  def read_example(self):
    """Reads a TfExample.

    Returns:
      In case of EOD returns ``None``, otherwise a dictionary whose keys
      are the feature names, and values the tensors containing their
      data.
    """
    ex = torch_xla._XLAC._xla_tfexample_read(self._reader)
    if self._transforms is None or ex is None:
      return ex
    return self._transform_example(ex)

  def _transform_example(self, ex):
    for lbl, data in ex.items():
      trs = self._transforms.get(lbl, None)
      if trs is not None:
        if callable(trs):
          ex[lbl] = trs(data)
        elif trs == 'STR':
          ex[lbl] = data.numpy().tobytes().decode('ascii')
        else:
          raise RuntimeError('Invalid transform: {}'.format(trs))
    return ex

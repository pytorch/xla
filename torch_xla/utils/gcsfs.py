from __future__ import division
from __future__ import print_function

import collections
import io
import os
import re
import tempfile
import sys
import torch_xla

GcsBlob = collections.namedtuple('GcsBlob', 'path size mtime isdir')


def _mkblob(path, fstat):
  return GcsBlob(
      path=path,
      size=fstat['length'],
      mtime=fstat['mtime_nsec'] * 1.0e-9,
      isdir=fstat['is_directory'])


def _slurp_file(path):
  fstat = torch_xla._XLAC._xla_tffile_stat(path)
  gcs_file = torch_xla._XLAC._xla_tffile_open(path)
  return torch_xla._XLAC._xla_tffile_read(gcs_file, 0, fstat['length'])


class WriteableFile(io.RawIOBase):

  def __init__(self, path, init_data=None, append=False):
    super(WriteableFile, self).__init__()
    self._path = path
    self._wfile = tempfile.NamedTemporaryFile()
    if init_data is not None:
      self._wfile.write(init_data)
      if not append:
        self._wfile.seek(0, os.SEEK_SET)

  def close(self):
    if self._wfile is not None:
      self._sync()
      self._wfile = None

  def _sync(self):
    self._wfile.flush()
    offset = self._wfile.tell()
    self._wfile.seek(0, os.SEEK_SET)
    write(self._path, self._wfile.read())
    self._wfile.seek(offset, os.SEEK_SET)

  @property
  def closed(self):
    return self._wfile is None

  def fileno(self):
    raise OSError('Not supported on GCS files: {}'.format(self._path))

  def isatty(self):
    return False

  def flush(self):
    self._sync()

  def readable(self):
    return True

  def writable(self):
    return True

  def tell(self):
    return self._wfile.tell()

  def seekable(self):
    return True

  def truncate(self, size=None):
    return self._wfile.truncate(size=size)

  def seek(self, offset, whence=os.SEEK_SET):
    return self._wfile.seek(offset, whence)

  def readline(self, size=-1):
    return self._wfile.readline(size=size)

  def readlines(self, hint=-1):
    return self._wfile.readlines(hint=hint)

  def writelines(self, lines):
    return self._wfile.writelines(lines)

  def read(self, size=-1):
    return self._wfile.read(size=-1)

  def readall(self):
    return self._wfile.readall()

  def readinto(self, bbuf):
    return self._wfile.readinto(bbuf)

  def write(self, bbuf):
    return self._wfile.write(bbuf)

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.close()


def open(path, mode='r', encoding='utf-8'):
  """Opens a Google Cloud Storage (GCS) file for reading or writing.

  Args:
    path (string): The GCS path of the file. Must be "gs://BUCKET_NAME/PATH"
      where ``BUCKET_NAME`` is the name of the GCS bucket, and ``PATH`` is a `/`
      delimited path.
    mode (string, optional): The open mode, similar to the ``open()`` API.
      Default: 'r'
    encoding (string, optional): The character encoding to be used to decode
      bytes into strings when opening in text mode.
      Default: 'utf-8'

  Returns:
    The GCS file object.
  """
  if mode.startswith('w'):
    return WriteableFile(path)
  if mode.startswith('a') or mode.startswith('r+'):
    try:
      data = _slurp_file(path)
    except:
      data = None
    return WriteableFile(path, init_data=data, append=mode.startswith('a'))
  data = _slurp_file(path)
  if mode.find('t') >= 0:
    return io.StringIO(data.decode(encoding))
  return io.BytesIO(data)


def list(path):
  """Lists the content of a GCS bucket.

  Args:
    path (string): The GCS path of the file. Must be "gs://BUCKET_NAME/PATH"
      where ``BUCKET_NAME`` is the name of the GCS bucket, and ``PATH`` is a `/`
      delimited path.

  Returns:
    A list of ``GcsBlob`` objects.
  """
  blobs = []
  for mpath in torch_xla._XLAC._xla_tffs_list(path):
    try:
      fstat = torch_xla._XLAC._xla_tffile_stat(mpath)
      blobs.append(_mkblob(mpath, fstat))
    except:
      pass
  return blobs


def stat(path):
  """Fetches the information of a GCS file.

  Args:
    path (string): The GCS path of the file. Must be "gs://BUCKET_NAME/PATH"
      where ``BUCKET_NAME`` is the name of the GCS bucket, and ``PATH`` is a `/`
      delimited path.

  Returns:
    A ``GcsBlob`` object.
  """
  fstat = torch_xla._XLAC._xla_tffile_stat(path)
  return _mkblob(path, fstat)


def remove(path):
  """Removes a GCS blob.

  Args:
    path (string): The GCS path of the file. Must be "gs://BUCKET_NAME/PATH"
      where ``BUCKET_NAME`` is the name of the GCS bucket, and ``PATH`` is a `/`
      delimited path.
  """
  torch_xla._XLAC._xla_tffs_remove(path)


def read(path):
  """Reads the whole content of a GCS blob.

  Args:
    path (string): The GCS path of the file. Must be "gs://BUCKET_NAME/PATH"
      where ``BUCKET_NAME`` is the name of the GCS bucket, and ``PATH`` is a `/`
      delimited path.

  Returns:
    The bytes stored within the GCS blob.
  """
  return _slurp_file(path)


def write(path, content):
  """Write a string/bytes or file into a GCS blob.

  Args:
    path (string): The GCS path of the file. Must be "gs://BUCKET_NAME/PATH"
      where ``BUCKET_NAME`` is the name of the GCS bucket, and ``PATH`` is a `/`
      delimited path.
    content (string, bytes or file object): The content to be written into
      ``path``.
  """
  if not isinstance(content, (bytes, str)):
    content = content.read()
  gcs_file = torch_xla._XLAC._xla_tffile_create(path)
  torch_xla._XLAC._xla_tffile_write(gcs_file, content)
  torch_xla._XLAC._xla_tffile_flush(gcs_file)

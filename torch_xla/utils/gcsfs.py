from __future__ import division
from __future__ import print_function

import builtins
import collections
import glob
import io
import locale
import os
import re
import tempfile
import sys
import torch_xla

GcsBlob = collections.namedtuple('GcsBlob', 'path size mtime isdir')

CLOUD_STORAGE_PREFIX = 'gs://'


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

  def __init__(self, path, init_data=None, append=False, encoding=None):
    super(WriteableFile, self).__init__()
    self._path = path
    self._encoding = encoding
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

  def _get_bytes(self, data):
    return data if isinstance(data, bytes) else data.encode(self._encoding)

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
    return self._wfile.write(self._get_bytes(bbuf))

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.close()


def open(path, mode='r', encoding=None):
  """Opens a Google Cloud Storage (GCS) file for reading or writing.

  Args:
    path (string): The GCS path of the file. Must be "gs://BUCKET_NAME/PATH"
      where ``BUCKET_NAME`` is the name of the GCS bucket, and ``PATH`` is a `/`
      delimited path.
    mode (string, optional): The open mode, similar to the ``open()`` API.
      Default: 'r'
    encoding (string, optional): The character encoding to be used to decode
      bytes into strings when opening in text mode.
      Default: None

  Returns:
    The GCS file object.
  """
  binary = mode.find('b') >= 0
  if encoding is None:
    encoding = locale.getpreferredencoding()
  if mode.startswith('w'):
    return WriteableFile(path, encoding=encoding)
  if mode.startswith('a') or mode.startswith('r+'):
    try:
      data = _slurp_file(path)
    except:
      data = None
    return WriteableFile(
        path, init_data=data, append=mode.startswith('a'), encoding=encoding)
  data = _slurp_file(path)
  if binary:
    return io.BytesIO(data)
  return io.StringIO(data.decode(encoding))


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


def rmtree(path):
  """Removes all the GCS blobs within a given path.

  Args:
    path (string): The GCS path of the file pattern or folder. Must be
      "gs://BUCKET_NAME/PATH" where ``BUCKET_NAME`` is the name of the GCS
        bucket, and ``PATH`` is a `/` delimited path.
  """
  if path.find('*') < 0:
    if not path.endswith('/'):
      path += '/'
    path += '*'
  ex = None
  for blob in list(path):
    try:
      if not blob.isdir:
        remove(blob.path)
    except Exception as e:
      ex = e
  if ex is not None:
    raise ex


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


def generic_open(path, mode='r', encoding=None):
  """Opens a file (GCS or not) for reding or writing.

  Args:
    path (string): The path of the file to be opened. If a GCS path, it must be
      "gs://BUCKET_NAME/PATH" where ``BUCKET_NAME`` is the name of the GCS
        bucket, and ``PATH`` is a `/` delimited path.
    mode (string, optional): The open mode, similar to the ``open()`` API.
      Default: 'r'
    encoding (string, optional): The character encoding to be used to decode
      bytes into strings when opening in text mode.
      Default: None

  Returns:
    The opened file object.
  """
  if path.startswith(CLOUD_STORAGE_PREFIX):
    return open(path, mode=mode, encoding=encoding)
  else:
    return builtins.open(path, mode=mode, encoding=encoding)


def generic_write(output_string, output_path, makedirs=False):
  """Write a string/bytes or file into a GCS blob or local disk.

  Depending on the output_path passed in, this API can write to local or GCS
  file. Checks if the `output_path` starts with
  the 'gs://' prefix, and uses `open` otherwise.

  Args:
    output_string (string): The string to be written to the output.
    output_path (string): The GCS path or local path of the output.
    makedirs (bool): Whether the `path` parent folders should be created if
      missing.
      Default: False
  """
  if output_path.startswith(CLOUD_STORAGE_PREFIX):
    write(output_path, output_string)
  else:
    if makedirs:
      dpath = os.path.dirname(output_path)
      if not os.path.isdir(dpath):
        os.makedirs(dpath, exist_ok=True)
    mode = 'wb' if isinstance(output_string, bytes) else 'wt'
    with builtins.open(output_path, mode=mode) as fd:
      fd.write(output_string)


def generic_read(path):
  """Reads the whole content of the provided location.

  Args:
    path (string): The GCS path or local path to be read.

  Returns:
    The bytes stored within the GCS blob or local file.
  """
  if path.startswith(CLOUD_STORAGE_PREFIX):
    return read(path)
  else:
    with builtins.open(path, mode='rb') as fd:
      return fd.read()


def generic_glob(path, recursive=False):
  """Lists all the names within a specified path.

  Args:
    path (string): The path to be listed (can have wildcards), either local
      file system, or GCS.
    recursive (bool): Whether the glob operation should recurse into subdirectories.

  Returns:
    The names list within the provided path.
  """
  if path.startswith(CLOUD_STORAGE_PREFIX):
    return torch_xla._XLAC._xla_tffs_list(path)
  else:
    return glob.glob(path, recursive=recursive)

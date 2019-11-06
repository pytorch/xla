# To use this module the Google Cloud Storage libraries need to be installed:
#
#  pip install --upgrade google-cloud-storage
#
# For information on how to setup authentication follow the link below:
#
#  https://cloud.google.com/storage/docs/reference/libraries
#

from __future__ import division
from __future__ import print_function

import collections
import io
import os
import re
import tempfile
import sys

try:
  from google.cloud import storage as gcs
except:
  msg = """Google Cloud Storage libraries are missing.
Please install them using the following command:

  pip install --upgrade google-cloud-storage

Also follow the instructions in the link below to configure authentication:

  https://cloud.google.com/storage/docs/reference/libraries

"""
  print(msg, file=sys.stderr)
  raise

GcsBlob = collections.namedtuple('GcsBlob', 'path size')


class WriteableFile(io.RawIOBase):

  def __init__(self, path, init_data=None, append=False):
    super(WriteableFile, self).__init__()
    self._path = path
    self._gcs_client = gcs.Client()
    self._blob = gcs.Blob.from_string(path, client=self._gcs_client)
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
    self._blob.upload_from_file(self._wfile)
    self._wfile.seek(offset, os.SEEK_SET)

  @property
  def closed(self):
    return self._closed

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
  gcs_client = gcs.Client()
  blob = gcs.Blob.from_string(path, client=gcs_client)
  if mode.startswith('a') or mode.startswith('r+'):
    data = blob.download_as_string() if blob.exists() else None
    return WriteableFile(path, init_data=data, append=mode.startswith('a'))
  data = blob.download_as_string()
  if mode.find('t') >= 0:
    return io.StringIO(data.decode(encoding))
  return io.BytesIO(data)


def _get_blob_path(bpath):
  # The paths returned by the list_blobs() API have the
  # '/b/BUCKET_NAME/o/BLOB_PATH' format.
  m = re.match(r'/b/[^/]+/o/(.+)', bpath)
  if not m:
    raise RuntimeError('GCS invalid blob path: {}'.format(bpath))
  return m.group(1)


def _parse_gcs_path(path, wants_path=True):
  m = re.match(r'gs://([^/]+)(.*)', path)
  if not m:
    raise ValueError('GCS invalid path: {}'.format(path))
  if len(m.groups()) > 1:
    bpath = m.group(2)
    if bpath.startswith('/'):
      bpath = bpath[1:]
  else:
    bpath = None
  if not bpath and wants_path:
    raise RuntimeError('GCS path missing: {}'.format(path))
  return m.group(1), bpath


def list(path):
  """Lists the content of a GCS bucket.

  Args:
    path (string): The GCS path of the file. Must be "gs://BUCKET_NAME/PATH"
      where ``BUCKET_NAME`` is the name of the GCS bucket, and ``PATH`` is a `/`
      delimited path.

  Returns:
    A list of ``GcsBlob`` object, having ``path`` and ``size`` fields.

  Raises:
    ValueError: If an invalid GCS path is supplied.
  """
  bucket_name, bpath = _parse_gcs_path(path, wants_path=False)
  gcs_client = gcs.Client()
  blobs = []
  for blob in gcs_client.list_blobs(bucket_name, prefix=bpath, delimiter='/'):
    blobs.append(GcsBlob(path=_get_blob_path(blob.path), size=blob.size))
  return blobs


def remove(path):
  """Removes a GCS blob.

  Args:
    path (string): The GCS path of the file. Must be "gs://BUCKET_NAME/PATH"
      where ``BUCKET_NAME`` is the name of the GCS bucket, and ``PATH`` is a `/`
      delimited path.

  Raises:
    ValueError: If an invalid GCS path is supplied.
  """
  bucket_name, bpath = _parse_gcs_path(path)
  gcs_client = gcs.Client()
  bucket = gcs_client.get_bucket(bucket_name)
  bucket.delete_blob(bpath)


def write(path, content, service_account_filename=None):
  """Write a string/bytes or file into a GCS blob.

  Args:
    path (string): The GCS path of the file. Must be "gs://BUCKET_NAME/PATH"
      where ``BUCKET_NAME`` is the name of the GCS bucket, and ``PATH`` is a `/`
      delimited path.
    content (string, bytes or file object): The content to be written into
      ``path``.
    service_account_filename (string, optional): Filename of the JSON service
      account key to authenticate writes to Google Cloud Storage.
  """
  bucket_name, bpath = _parse_gcs_path(path)
  if service_account_filename:
    gcs_client = gcs.Client.from_service_account_json(
        service_account_filename)
  else:
    gcs_client = gcs.Client()
  bucket = gcs_client.get_bucket(bucket_name)
  blob = bucket.blob(bpath)
  if isinstance(content, (bytes, str)):
    blob.upload_from_string(content)
  else:
    blob.upload_from_file(content)

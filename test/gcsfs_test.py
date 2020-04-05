import locale
import os
import shutil
import sys
import unittest
import uuid
import torch_xla.utils.gcsfs as gcs

_TEST_PATH = os.environ.get('GCS_TEST_PATH', None)
_TEST_UUID = str(uuid.uuid1())


def _gcs_test_path(name=None):
  path = os.path.join(_TEST_PATH, _TEST_UUID)
  return os.path.join(path, name) if name else path


def _local_test_path(name=None):
  path = os.environ.get('TMPDIR', '/tmp')
  path = os.path.join(path, _TEST_UUID)
  if not os.path.isdir(path):
    os.mkdir(path, mode=0o755)
  return os.path.join(path, name) if name else path


def _global_cleanup():
  try:
    shutil.rmtree(_local_test_path())
  except:
    pass
  try:
    gcs.rmtree(_gcs_test_path())
  except:
    pass


def _create_content(mode, content=None, size=None):
  if content is None:
    content = os.urandom(size)
  encoding = locale.getpreferredencoding()
  if mode.find('b') >= 0:
    return content if isinstance(content, bytes) else content.decode(encoding)
  else:
    return content if isinstance(content, str) else content.encode(encoding)


def _create_gcs_file(path, mode, content=None, size=None, cleanup=None):
  content = _create_content(mode, content=content, size=size)
  with gcs.open(path, mode=mode) as fd:
    fd.write(content)
  if cleanup is not None:
    cleanup.append(lambda: gcs.remove(path))
  return content


def _create_file(path, mode, content=None, size=None, cleanup=None):
  content = _create_content(mode, content=content, size=size)
  with open(path, mode=mode) as fd:
    fd.write(content)
  if cleanup is not None:
    cleanup.append(lambda: os.remove(path))
  return content


class GcsTest(unittest.TestCase):

  def setUp(self):
    self._cleanup = []

  def tearDown(self):
    for fn in self._cleanup:
      try:
        fn()
      except:
        pass

  def test_write_compare(self):
    SIZE = 10000000  # 10MB
    FNAME = 'test_write_compare'
    gcs_path = _gcs_test_path(name=FNAME)
    content = _create_gcs_file(gcs_path, 'wb', size=SIZE, cleanup=self._cleanup)
    rcontent = gcs.read(gcs_path)
    self.assertEqual(len(content), len(rcontent))
    self.assertEqual(type(content), type(rcontent))
    self.assertEqual(content, rcontent)

  def test_text_write_compare(self):
    CONTENT = """ABC
CD E
FZXY
"""
    FNAME = 'test_text_write_compare'
    gcs_path = _gcs_test_path(name=FNAME)
    content = _create_gcs_file(gcs_path, 'w', content=CONTENT, cleanup=self._cleanup)
    rcontent = gcs.read(gcs_path).decode(locale.getpreferredencoding())
    self.assertEqual(len(content), len(rcontent))
    self.assertEqual(type(content), type(rcontent))
    self.assertEqual(content, rcontent)

  def test_list(self):
    SIZE = 10000000  # 10MB
    FNAME = 'test_list'
    gcs_path = _gcs_test_path(name=FNAME)
    content = _create_gcs_file(gcs_path, 'wb', size=SIZE, cleanup=self._cleanup)
    self.assertEqual(len(content), SIZE)
    blobs = gcs.list(gcs_path)
    self.assertEqual(len(blobs), 1)
    self.assertEqual(blobs[0].size, SIZE)
    self.assertFalse(blobs[0].isdir)


if __name__ == '__main__':
  if _TEST_PATH is not None:
    try:
      unittest.main()
    except:
      _global_cleanup()
      raise
  else:
    print('The GCS_TEST_PATH environment variable must be set', file=sys.stderr)

import argparse
import os
import subprocess

import torch_xla.utils.gcsfs as gcsfs


_CLOUD_STORAGE_PREFIX = 'gs://'
_TMP_FILENAME = '/tmp/pytorch_tpu_metrics_tmp'
_XLA_METRICS_FILE = 'XLA_METRICS_FILE'

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='PyTorch on TPU distrubuted training',
      epilog=('Usage example: metrics_test_wrapper.py '
          '--golden_metrics_filename="gcs://bucket/file" '
          '--metrics_output_filename="gcs://bucket/file" '
          '--service_account_filename="path/to/file" -- python train.py'))
  parser.add_argument('--golden_metrics_filename', type=str, default=None)
  parser.add_argument('--metrics_output_filename', type=str, default=None)
  parser.add_argument('--service_account_filename', type=str, default=None)
  parser.add_argument(
      'positional',
      nargs='+',
      type=str,
      help='The python command to run.')
  FLAGS = parser.parse_args()
  if not (FLAGS.golden_metrics_filename or FLAGS.metrics_output_filename):
    raise ValueError('At least one of golden_metrics_filename or '
                     'metrics_output_filename is required.')

  os.environ[_XLA_METRICS_FILE] = _TMP_FILENAME
  print("CALLING SUBPROCESS: {}".format(FLAGS.positional))
  subprocess.call(FLAGS.positional)
  print("AFTER SUBPROCESS")

  import pdb; pdb.set_trace()
  with open(_TMP_FILENAME, 'r') as tmp_metrics:
    metrics = tmp_metrics.readlines()

  if FLAGS.metrics_output_filename:
    try:
      if FLAGS.metrics_output_filename.find(_CLOUD_STORAGE_PREFIX) == 0:
        gcsfs.write(FLAGS.metrics_output_filename, metrics)
    except Exception as e:
      print('Failed to write metrics to file: {}'.format(e))

  if FLAGS.golden_metrics_filename:
    # TODO(zcain): Read golden metrics file and compare to current metrics.
    pass

  os.remove(_TMP_FILENAME)
  del os.environ[_XLA_METRICS_FILE]

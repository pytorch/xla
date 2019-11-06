"""Wrap PyTorch XLA jobs with extra logic around XLA metrics.

It can be used to save a historical record of metrics and/or check metrics
against a golden set of metrics to search for performance regressions.

Example usage:
python metrics_test_wrapper.py --metrics_output_filename="metrics.txt" \
    -- python test/test_train_mnist.py --num_epochs=1
"""
import argparse
import os
import subprocess
import sys

import torch_xla.utils.gcsfs as gcsfs


_CLOUD_STORAGE_PREFIX = 'gs://'
_TMP_FILENAME = '/tmp/pytorch_tpu_metrics_tmp'
_XLA_METRICS_FILE = 'XLA_METRICS_FILE'


def _run_subprocess(cmd):
  # Use _XLA_METRICS_FILE to pass the metrics report from the subprocess
  # to this process.
  os.environ[_XLA_METRICS_FILE] = _TMP_FILENAME
  sp_return_code = subprocess.call(cmd)
  with open(_TMP_FILENAME, 'r') as tmp_metrics:
    metrics = tmp_metrics.read()
  os.remove(_TMP_FILENAME)
  del os.environ[_XLA_METRICS_FILE]

  # Use only the metrics from the last step.
  metrics = metrics[metrics.rfind('[MetricsData;'):]
  return metrics, sp_return_code


def _write_to_disk(output_string, output_filename,
                   service_account_filename=None):
  if not output_string:
    return
  try:
    if output_filename.find(_CLOUD_STORAGE_PREFIX) == 0:
      gcsfs.write(output_filename, output_string,
                  service_account_filename=service_account_filename)
    else:
      with open(output_filename, 'w') as outfile:
        outfile.write(output_string)
  except Exception as e:
    print('Failed to write metrics to file: {}'.format(e))


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

  metrics, sp_return_code = _run_subprocess(FLAGS.positional)

  # Include the params for this invocation when saving metrics.
  output_string = '{}\n\n{}'.format(FLAGS, metrics)
  _write_to_disk(output_string, FLAGS.metrics_output_filename,
                 service_account_filename=FLAGS.service_account_filename)

  if FLAGS.golden_metrics_filename:
    # TODO(zcain): Read golden metrics file and compare to current metrics.
    pass

  sys.exit(sp_return_code)

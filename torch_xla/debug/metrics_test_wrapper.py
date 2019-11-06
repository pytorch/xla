"""Wrap PyTorch XLA jobs with extra logic around XLA metrics.

It can be used to save a historical record of metrics and/or check metrics
against a golden set of metrics to search for performance regressions.

Example usage:
python metrics_test_wrapper.py --metrics_output_filename="metrics.txt" \
    -- python test/test_train_mnist.py --num_epochs=1
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import torch_xla.utils.gcsfs as gcsfs


_CLOUD_STORAGE_PREFIX = 'gs://'
_XLA_METRICS_FILE = 'XLA_METRICS_FILE'


def _run_subprocess(cmd, distributed=False):
  # Use _XLA_METRICS_FILE to pass the metrics report from the subprocess
  # to this process. Note that distributed training jobs write 1 file per
  # core.
  tmp_dir = tempfile.mkdtemp()
  _, tmp_path = tempfile.mkstemp(dir=tmp_dir)
  file_to_read = '{}.0'.format(tmp_path) if distributed else tmp_path

  # Run the PyTorch XLA script.
  os.environ[_XLA_METRICS_FILE] = tmp_path
  sp_return_code = subprocess.call(cmd)
  with open(file_to_read, 'r') as tmp_metrics:
    metrics = tmp_metrics.read()

  # Cleanup.
  shutil.rmtree(tmp_dir)
  del os.environ[_XLA_METRICS_FILE]

  # Use only the metrics from the last step.
  metrics = metrics[metrics.rfind('[MetricsData;'):]
  return metrics, sp_return_code


def _write_to_disk(output_string, output_filename):
  if not output_filename:
    return
  try:
    if output_filename.find(_CLOUD_STORAGE_PREFIX) == 0:
      gcsfs.write(output_filename, output_string)
    else:
      with open(output_filename, 'w') as outfile:
        outfile.write(output_string)
    print('Succeeded writing metrics to file: {}'.format(output_filename))
  except Exception as e:
    print('Failed writing metrics to file: {}'.format(e))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
          description='PyTorch on TPU: Verify metrics and/or save to disk.',
      epilog=('Usage example: metrics_test_wrapper.py '
          '--golden_metrics_filename="gcs://bucket/file" '
          '--metrics_output_filename="gcs://bucket/file" '
          '--service_account_filename="path/to/file" -- python train.py'))
  parser.add_argument('--golden_metrics_filename', type=str, default=None,
                      help='Read metrics from here and compare to current.')
  parser.add_argument('--metrics_output_filename', type=str, default=None,
                      help='If provided, write current metrics here.')
  parser.add_argument('--service_account_filename', type=str, default=None,
                      help='Used to authenticate to Google Cloud Storage.')
  parser.add_argument('--distributed', action='store_true',
                      help='Whether the subprocess is a distributed job.')
  parser.add_argument(
      'positional',
      nargs='+',
      type=str,
      help='The python command to run.')
  FLAGS = parser.parse_args()
  if not (FLAGS.golden_metrics_filename or FLAGS.metrics_output_filename):
    raise ValueError('At least one of golden_metrics_filename or '
                     'metrics_output_filename is required.')

  metrics, sp_return_code = _run_subprocess(FLAGS.positional,
                                            distributed=FLAGS.distributed)

  # Include the params for this invocation when saving metrics.
  output_string = '{}\n\n{}'.format(FLAGS, metrics)
  if FLAGS.service_account_filename:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = FLAGS.service_account_filename
  _write_to_disk(output_string, FLAGS.metrics_output_filename)

  if FLAGS.golden_metrics_filename:
    # TODO(zcain): Read golden metrics file and compare to current metrics.
    pass

  sys.exit(sp_return_code)

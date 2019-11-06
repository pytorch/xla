"""Wrap PyTorch XLA training jobs with extra logic around XLA metrics.

It can be used to save a historical record of metrics and/or check metrics
against a golden set of metrics to search for performance regressions.

Example usage:
python metrics_test_wrapper.py --metrics_output_filename="metrics.txt" \
    -- python test/test_train_mnist.py --num_epochs=1
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import torch_xla.utils.gcsfs as gcsfs


_CLOUD_STORAGE_PREFIX = 'gs://'
_XLA_METRICS_FILE = 'XLA_METRICS_FILE'


def _find_correct_metrics_file(base_metrics_file):
  # Distributed training jobs write 1 file per core. If XLA_METRICS_FILE is set
  # to 'base_metrics_file', then the distributed output files have the format
  # 'base_metrics_file.0', 'base_metrics_file.1', etc.
  metrics_files = glob.glob('{}*'.format(base_metrics_file))
  if len(metrics_files) == 1:
      # Non-distributed case: the correct file is simply 'base_metrics_file'.
    file_to_read = base_metrics_file
  else:
    # Choose the file of the first core. When sorted alphabetically, the first
    # element will be 'base_metrics_file' which is empty in the distributed
    # case. The second element will be 'base_metrics_file.0'.
    file_to_read = sorted(metrics_files)[1]
  return file_to_read


def _run_subprocess(cmd):
  # Use _XLA_METRICS_FILE to pass the metrics report from the subprocess
  # to this process.
  tmp_dir = tempfile.mkdtemp()
  _, tmp_path = tempfile.mkstemp(dir=tmp_dir)

  # Run the PyTorch XLA script.
  os.environ[_XLA_METRICS_FILE] = tmp_path
  sp_return_code = subprocess.call(cmd)

  with open(_find_correct_metrics_file(tmp_path), 'r') as tmp_metrics:
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
          '--metrics_output_filename="gcs://bucket/file" -- python train.py'))
  parser.add_argument('--golden_metrics_filename', type=str, default=None,
                      help='Read metrics from here and compare to current.')
  parser.add_argument('--metrics_output_filename', type=str, default=None,
                      help='If provided, write current metrics here.')
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
  _write_to_disk(output_string, FLAGS.metrics_output_filename)

  if FLAGS.golden_metrics_filename:
    # TODO(zcain): Read golden metrics file and compare to current metrics.
    pass

  sys.exit(sp_return_code)

"""Wrap PyTorch XLA training jobs with extra logic around XLA metrics.

It can be used to save a historical record of metrics and/or check metrics
against a golden set of metrics to search for performance regressions.

This script expects a certain directory structure to setup tests and to hold
metrics results. In the outermost directory, there should be a base config file
detailed below. Within that directory, there should be a subdirectory for each
test that contains a metrics_history subdirectory and (optionally) a config file
to override fields of the base config. In other words, it should look like this:

    root/
      base_config.json
      mnist/
        config_overrides.json
        metrics_history/

TODO(zcain): Design and document the config file.

Example usage:
python metrics_test_wrapper.py --root="gs://model_metrics" \
    --test_folder_name="mnist" -- python test/test_train_mnist.py --num_epochs=1
"""
import argparse
import datetime
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import torch_xla.utils.gcsfs as gcsfs


_CLOUD_STORAGE_PREFIX = 'gs://'
_METRICS_HISTORY_DIR_NAME = 'metrics_history'
_XLA_METRICS_FILE = 'XLA_METRICS_FILE'


def _find_correct_metrics_file(base_metrics_file):
  # Distributed training jobs write 1 file per core. If XLA_METRICS_FILE is set
  # to 'base_metrics_file', then the distributed output files have the format
  # 'base_metrics_file.0', 'base_metrics_file.1', etc. Non-distributed training
  # will simply write to 'base_metrics_file'.
  metrics_files = glob.glob('{}*'.format(base_metrics_file))

  # Use 'base_metrics_file' if it contains the metrics, otherwise use the
  # metrics file from the first numbered core, e.g. 'base_metrics_file.0'.
  for filename in sorted(metrics_files):
    if os.path.getsize(filename) > 0:
      return filename

  raise FileNotFoundError(
      'No non-empty file was found matching: {}'.format(base_metrics_file))


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
          '--root="gs://model_metrics" '
          '--test_name="mnist" -- python train.py'))
  parser.add_argument('--root', type=str, default=None,
                      help='Root dir for metrics test data. See docstring at '
                      'the top of this script.')
  parser.add_argument('--test_folder_name', type=str, default=None,
                      help='Folder within root/ for this test. See docstring '
                      'at the top of this script.')
  parser.add_argument(
      'positional',
      nargs='+',
      type=str,
      help='The python command to run.')
  FLAGS = parser.parse_args()
  if not FLAGS.root or not FLAGS.test_folder_name:
    raise ValueError('root and test_folder_name are required arguments.')

  # TODO(zcain): Verify that root contains base_config.
  # TODO(zcain): Read tolerances from base_config and maybe override with
  #              test-specific config from test_folder_name.
  # TODO(zcain): Read metrics_history for current test, retrieve N days of
  #              history.
  # TODO(zcain): Calculate mean and std dev for each metrics, then compare
  #              against metrics from the subprocess call.
  
  metrics, sp_return_code = _run_subprocess(FLAGS.positional)

  # Include the params for this invocation when saving metrics.
  output_string = '{}\n\n{}'.format(FLAGS, metrics)
  output_filename = os.path.join(
      FLAGS.root, FLAGS.test_folder_name, _METRICS_HISTORY_DIR_NAME,
      datetime.datetime.utcnow().strftime('%Y_%m_%d'))
  _write_to_disk(output_string, output_filename)

  sys.exit(sp_return_code)

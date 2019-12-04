"""Wrap PyTorch XLA training jobs with extra logic around XLA metrics.

It can be used to save a historical record of metrics and/or check metrics
against a golden set of metrics to search for performance regressions.

This script expects a certain directory structure to setup tests and to hold
metrics results. In the outermost directory, there should be a base config file
detailed below. Within that directory, there should be a subdirectory for each
test that contains a metrics_history subdirectory and (optionally) a config file
named 'config.json' to override fields of the base config. In other words, it
should look like this:

    root/
      config.json
      mnist/
        config.json
        metrics_history/

config.json has 2 relevant fields:
  1. `write_metrics_to_disk`: (bool) If false, this wrapper will not write
     any XLA metrics to disk.
  2. `regression_test_config`: (dict) The config that will be used to determine
     whether any metrics have regressed in a meaningful way. If absent, this
     wrapper script will not perform any regression checks. For more details,
     see `compare_metrics` in `torch_xla/debug/metrics_compare_utils`. The
     config allows different checks for individual metrics.

The config.json found in root/ is used as the base and any config.json files
found in child directories will overwrite some or all of the base config for
that specific test. Simple recommended starter config.json:

{
  "write_metrics_to_disk": true,
  "regression_test_config": {
    "base_expression": "v <= v_mean + (v_stddev * 3.0)"
  }
}

Example usage:
python metrics_test_wrapper.py --root="gs://model_metrics" \
    --test_folder_name="mnist" -- python test/test_train_mnist.py --num_epochs=1
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import torch_xla.debug.metrics_compare_utils as mcu
import torch_xla.utils.gcsfs as gcsfs

try:
  import google.api_core.exceptions
except:
  msg = """Google Cloud Storage libraries are missing.
Please install them using the following command:

  pip install --upgrade google-cloud-storage

Also follow the instructions in the link below to configure authentication:

  https://cloud.google.com/storage/docs/reference/libraries
"""
  print(msg, file=sys.stderr)
  raise


_CLOUD_STORAGE_PREFIX = 'gs://'
_METRICS_HISTORY_DIR_NAME = 'metrics_history'
_XLA_METRICS_FILE = 'XLA_METRICS_FILE'
_METRICS_FILE_PATTERN = r'.*\d{4}_\d{2}_\d{2}'


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

  # Run the user-supplied command.
  metrics, sp_return_code = _run_subprocess(FLAGS.positional)

  # Retrieve any config files that affect this test.
  # NOTE: these are ordered in increasing specificity. For example, if there
  # was a base config that affects all tests and a specific config for a
  # particular test, then the base config will be the first element in the
  # list and the most specific config will be the last element.
  ordered_config_dicts = []

  path_to_search = FLAGS.test_folder_name
  while True:
    try:
      f = gcsfs.open(os.path.join(FLAGS.root, path_to_search, 'config.json'))
      ordered_config_dicts.insert(0, json.load(f))
    except google.api_core.exceptions.NotFound:
      pass
    if path_to_search == '':
      break
    path_to_search = os.path.split(path_to_search)[0]
  if not ordered_config_dicts:
    print('No config files found. See example usage at top of '
          'metrics_test_wrapper.py')
    sys.exit(1)  # Return non-OK status since config is required.

  # Consolidate configs into 1 dict by overwriting the least-specific configs
  # with the increasingly more-specific configs.
  config = ordered_config_dicts[0]
  for c in ordered_config_dicts:
    config.update(c)

  # Collect historical metrics for this test and check for any regressions in
  # the current run vs. the averages from previous runs.
  metrics_storage_dir = os.path.join(
      FLAGS.root, FLAGS.test_folder_name, _METRICS_HISTORY_DIR_NAME)
  metrics_storage_dir += '/'
  regression_test_config = config.get('regression_test_config', None)
  if regression_test_config:
    metrics_file_pattern = re.compile(_METRICS_FILE_PATTERN)
    prev_metrics_files = [f for f in gcsfs.list(
        metrics_storage_dir) if metrics_file_pattern.match(f.path)]
    prev_metrics_strings = [gcsfs.open(
        os.path.join(FLAGS.root, f.path), mode='rt').read() for f in
        prev_metrics_files]
    data_points = mcu.get_data_points_from_metrics_reports(
        prev_metrics_strings)
    regression_report = mcu.compare_metrics(
        data_points, metrics, config=regression_test_config)
  else:
    print('Unable to check for metrics regressions. Config should contain '
          '"regression_test_config" key -- see example at the top of '
          'metrics_test_wrapper.py.')
    regression_report = ''

  # Write the metrics from the current run to disk unless disabled by config.
  if config.get('write_metrics_to_disk', True):
    # Include the params for this invocation when saving metrics.
    output_string = '{}\n\n{}'.format(FLAGS, metrics)
    output_filename = os.path.join(
        metrics_storage_dir, 'ZCAIN_TEST_' + datetime.datetime.utcnow().strftime('%Y_%m_%d'))
    _write_to_disk(output_string, output_filename)

  if regression_report:
    print('Metrics regression report:\n{}'.format(regression_report))
    sys.exit(1)  # Return non-OK status code since there was a regression.
  sys.exit(sp_return_code)

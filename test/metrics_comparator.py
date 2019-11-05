"""Methods for tracking and comparing XLA metrics over time."""

_GCS_PREFIX = "gcs://"

def metrics_have_regressed(current_metrics, golden_metrics_filename,
                           output_filename=None, prefix=None,
                           service_account_filename=None):
  """Return True if current_metrics regressed compared to golden metrics.

  TODO(zcain): More detail on which regressions we check for.

  Optionally save a copy of current_metrics to disk.

  Args:
    current_metrics (string): Output of XLA metrics report.
    golden_metrics_filename (string): File containing the golden metrics to
      compare to current_metrics. Note for Google Cloud Storage: this should
      be in the format "gcs://bucket_name/filename".
    output_filename (string, optional): If provided, the current_metrtics
      will be written out to this file.
    prefix (string, optional): If provided, this prefix will precede
      current_metrics in the output file.
    service_account_filename (string, optional): Filename of the JSON service
      account key to authenticate reads/writes to Google Cloud Storage.

  Returns:
    True if current_metrics shows regression compared to golden metrics.
  """
  if output_filename:
    try:
      _write_metrics_to_file(current_metrics, output_filename, prefix=prefix,
                            service_account_filename=service_account_filename)
    except Exception as e:
      print("Failure writing metrics to disk: {}".format(e))

  # TODO(zcain): Define regression, add functionality to read metrics, add
  # logic to compare metrics.
  return False
  

def _write_metrics_to_file(metrics, filename, prefix=None,
                           service_account_filename=None):
  """Write metrics to GCS or to local filesystem.

  Args:
    metrics (string): Metrics to write to file.
    filename (string): Write destination. Note for Google Cloud Storage: this
      should be in the format "gcs://bucket_name/filename".
    prefix (string, optional): Optional text to be written to the output file
      just before the metrics.
    service_account_filename (string, optional): Filename of the JSON service
      account key to authenticate writes to Google Cloud Storage. Not necessary
      for writing to local filesytem.
  """
  output_string = "Metrics:\n{}".format(metrics)
  if prefix:
    output_string = "{}\n\n{}".format(prefix, output_string)
  if filename.find(_GCS_PREFIX) >= 0:
    import gcs_utils
    gcs_utils.write_to_gcs(service_account_filename,
                           filename,
                           output_string)
  else:
    with open(filename, 'w') as outfile:
      outfile.write(output_string)

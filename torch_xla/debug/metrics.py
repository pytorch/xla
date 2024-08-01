import torch_xla


def counter_names():
  """Retrieves all the currently active counter names."""
  return torch_xla._XLAC._xla_counter_names()


def counter_value(name):
  """Returns the value of an active counter.

  Args:
    name (string): The name of the counter whose value needs to be retrieved.

  Returns:
    The counter value as integer.
  """
  return torch_xla._XLAC._xla_counter_value(name)


def clear_counters():
  """Clear the value of all counters.
  """
  return torch_xla._XLAC._clear_xla_counters()


def metric_names():
  """Retrieves all the currently active metric names."""
  return torch_xla._XLAC._xla_metric_names()


def metric_data(name):
  """Returns the data of an active metric.

  Args:
    name (string): The name of the metric whose data needs to be retrieved.

  Returns:
    The metric data, which is a tuple of (TOTAL_SAMPLES, ACCUMULATOR, SAMPLES).
    The `TOTAL_SAMPLES` is the total number of samples which have been posted to
    the metric. A metric retains only a given number of samples (in a circular
    buffer).
    The `ACCUMULATOR` is the sum of the samples over `TOTAL_SAMPLES`.
    The `SAMPLES` is a list of (TIME, VALUE) tuples.
  """
  return torch_xla._XLAC._xla_metric_data(name)


def clear_metrics():
  """Clear the value of all metrics.
  """
  return torch_xla._XLAC._clear_xla_metrics()


def clear_all():
  """Clear the value of all metrics and all counters.
  """
  clear_metrics()
  clear_counters()


def metrics_report():
  """Retrieves a string containing the full metrics and counters report."""
  return torch_xla._XLAC._xla_metrics_report()


def short_metrics_report(counter_names: list = None, metric_names: list = None):
  """Retrieves a string containing the full metrics and counters report.

  Args:
    counter_names (list): The list of counter names whose data needs to be printed.
    metric_names (list): The list of metric names whose data needs to be printed.
  """
  if not counter_names:
    counter_names = ['CachedCompile', 'MarkStep', 'DynamoSyncInputExecuteTime']
  if not metric_names:
    metric_names = [
        'CompileTime', 'ExecuteTime', 'ExecuteReplicatedTime',
        'TransferToDeviceTime', 'TransferFromDeviceTime'
    ]
  return torch_xla._XLAC._short_xla_metrics_report(counter_names, metric_names)


def executed_fallback_ops():
  """Retrieves a list of operations that were run in fallback mode."""
  return torch_xla._XLAC._get_executed_fallback_ops()

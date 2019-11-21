"""Logic to compare XLA metrics and identify significant changes."""

_METRIC_NAME_AND_SAMPLES_AND_ACCUMULATOR = \
  "Metric: (\S+)\s+TotalSamples: (\d+)\s+Accumulator: (\S+)"
_COUNTER_NAME_AND_VALUE = "Counter: (\S+)\s+Value: (\d+)"

def _parse_metrics_report(report):
  import pdb; pdb.set_trace()

  data_points = {}
  metrics_lines = metrics_data_string.splitlines()
  for i in range(metrics_lines):
    line = metrics_lines[i].strip()
    if line.startswith(_METRICS_LINE_START):
      metric_name = line.replace(_METRICS_LINE_START, '')
      i+=1
      # find TotalSamples line
      # find Accumulator line
      # convert to standard metric, e.g. KB/MB/GB/TB to bytes and d/h/m/s/ms/us to ?s?
      pass
    elif line.startswith(_COUNTER_LINE_START):
      counter_name = line.replace(_COUNTER_LINE_START, '')
      # find counter name
      # find Value: line
      data_points['{}_{}'.format(counter_name, 'Value')] = value
  return data_points


def get_raw_data_points(metrics_reports):
  """

  Args:
    metrics_reports(list of strings): List of strings from calls to
      metrics_report(). NOTE: order will be maintained in the output.

  Returns:
    dict of metric name to list of values, one value from each report in
    metrics_reports. Order of values for each metric in the output dict
    will match the order of metrics reports in the input. Output dict keys
    will look like "CompileTime_TotalSamples" and "CompileTime_Accumulator" for
    metrics and will look like "CreateCompileHandles_Value" for counters.
  """
  raw_data_points = collections.defaultdict(lambda: [None] * len(metrics_reports))
  reports = [_parse_metrics_report(report) for report in metrics_reports)]
  for report_index in range(len(metrics_reports)):
    for metric_name in report:
      raw_data_points[metric_name][report_index] = reports[report_index].get(metric_name, None)
  return raw_data_points
      


def compare_metrics(raw_data_points, metrics_report,
                    config={'base_tolerance': 2}):
  """

  Args:
    raw_data_points(dict): Output of get_raw_data_points().
    metrics_report(string): Output of torch_xla metrics_report(). These will
      be compared with the aggregates of raw_data_points.
    config_dict(dict): Configuration for metrics comparison. Tolerances are
      expressed as the number of standard deviations away from the mean.
      You can supply a per-metric tolerance for any aggregated metric name,
      otherwise base_tolerance will be used.

  Returns:
    Metrics difference report (string). For any metric that differed between
    metrics_report and the aggregates of raw_data_points, this report will have
    1 line reporting the difference.
  """
  pass

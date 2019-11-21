"""Logic to compare XLA metrics and identify significant changes."""

import re

_METRIC_REGEX = r'Metric: (\S+)\s+TotalSamples: (\d+)\s+Accumulator: (\S+)'
_COUNTER_REGEX = r'Counter: (\S+)\s+Value: (\d+)'
#_TIME_FIND_REGEX = r'(?P<days>\d+d)?(?P<hours>\d+h)?(?P<minutes>\d+m)?(?P<seconds>\d+s)?(?P<milliseconds>\d+ms)?(?P<microseconds>[\d.]+us)?'
_TIME_FIND_REGEX = r'((?P<days>\d+)d)?((?P<hours>\d+)h)?((?P<minutes>\d+)m)?((?P<seconds>\d+)s)?((?P<milliseconds>\d+)ms)?((?P<microseconds>[\d.]+)us)?'

def _accumulator_to_number(accumulator_string):
  # Converts the various XLA metrics report Accumulator strings into a number
  # using a single unit. Cases covered:
  #  1. 01d01h01m01s01ms01.5us --> (float) seconds
  #  2. TB/GB/MB/KB --> (float) GB
  # if no letters in string
  try:
    return float(accumulator_string)
  except ValueError as e:
    pass

  print("NOW WE PARSE")
  import pdb; pdb.set_trace()

  # Try to parse the string as a duration.
  m = re.match(_TIME_FIND_REGEX, accumulator_string)
  if sum([1 for v in m.groups() if v is not None]):
    gd = m.groupdict()
    for k, v in gd.items():
      new_v = 0.0 if v is None else float(v)
      gd[k] = new_v
    total_sec = 0.0
    total_sec += gd.get('days') * 24 * 60 * 60
    total_sec += gd.get('hours') * 60 * 60
    total_sec += gd.get('minutes') * 60
    total_sec += gd.get('seconds')
    total_sec += gd.get('milliseconds') * 0.001
    total_sec += gd.get('microseconds') * 1e-6
    print(accumulator_string)
    print("TOTAL SEC:{}".format(total_sec))
    return total_sec

  # Try to parse the string as disk space.
  print("PARSE AS DISK")
  

def _parse_metrics_report(report):
  import pdb; pdb.set_trace()
  data_points = {}

  metrics_matches = re.findall(_METRIC_REGEX, report)
  for match in metrics_matches:
    # Each tuple is of form (name, num samples, accumulator).
    data_points['{}_{}'.format(match[0], 'TotalSamples')] = int(match[1])
    data_points['{}_{}'.format(
        match[0], 'Accumulator')] = _accumulator_to_number(match[2])
  
  counters_matches = re.findall(_COUNTER_REGEX, report)
  for match in counters_matches:
    # Each tuple is of the form (name, counter value).
    data_points['{}_{}'.format(match[0], 'Value')] = int(match[1])

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
  reports = [_parse_metrics_report(report) for report in metrics_reports]
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

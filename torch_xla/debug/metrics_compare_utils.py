"""Logic to compare XLA metrics and identify significant changes."""

import collections
import re

from numpy import mean
from numpy import std

#_METRIC_REGEX = r'Metric: (\S+)\s+TotalSamples: (\d+)\s+Accumulator: (\S+)\s+\S+Percentiles: 1%=(\S+); 5%=(\S+); 10%=(\S+); 20%=(\S+); 50%=(\S+); 80%=(\S+); 90%=(\S+); 95%=(\S+); 99%=(\S+);'
#_METRIC_REGEX = r'Metric: (\S+)\s+TotalSamples: (\d+)\s+Accumulator: (\S+)[^P]+Percentiles: 1%=(\S+); 5%=(\S+); 10%=(\S+); 20%=(\S+); 50%=(\S+); 80%=(\S+); 90%=(\S+); 95%=(\S+); 99%=(\S+)'
_METRIC_REGEX = r'Metric: (?P<metric_name>\S+)\s+TotalSamples: (?P<TotalSamples>\d+)\s+Accumulator: (?P<Accumulator>\S+)[^P]+Percentiles: 1%=(?P<Percentile_1>\S+); 5%=(?P<Percentile_5>\S+); 10%=(?P<Percentile_10>\S+); 20%=(?P<Percentile_20>\S+); 50%=(?P<Percentile_50>\S+); 80%=(?P<Percentile_80>\S+); 90%=(?P<Percentile_90>\S+); 95%=(?P<Percentile_95>\S+); 99%=(?P<Percentile_99>\S+)'
_COUNTER_REGEX = r'Counter: (\S+)\s+Value: (\d+)'
_TIME_FIND_REGEX = r'((?P<days>\d+)d)?((?P<hours>\d+)h)?((?P<minutes>\d+)m(?!s))?((?P<seconds>\d+)s)?((?P<milliseconds>\d+)ms)?((?P<microseconds>[\d.]+)us)?'
_DISK_SIZE_REGEX = r'((?P<petabytes>[\d.]+)PB)?((?P<terabytes>[\d.]+)TB)?((?P<gigabytes>[\d.]+)GB)?((?P<megabytes>[\d.]+)MB)?((?P<kilobytes>[\d.]+)KB)?((?P<bytes>[\d.]+)B)?'


def _regex_matches_groupdict(regex, target_str):
  m = re.match(regex, target_str, re.IGNORECASE)
  if any(m.groups()):  # Regex results will be str or None.
    gd = m.groupdict()
    for k, v in gd.items():
      new_v = 0.0 if v is None else float(v)
      gd[k] = new_v
    return gd
  return None


#def _accumulator_to_number(metric_str):
def _metric_str_to_number(metric_str):
  # Convert XLA metrics report Accumulator strings into a number using
  # a standardized unit. Returns the value as float and unit as string.
  #
  # Cases covered:
  #  1. No units
  #  2. 01d01h01m01s01ms01.5us --> (float) seconds
  #  3. PB/TB/GB/MB/KB/B --> (float) MB

  # Try to parse the string as a number (no units).
  try:
    return float(metric_str), ''
  except ValueError as e:
    pass

  # Try to parse the string as a duration.
  time_gd = _regex_matches_groupdict(_TIME_FIND_REGEX, metric_str)
  if time_gd:
    total_sec = 0.0
    total_sec += time_gd.get('days') * 24 * 60 * 60
    total_sec += time_gd.get('hours') * 60 * 60
    total_sec += time_gd.get('minutes') * 60
    total_sec += time_gd.get('seconds')
    total_sec += time_gd.get('milliseconds') * 1e-3
    total_sec += time_gd.get('microseconds') * 1e-6
    return total_sec, 'sec'

  # Try to parse the string as disk space.
  disk_gd = _regex_matches_groupdict(_DISK_SIZE_REGEX, metric_str)
  if disk_gd:
    total_mb = 0.0
    total_mb += disk_gd.get('petabytes') * 1e9
    total_mb += disk_gd.get('terabytes') * 1e6
    total_mb += disk_gd.get('gigabytes') * 1e3
    total_mb += disk_gd.get('megabytes')
    total_mb += disk_gd.get('kilobytes') * 1e-3
    total_mb += disk_gd.get('bytes') * 1e-6
    return total_mb, 'mb'

  raise ValueError('Unknown metric_str format: {}'.format(
      metric_str))
  

def _parse_metrics_report(report):
  # Convert a string metrics report to a dict of parsed values.
  data_points = {}

  # Parse metrics into data points.
  metric_match_gd = [m.groupdict() for m in re.finditer(_METRIC_REGEX, report)] 
  for gd in metric_match_gd:
    metric_name = gd.pop('metric_name')
    for k, v in gd.items():
      parsed_v, units = _metric_str_to_number(v)
      full_key = '{}__{}{}{}'.format(
        metric_name, k, '_' if units else '', units)
      data_points[full_key] = parsed_v
 
  # Parse counters into data points.
  counters_matches = re.findall(_COUNTER_REGEX, report)
  # Each match tuple is of the form (name, counter value).
  for match in counters_matches:
    data_points['{}__{}'.format(match[0], 'Value')] = int(match[1])

  return data_points


def get_data_points_from_metrics_reports(metrics_reports):
  """Collect numeric datapoints from a list of metrics reports.

  Args:
    metrics_reports(list of strings): List of strings from calls to
      metrics_report(). NOTE: order will be maintained in the output.

  Returns:
    dict of metric name to list of values, one value from each report in
    metrics_reports. Order of values for each metric in the output dict
    will match the order of metrics reports in the input.
    
    For example, if the args were [report1, report2, report3], output might be:
    {
      'CompileTime__Accumulator_sec': [50, None, 80],  # missing from report2
      'InboundData__TotalSamples': [11, 12, 11],
      'CreateCompileHandles__Value': [20, 20, 19],
      ...
    }
  """
  data_points = collections.defaultdict(lambda: [None] * len(metrics_reports))
  for report_index in range(len(metrics_reports)):
    parsed_report = _parse_metrics_report(metrics_reports[report_index])
    for metric_name in parsed_report:
      data_points[metric_name][report_index] = parsed_report.get(
          metric_name, None)
  return data_points


def _compute_aggregates(data_points):
  # Compute mean and standard deviation for each key in data_points.
  aggregates = {}
  for k, v_list in data_points.items():
    v_list = [v for v in v_list if v is not None]
    if v_list:
      aggregates[k] = {'mean': mean(v_list), 'stddev': std(v_list)}
  return aggregates


def compare_metrics(data_points, metrics_report,
                    config={'base_tolerance': 2.0}):
  """Compare metrics_report to historical averages and report differences.

  Args:
    data_points(dict): Output of get_data_points_from_metrics_reports().
    metrics_report(string): Output of torch_xla metrics_report(). These will
      be compared with the aggregates of raw_data_points.
    config(dict): Configuration for metrics comparison. Tolerances are
      expressed as the number of standard deviations away from the mean.
      You can supply a per-metric tolerance for any aggregated metric name,
      otherwise base_tolerance will be used. Example config:
      {
        'base_tolerance': 2.0,
        'TransferFromServerTime_Accumulator_sec_tolerance': 10.0,
        ...
      }

  Returns:
    Metrics difference report (string). For any metric that differed between
    metrics_report and the aggregates of raw_data_points, this report will have
    1 line reporting the difference.
  """
  parsed_report = _parse_metrics_report(metrics_report)
  means_and_stddevs = _compute_aggregates(data_points)

  difference_report = ''
  for k, v in sorted(parsed_report.items()):
    if k not in means_and_stddevs:
      # Alert if we have new ops falling back to CPU.
      if k.startswith('aten::'):
        difference_report += 'Found new aten counter: {}: {}\n'.format(k, v)
    else:
      v_mean = means_and_stddevs[k]['mean']
      v_stddev = means_and_stddevs[k]['stddev']

      # base_tolerance can be overridden for individual metrics.
      mul = config.get('{}_tolerance'.format(k), None)
      if mul is None:
        mul = config['base_tolerance']

      lower_limit = v_mean - (v_stddev * mul)
      upper_limit = v_mean + (v_stddev * mul)
      if v < lower_limit or v > upper_limit:
        difference_report += ('{} is outside the expected range using '
            'tolerance: {}. Lower limit: {}  Upper limit: {}  '
            'Actual Value: {}\n'.format(
                k, mul, lower_limit, upper_limit, v))
  return difference_report

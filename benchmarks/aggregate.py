"""Processes .csv result files and aggregates them."""

import argparse
import csv
from datetime import datetime
import logging
import os
import re
import sys
import tiers
import itertools
from typing import Any, Dict, List
import numpy as np
from scipy.stats.mstats import gmean

try:
  import matplotlib.pyplot as plt
  import matplotlib.dates as mdates
  has_matplotlib = True
except ImportError:
  has_matplotlib = False

logger = logging.getLogger(__name__)

test_to_csv_field_name = {
    'inference': {
        'test': 'eval',
        'xla_label': 'openxla_eval',
        'inductor_label': 'inductor',
    },
    'training': {
        'test': 'train',
        'xla_label': 'openxla',
        'inductor_label': 'inductor',
    },
}


def find_files(input_dirname: str) -> List[str]:
  files = []
  for root, _, filenames in os.walk(input_dirname):
    for filename in filenames:
      match = re.search(r'.*\.csv$', filename)
      if match:
        files.append(os.path.join(root, filename))
  return files


def clean_up_accelerator_model(model: str) -> str:
  if re.search(r'One of Tesla V100', model):
    return 'v100'
  if re.search(r'One of Quadro P1000, NVIDIA RTX A6000', model):
    return 'a6000'
  if re.search(r'NVIDIA A100-SXM4-40GB', model):
    return 'a100'
  sys.exit(f"fatal: cannot recognize accelerator model: '{model}'.")


def skip_model(args, model_name: str):
  return (not re.search("|".join(args.filter), model_name, re.I) or
          re.search("|".join(args.exclude), model_name, re.I))


def process_file(args, results_map: Dict[str, Any], filename: str):
  with open(filename) as check_header_file:
    try:
      has_header = csv.Sniffer().has_header(check_header_file.read(1024))
    except csv.Error:
      logger.error('Cannot read CSV in %s, skipping.', filename)
      return
    if not has_header:
      logger.error('Cannot interpret %s: missing headers.', filename)
      return
  fields = (
      'model_name',
      'accelerator_model',
      'dynamo',
      'test',
      'batch_size',
      'median_total_time',
  )
  with open(filename) as read_file:
    reader = csv.reader(read_file)
    headers = next(reader)
    if headers[0] != 'timestamp':
      logger.error('Missing timestamp in CSV in %s, skipping.', filename)
      return
    field2index = {}
    for i, header in enumerate(headers):
      for field in fields:
        if field == header:
          field2index[field] = i
    for row in reader:
      timestamp = row[0]
      model_name = row[field2index['model_name']]
      if skip_model(args, model_name):
        continue
      accelerator_model = clean_up_accelerator_model(
          row[field2index['accelerator_model']])
      if accelerator_model != args.accelerator:
        continue
      dynamo = row[field2index['dynamo']]
      test = row[field2index['test']]
      if test != test_to_csv_field_name[args.test]['test']:
        continue
      batch_size = row[field2index['batch_size']]
      median_total_time = row[field2index['median_total_time']]
      if timestamp not in results_map:
        results_map[timestamp] = {}
      if dynamo not in results_map[timestamp]:
        results_map[timestamp][dynamo] = {}
      if (model_name not in results_map[timestamp][dynamo]):
        results_map[timestamp][dynamo][model_name] = {}
      if (batch_size not in results_map[timestamp][dynamo][model_name]):
        results_map[timestamp][dynamo][model_name][batch_size] = {}
      results_map[timestamp][dynamo][model_name][batch_size] = median_total_time


def summarize_speedups(acc_map: Dict[str, Any], label: str):
  if label not in acc_map:
    return
  acc_map[f'{label}:gmean'] = gmean(acc_map[label])
  for p in (5, 50, 95):
    percentile = float(np.percentile(acc_map[label], p))
    acc_map[f'{label}:p{p}'] = percentile


# The speedup values are stored in acc_map[out_label]; the corresponding
# model names are stored in acc_map[f'{out_label}:model_name'].
def compute_speedups(acc_map: Dict[str, Any], baseline: Dict[str, Any],
                     out_label: str, in_label: str):
  model_label = f'{out_label}:model_name'
  if in_label not in acc_map:
    return
  for model_name, v in acc_map[in_label].items():
    if model_name not in baseline:
      continue
    speedups = []
    # If we are running several batch sizes, keep the geomean of their speedups.
    for batch_size in v:
      experiment_time = v[batch_size]
      baseline_time = baseline[model_name].get(batch_size, None)
      if not experiment_time or not baseline_time:
        continue
      speedups.append(float(baseline_time) / float(experiment_time))
    if speedups:
      if out_label not in acc_map:
        acc_map[out_label] = []
      acc_map[out_label].append(gmean(speedups))
      if model_label not in acc_map:
        acc_map[model_label] = []
      acc_map[model_label].append(model_name)
  summarize_speedups(acc_map, out_label)


# A benchmark's baseline is the oldest Inductor perf number we have for it.
# This way we can track both Pytorch/XLA and Inductor perf improvements over
# time.
def compute_baseline(results_map: Dict[str, Any]) -> Dict[str, Any]:
  baseline = {}
  for ts in sorted(list(results_map.keys())):
    if 'inductor' not in results_map[ts]:
      continue
    for model_name in results_map[ts]['inductor']:
      if model_name not in baseline:
        baseline[model_name] = {}
      for batch_size in results_map[ts]['inductor'][model_name]:
        if batch_size not in baseline[model_name]:
          baseline[model_name][batch_size] = results_map[ts]['inductor'][
              model_name][batch_size]
  return baseline


def process_results(args, results_map: Dict[str, Any]):
  baseline = compute_baseline(results_map)
  for timestamp in results_map:
    acc_map = results_map[timestamp]

    name2field = test_to_csv_field_name[args.test]
    compute_speedups(acc_map, baseline, 'xla:speedups', name2field['xla_label'])
    compute_speedups(acc_map, baseline, 'inductor:speedups',
                     name2field['inductor_label'])


def maketitle(args, title: str):
  if args.title:
    title += f'\n{args.title}'
  return title


def pr_latest(results_map: Dict[str, Any], args, timestamps: List[str]):
  prefixes = ('inductor', 'xla')
  speedups = [[], []]
  model_names = [[], []]
  speedup_timestamps = [[], []]

  for i, pfx in enumerate(prefixes):
    label = f'{pfx}:speedups'
    model_label = f'{label}:model_name'
    for timestamp in reversed(timestamps):
      acc_map = results_map[timestamp]
      if label in acc_map:
        (speedups[i], model_names[i]) = map(
            list, zip(*sorted(zip(acc_map[label], acc_map[model_label]))))
        speedup_timestamps[i] = timestamp
        break
  if not speedups[0] or not speedups[1]:
    logger.warning(f'cannot find data for accelerator {args.accelerator}')
    return

  if args.format == 'csv':
    print('# WorkloadNumber,Speedup(Inductor/Oldest Inductor),'
          'ModelName(Inductor),Speedup(PytorchXLA/Oldest Inductor),'
          'ModelName(PytorchXLA)')
    # Use zip_longest because the latest timestamp might not have complete
    # results for all benchmarks.
    for i, z in enumerate(itertools.zip_longest(speedups[0], speedups[1])):
      print(','.join(
          map(str, [
              i, z[0], model_names[0][i] if z[0] else None, z[1],
              model_names[1][i] if z[1] else None
          ])))
  else:
    plt.axhline(y=1.0, color='lightgray')
    plt.plot(speedups[0], label='Inductor', marker='^')
    plt.plot(speedups[1], label='PytorchXLA', marker='o')
    plt.legend()
    datestr = datetime.utcfromtimestamp(float(speedup_timestamps[0]))
    if speedup_timestamps[0] != speedup_timestamps[1]:
      datestr = f'{datestr} (Inductor)'
      datestr += ', {datetime.utcfromtimestamp(float(speedup_timestamps[1]))} (PytorchXLA)'
    plt.title(
        maketitle(args,
                  f'Speedup over Oldest Benchmarked Inductor as of {datestr}'))
    plt.xlabel('Workload Number')
    plt.ylabel(f'Speedup')
    plt.savefig(sys.stdout.buffer)


def pr_histogram(results_map: Dict[str, Any], args, timestamps: List[str]):
  percentiles = [f'p{p}' for p in (95, 50, 5)]
  prefixes = ('inductor', 'xla')
  labels = [f'{pfx}:speedups:{p}' for pfx in prefixes for p in percentiles]
  titles = [
      l.replace(':speedups:',
                ' ').replace('xla',
                             'PytorchXLA').replace('inductor', 'Inductor')
      for l in labels
  ]
  x = []
  y = [[] for i in range(len(labels))]
  for timestamp in timestamps:
    if labels[0] in results_map[timestamp]:
      for label in labels:
        assert label in results_map[timestamp]
      x.append(datetime.utcfromtimestamp(float(timestamp)))
      for i, label in enumerate(labels):
        y[i].append(results_map[timestamp][label])
  if args.format == 'csv':
    titles = ['# Datetime(UTC)'] + titles
    print(','.join(titles))
    for i, utc in enumerate(x):
      print(','.join([str(utc)] + [str(y[j][i]) for j in range(len(labels))]))
  else:
    fig, ax = plt.subplots()
    ax.axhline(y=1.0, color='lightgray')
    markers = ('^', 'o')
    linestyles = ('solid', 'dotted')
    for i, label in enumerate(labels):
      style = int(i / len(percentiles))
      ax.plot(
          x,
          y[i],
          label=titles[i],
          marker=markers[style],
          linestyle=linestyles[style])
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Geomean Speedup")
    plt.title(
        maketitle(args,
                  'Histogram of Speedup over Oldest Benchmarked Inductor'))
    plt.savefig(sys.stdout.buffer)


def pr_gmean(results_map: Dict[str, Any], args, timestamps: List[str]):
  label = f'speedups:gmean'
  x = []
  y0 = []
  y1 = []
  for timestamp in timestamps:
    if 'inductor:speedups:gmean' not in results_map[
        timestamp] or 'xla:speedups:gmean' not in results_map[timestamp]:
      continue
    x.append(datetime.utcfromtimestamp(float(timestamp)))
    y0.append(results_map[timestamp]['inductor:speedups:gmean'])
    y1.append(results_map[timestamp]['xla:speedups:gmean'])
  if args.format == 'csv':
    print(
        '# Datetime(UTC),Speedup(Inductor/Oldest Inductor),Speedup(PytorchXLA/Oldest Inductor)'
    )
    for a, b, c in zip(x, y0, y1):
      print(','.join(map(str, [a, b, c])))
  else:
    fig, ax = plt.subplots()
    ax.axhline(y=1.0, color='lightgray')
    ax.plot(x, y0, marker='^', label='Inductor')
    ax.plot(x, y1, marker='o', label='PytorchXLA')
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Geomean Speedup")
    plt.title(maketitle(args, 'Speedup over Oldest Benchmarked Inductor'))
    plt.savefig(sys.stdout.buffer)


def pr_results(results_map: Dict[str, Any], args):
  timestamps = list(results_map.keys())
  timestamps.sort()

  if args.format == 'png' and not has_matplotlib:
    sys.exit('Fatal: cannot find matplotlib packages needed for PNG output.')

  if args.report == 'latest':
    return pr_latest(results_map, args, timestamps)
  elif args.report == 'histogram':
    return pr_histogram(results_map, args, timestamps)
  elif args.report == 'speedup':
    return pr_gmean(results_map, args, timestamps)
  else:
    sys.exit('unreachable')


def parse_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--accelerator',
      default='v100',
      choices=['a100', 'v100', 'a6000'],
      help='Accelerator.')
  parser.add_argument(
      "--exclude",
      "-x",
      action="append",
      default=[],
      help="filter out benchmarks with regexp")
  parser.add_argument(
      "--exclude-by-tier",
      type=int,
      action="append",
      default=[],
      help="filter out benchmarks by predefined tier 1-3",
  )
  parser.add_argument(
      "--filter",
      "-k",
      action="append",
      default=[],
      help="filter benchmarks with regexp")
  parser.add_argument(
      "--filter-by-tier",
      type=int,
      action="append",
      default=[],
      help="filter benchmarks by predefined tier 1-3",
  )
  parser.add_argument(
      "--format", default='csv', choices=['csv', 'png'], help='Output format')
  parser.add_argument(
      '--input-dirname', '-i', required=True, type=str, help='Input directory.')
  parser.add_argument(
      '--report',
      default='speedup',
      choices=['latest', 'histogram', 'speedup'],
      help='What report to generate.')
  parser.add_argument(
      '--test',
      default='inference',
      choices=['inference', 'training'],
      help='Test mode.')
  parser.add_argument('--title', type=str, help="Plot title.")
  args = parser.parse_args(args)

  tiers.append_filter_by_tier(args.filter, args.filter_by_tier)
  tiers.append_filter_by_tier(args.exclude, args.exclude_by_tier)
  args.filter = args.filter or [r"."]
  args.exclude = args.exclude or [r"^$"]

  return args


def main():
  args = parse_args()
  filenames = find_files(args.input_dirname)
  results_map = {}

  # Some CSV files have lots of errors from execution; expand CSV's size limit.
  csv.field_size_limit(1024 * 1024)

  for filename in filenames:
    process_file(args, results_map, filename)
  process_results(args, results_map)
  if not results_map:
    sys.exit('no results found')
  pr_results(results_map, args)


if __name__ == '__main__':
  main()

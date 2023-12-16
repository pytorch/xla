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

_test_to_csv_field_name = {
    'inference': 'eval',
    'training': 'train',
}

_markers = ('^', 'o', 's')


# Round floats before printing them so that tiny differences don't break tests.
def pr_round(x):
  return round(x, 8)


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
      if test != _test_to_csv_field_name[args.test]:
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

    compute_speedups(acc_map, baseline, 'xla:speedups', 'openxla')
    compute_speedups(acc_map, baseline, 'xla_eval:speedups', 'openxla_eval')
    compute_speedups(acc_map, baseline, 'inductor:speedups', 'inductor')


def maketitle(args, title: str):
  if args.title:
    title += f'\n{args.title}'
  return title


def get_pr_titles(args):
  titles = ['Inductor', 'PytorchXLA']
  data_labels = ['inductor', 'xla']
  if args.test == "inference":
    titles.append('PytorchXLA_Eval')
    data_labels.append('xla_eval')
  return [titles, data_labels]


def pr_latest(results_map: Dict[str, Any], args, timestamps: List[str]):
  titles, data_labels = get_pr_titles(args)
  speedups = [[] for _ in titles]
  model_names = [[] for _ in titles]

  for i, pfx in enumerate(data_labels):
    label = f'{pfx}:speedups'
    model_label = f'{label}:model_name'
    for timestamp in reversed(timestamps):
      acc_map = results_map[timestamp]
      if label in acc_map:
        speedups[i], model_names[i] = map(
            list, zip(*sorted(zip(acc_map[label], acc_map[model_label]))))
        speedups[i] = list(map(pr_round, speedups[i]))
        break
  if not speedups[0] or not speedups[1]:
    logger.warning(f'cannot find data for accelerator {args.accelerator}')
    return

  if args.format == 'csv':
    print(','.join(['# WorkloadNumber'] + [
        f'Speedup({title}/Oldest Inductor),ModelName({title})'
        for title in titles
    ]))
    # Note: the latest timestamp might not have results for all benchmarks.
    max_len = max([len(l) for l in speedups])

    def pad_array(arr, desired_len):
      if len(arr) >= desired_len:
        return
      arr += [''] * (desired_len - len(arr))

    for i in range(len(titles)):
      pad_array(speedups[i], max_len)
      pad_array(model_names[i], max_len)

    for j in range(max_len):
      print(','.join(
          map(str, [j] + [
              v for i in range(len(titles))
              for v in (speedups[i][j], model_names[i][j])
          ])))
  else:
    plt.axhline(y=1.0, color='lightgray')
    for i in range(len(titles)):
      plt.plot(speedups[i], label=titles[i], marker=_markers[i])
    plt.legend()
    plt.title(maketitle(args, f'Speedup over Oldest Benchmarked Inductor'))
    plt.xlabel('Workload Number')
    plt.ylabel(f'Speedup')
    plt.savefig(sys.stdout.buffer)


def pr_histogram(results_map: Dict[str, Any], args, timestamps: List[str]):
  titles, data_labels = get_pr_titles(args)
  percentiles = [f'p{p}' for p in (95, 50, 5)]
  labels = [f'{pfx}:speedups:{p}' for pfx in data_labels for p in percentiles]
  full_titles = [f'{title} {p}' for title in titles for p in percentiles]
  x = []
  y = [[] for i in range(len(labels))]
  for timestamp in timestamps:
    if labels[0] in results_map[timestamp]:
      for label in labels:
        assert label in results_map[timestamp]
      x.append(datetime.utcfromtimestamp(float(timestamp)))
      for i, label in enumerate(labels):
        y[i].append(
            pr_round(results_map[timestamp][label] if label in
                     results_map[timestamp] else ''))
  if args.format == 'csv':
    full_titles = ['# Datetime(UTC)'] + full_titles
    print(','.join(full_titles))
    for j, utc in enumerate(x):
      print(','.join([str(utc)] + [str(y[i][j]) for i in range(len(labels))]))
  else:
    fig, ax = plt.subplots()
    ax.axhline(y=1.0, color='lightgray')
    linestyles = ('solid', 'dotted', 'dashed')
    for i, label in enumerate(labels):
      style = int(i / len(percentiles))
      ax.plot(
          x,
          y[i],
          label=full_titles[i],
          marker=_markers[style],
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
  titles, data_labels = get_pr_titles(args)
  labels = [f"{x}:speedups:gmean" for x in data_labels]
  y = [[] for _ in labels]
  for timestamp in timestamps:
    if all(label not in results_map[timestamp] for label in labels):
      continue
    x.append(datetime.utcfromtimestamp(float(timestamp)))
    for i, label in enumerate(labels):
      y[i].append(
          pr_round(results_map[timestamp][label]) if label in
          results_map[timestamp] else '')
  if args.format == 'csv':
    print(','.join(['# Datetime(UTC)'] +
                   [f"Speedup({title}/Oldest Inductor)" for title in titles]))
    for j, x in enumerate(x):
      print(','.join(map(str, [x] + [y[i][j] for i in range(len(labels))])))
  else:
    fig, ax = plt.subplots()
    ax.axhline(y=1.0, color='lightgray')
    for i in range(len(labels)):
      ax.plot(x, y[i], marker=_markers[i], label=titles[i])
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
  parser.add_argument('input_file', nargs='+')
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
  filenames = args.input_file
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

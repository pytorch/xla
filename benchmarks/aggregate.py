"""Processes .csv result files and aggregates them."""
# TODO: support more plots:
# - Speedup of Inductor and PytorchXLA over the oldest Inductor data set.
#   This will allow us to have a sense of how fast Inductor is improving
#   as well as PytorchXLA.
# - Number of working Inductor and PytorchXLA workloads.

import argparse
import csv
from datetime import date
import logging
import os
import re
import sys
import matplotlib.pyplot as plt
from typing import Any
import numpy as np
from scipy.stats.mstats import gmean

try:
  from .tiers import append_filter_by_tier
except ImportError:
  from tiers import append_filter_by_tier

logger = logging.getLogger(__name__)


def find_files(input_dirname: str) -> list[str]:
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


def process_file(args, results_map: dict[str, Any], filename: str):
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
      dynamo = row[field2index['dynamo']]
      test = row[field2index['test']]
      batch_size = row[field2index['batch_size']]
      median_total_time = row[field2index['median_total_time']]
      if timestamp not in results_map:
        results_map[timestamp] = {}
      if accelerator_model not in results_map[timestamp]:
        results_map[timestamp][accelerator_model] = {}
      if dynamo not in results_map[timestamp][accelerator_model]:
        results_map[timestamp][accelerator_model][dynamo] = {}
      if test not in results_map[timestamp][accelerator_model][dynamo]:
        results_map[timestamp][accelerator_model][dynamo][test] = {}
      if (model_name
          not in results_map[timestamp][accelerator_model][dynamo][test]):
        results_map[timestamp][accelerator_model][dynamo][test][model_name] = {}
      if (batch_size not in results_map[timestamp][accelerator_model][dynamo]
          [test][model_name]):
        results_map[timestamp][accelerator_model][dynamo][test][model_name][
            batch_size] = {}
      results_map[timestamp][accelerator_model][dynamo][test][model_name][
          batch_size] = median_total_time


def summarize_speedups(acc_map: dict[str, Any], label: str):
  if label not in acc_map:
    return
  acc_map[f'{label}:gmean'] = gmean(acc_map[label])
  for p in (5, 50, 95):
    percentile = float(np.percentile(acc_map[label], p))
    acc_map[f'{label}:p{p}'] = percentile


# The speedup values are stored in acc_map[label]; the corresponding
# model names are stored in acc_map[f'{label}:model_name'].
def compute_speedups(acc_map: dict[str, Any], label: str, xla_label,
                     inductor_label, test_label):
  model_label = f'{label}:model_name'
  if xla_label not in acc_map:
    return
  if inductor_label not in acc_map:
    return
  if (test_label not in acc_map[xla_label] or
      test_label not in acc_map[inductor_label]):
    return
  for model_name, v in acc_map[xla_label][test_label].items():
    if model_name not in acc_map[inductor_label][test_label]:
      continue
    speedups = []
    # If we are running several batch sizes, keep the geomean of their speedups.
    for batch_size in v:
      xla_time = v[batch_size]
      inductor_time = acc_map[inductor_label][test_label][model_name].get(
          batch_size, None)
      if not xla_time or not inductor_time:
        continue
      speedups.append(float(inductor_time) / float(xla_time))
    if speedups:
      if label not in acc_map:
        acc_map[label] = []
      acc_map[label].append(gmean(speedups))
      if model_label not in acc_map:
        acc_map[model_label] = []
      acc_map[model_label].append(model_name)
  summarize_speedups(acc_map, label)


def process_results(results_map: dict[str, Any]):
  for timestamp in results_map:
    for accelerator in results_map[timestamp]:
      acc_map = results_map[timestamp][accelerator]

      compute_speedups(acc_map, 'speedups:inference', 'openxla_eval',
                       'inductor', 'eval')
      compute_speedups(acc_map, 'speedups:training', 'openxla', 'inductor',
                       'train')


def maketitle(args, title: str):
  if args.title:
    title += f'\n{args.title}'
  return title


def pr_latest(results_map: dict[str, Any], args, timestamps: list[str]):
  label = f'speedups:{args.test}'
  model_label = f'{label}:model_name'

  for timestamp in reversed(timestamps):
    if label not in results_map[timestamp][args.accelerator]:
      continue
    acc_map = results_map[timestamp][args.accelerator]
    (speedups,
     model_names) = map(list,
                        zip(*sorted(zip(acc_map[label], acc_map[model_label]))))

    if args.format == 'csv':
      print('# WorkloadNumber,Speedup,ModelName')
      for i, speedup in enumerate(speedups):
        print(','.join(map(str, [i, speedup, model_names[i]])))
    else:
      plt.axhline(y=1.0, color='lightgray')
      plt.plot(speedups, marker='o')
      plt.title(
          maketitle(
              args,
              f'Speedup of Pytorch/XLA over Inductor\n{date.fromtimestamp(float(timestamp))}'
          ))
      plt.xlabel('Workload Number')
      plt.ylabel(f'Speedup')
      plt.savefig(sys.stdout.buffer)
    return
  logger.warning(f'cannot find data for accelerator {args.accelerator}')


def pr_histogram(results_map: dict[str, Any], args, timestamps: list[str]):
  percentiles = [f'p{p}' for p in (5, 50, 95)]
  labels = [f'speedups:{args.test}:{p}' for p in percentiles]
  x = []
  y = [[] for i in range(len(percentiles))]
  for timestamp in timestamps:
    if labels[0] in results_map[timestamp][args.accelerator]:
      for label in labels:
        assert label in results_map[timestamp][args.accelerator]
      x.append(date.fromtimestamp(float(timestamp)))
      for i, label in enumerate(labels):
        y[i].append(results_map[timestamp][args.accelerator][label])
  if args.format == 'csv':
    titles = ['# Datetime'] + percentiles
    print(','.join(titles))
    for i, datetime in enumerate(x):
      print(','.join([str(datetime)] +
                     [str(y[j][i]) for j in range(len(percentiles))]))
  else:
    plt.axhline(y=1.0, color='lightgray')
    for i, p in enumerate(percentiles):
      plt.plot(x, y[i], label=p, marker='^')
      plt.legend()
      plt.xlabel("Date")
      plt.ylabel("Geomean Speedup")
      plt.title(
          maketitle(args, f"Histogram of Pytorch/XLA's Speedup over Inductor"))
      plt.savefig(sys.stdout.buffer)


def pr_gmean(results_map: dict[str, Any], args, timestamps: list[str]):
  label = f'speedups:{args.test}:gmean'
  x = []
  y = []
  for timestamp in timestamps:
    if label not in results_map[timestamp][args.accelerator]:
      continue
    x.append(date.fromtimestamp(float(timestamp)))
    gmean = results_map[timestamp][args.accelerator][label]
    y.append(gmean)
  if args.format == 'csv':
    print('# Datetime,Speedup')
    for a, b in zip(x, y):
      print(','.join(map(str, [a, b])))
  else:
    plt.axhline(y=1.0, color='lightgray')
    plt.plot(x, y, marker='^')
    plt.xlabel("Date")
    plt.ylabel("Geomean Speedup")
    plt.title(maketitle(args, f"Pytorch/XLA's Speedup over Inductor"))
    plt.savefig(sys.stdout.buffer)


def pr_results(results_map: dict[str, Any], args):
  timestamp_list = list(results_map.keys())
  timestamps = [
      ts for ts in timestamp_list if args.accelerator in results_map[ts]
  ]
  timestamps.sort()

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

  append_filter_by_tier(args.filter, args.filter_by_tier)
  append_filter_by_tier(args.exclude, args.exclude_by_tier)
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
  process_results(results_map)
  if not results_map:
    sys.exit('no results found')
  pr_results(results_map, args)


if __name__ == '__main__':
  main()

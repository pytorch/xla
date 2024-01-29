"""Processes .jsonl result files and aggregates them."""

import argparse
from collections import namedtuple
from datetime import datetime
import json
import logging
import math
import os
import re
import sys
import tiers
import itertools
from typing import Any, Dict, List, NamedTuple
import numpy as np
from scipy.stats.mstats import gmean

try:
  import matplotlib.pyplot as plt
  import matplotlib.dates as mdates
  has_matplotlib = True
except ImportError:
  has_matplotlib = False

logger = logging.getLogger(__name__)

Datapoint = namedtuple('Datapoint', 'avg, std')

_test_to_field_name = {
    'inference': 'eval',
    'training': 'train',
}
_fig_elinewidth = 0.5
_fig_capsize = 3

_markers = ('D', 'o', 's')


class DatapointSelector:

  def compile(row):
    total_times = row['metrics']['total_time'] if 'total_time' in row[
        'metrics'] else []
    if len(total_times) <= 1:
      return None
    compile_and_run_time = total_times[0]
    run_time = np.average(total_times[1:])
    compile_time = compile_and_run_time - run_time
    # Single sample size for compilation time, std = 0.
    return Datapoint(compile_time, std=0)

  def exec(row):
    total_times = row['metrics']['total_time'] if 'total_time' in row[
        'metrics'] else []
    # Skip the first three elements in total_times[]; the first one
    # includes compilation time, the 2nd and 3rd warm up the caches.
    skip_elems = 3
    if len(total_times) <= skip_elems:
      return None
    avg = np.average(total_times[skip_elems:])
    # Standard deviation of the sample, i.e. N-1 denominator.
    # Note: avoid NaN when we compute the std with just one sample.
    std = np.std(
        total_times[skip_elems:],
        ddof=1) if len(total_times) > skip_elems + 1 else 0.0
    return Datapoint(avg, std)


# Round floats before printing them so that tiny differences don't break tests.
def pr_round(x: NamedTuple):
  return Datapoint(round(x.avg, 8), round(x.std, 8))


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
  fields = {
      'experiment': ['accelerator_model', 'batch_size', 'dynamo', 'test'],
      'metrics': [],
      'model': ['model_name'],
      'timestamp': [],
  }
  with open(filename) as read_file:
    for line in read_file:
      try:
        r = json.loads(line.rstrip('\n|\r'))
      except json.JSONDecodeError as e:
        sys.exit(f'Invalid JSONL:\n{line}{e}')

      # check that all fields exist.
      for k in fields:
        if k not in r:
          sys.exit(f'JSONL record does not contain key {k}. JSONL: {r}')
        for kk in fields[k]:
          if kk not in r[k]:
            sys.exit(f'JSONL record does not contain key {k}.{kk}. JSONL: {r}')

      # Read in what we need.
      accelerator_model = clean_up_accelerator_model(
          r['experiment']['accelerator_model'])
      if accelerator_model != args.accelerator:
        continue
      model_name = r['model']['model_name']
      if skip_model(args, model_name):
        continue
      dynamo = r['experiment']['dynamo']
      test = r['experiment']['test']
      if test != _test_to_field_name[args.test]:
        continue

      dp = getattr(DatapointSelector, args.metric)(r)
      if dp is None:
        continue
      batch_size = r['experiment']['batch_size']
      timestamp = r['timestamp']

      if timestamp not in results_map:
        results_map[timestamp] = {}
      if dynamo not in results_map[timestamp]:
        results_map[timestamp][dynamo] = {}
      if (model_name not in results_map[timestamp][dynamo]):
        results_map[timestamp][dynamo][model_name] = {}
      if (batch_size not in results_map[timestamp][dynamo][model_name]):
        results_map[timestamp][dynamo][model_name][batch_size] = {}
      results_map[timestamp][dynamo][model_name][batch_size] = dp


# Speedup of a over baseline ("b"), with errors.
def compute_speedup(a: NamedTuple, b: NamedTuple) -> NamedTuple:
  rel_err_a = a.avg * a.std
  rel_err_b = b.avg * b.std
  rel_err = math.sqrt(rel_err_a**2 + rel_err_b**2)
  speedup = b.avg / a.avg
  err = rel_err * speedup
  return Datapoint(speedup, err)


# https://math.stackexchange.com/a/123297
def compute_geomean(a: List[NamedTuple]) -> NamedTuple:
  values = [v.avg for v in a]
  g = gmean(values)
  err = g / len(a) * math.sqrt(sum([(v.std / v.avg)**2 for v in a]))
  return Datapoint(g, err)


def summarize_speedups(acc_map: Dict[str, Any], label: str):
  if label not in acc_map:
    return
  acc_map[f'{label}:gmean'] = compute_geomean(acc_map[label])
  for p in (5, 50, 95):
    percentile = float(np.percentile([v.avg for v in acc_map[label]], p))
    # TODO: what stddev to pick here? Set it to 0.0 for now.
    acc_map[f'{label}:p{p}'] = Datapoint(percentile, 0.0)


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
      experiment_times = v[batch_size]
      baseline_times = baseline[model_name].get(batch_size, None)
      if not experiment_times or not baseline_times:
        continue
      speedups.append(compute_speedup(experiment_times, baseline_times))
    if speedups:
      if out_label not in acc_map:
        acc_map[out_label] = []
      acc_map[out_label].append(compute_geomean(speedups))
      if model_label not in acc_map:
        acc_map[model_label] = []
      acc_map[model_label].append(model_name)
  summarize_speedups(acc_map, out_label)


def populate_baseline(baseline: Dict[str, Any], inductor_results: Dict[str,
                                                                       Any]):
  for model_name in inductor_results:
    if model_name not in baseline:
      baseline[model_name] = {}
    for batch_size in inductor_results[model_name]:
      if batch_size not in baseline[model_name]:
        baseline[model_name][batch_size] = inductor_results[model_name][
            batch_size]


def compute_baseline(args, results_map: Dict[str, Any]) -> Dict[str, Any]:
  baseline = {}
  timestamps = list(results_map.keys())
  if not timestamps:
    return baseline
  timestamps.sort()
  if args.baseline == 'oldest':
    # A benchmark's baseline is the oldest Inductor perf number we have for it.
    # This way we can track both Pytorch/XLA and Inductor perf improvements over
    # time.
    for ts in timestamps:
      if 'inductor' not in results_map[ts]:
        continue
      populate_baseline(baseline, results_map[ts]['inductor'])

  elif args.baseline == 'latest':
    # Pick only results from the latest timestamp.
    ts = timestamps[-1]
    if 'inductor' not in results_map[ts]:
      sys.exit(f'No Inductor results in the latest timestamp {ts}')
    populate_baseline(baseline, results_map[ts]['inductor'])
  return baseline


def process_results(args, results_map: Dict[str, Any]):
  baseline = compute_baseline(args, results_map)
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
        f'Speedup({title}/{args.baseline.capitalize()} Inductor),StdDev,ModelName({title})'
        for title in titles
    ]))
    # Note: the latest timestamp might not have results for all benchmarks.
    max_len = max([len(l) for l in speedups])

    def pad_array(arr, desired_len, val):
      if len(arr) >= desired_len:
        return
      arr += [val] * (desired_len - len(arr))

    for i in range(len(titles)):
      pad_array(speedups[i], max_len, Datapoint('', ''))
      pad_array(model_names[i], max_len, '')

    for j in range(max_len):
      print(','.join(
          map(str, [j] + [
              v for i in range(len(titles))
              for v in (speedups[i][j].avg, speedups[i][j].std,
                        model_names[i][j])
          ])))
  else:
    plt.figure(figsize=(args.fig_width, args.fig_height))
    plt.axhline(y=1.0, color='lightgray')
    for i in range(len(titles)):
      plt.errorbar([j for j in range(len(speedups[i]))],
                   [v.avg for v in speedups[i]], [v.std for v in speedups[i]],
                   label=titles[i],
                   marker=_markers[i],
                   elinewidth=_fig_elinewidth,
                   capsize=_fig_capsize)
      # Annotate the plot with the model names.
      for j in range(len(speedups[i])):
        # Try to declutter the plot to avoid overlapping text when two lines
        # are very close. To achieve this we alternate the alignment of
        # the rotated annotations, either "bottom left" or "top right".
        # For details on matplotlib's alignment, see
        #   https://matplotlib.org/stable/gallery/text_labels_and_annotations/text_rotation.html
        valignment = ('bottom', 'top')
        halignment = ('left', 'right')
        annotation = plt.annotate(
            model_names[i][j], (j, speedups[i][j].avg),
            rotation=45,
            size=5.0,
            verticalalignment=valignment[i % len(valignment)],
            horizontalalignment=halignment[i % len(halignment)])
        # Make overlapping text more legible by making it transparent.
        annotation.set_alpha(0.5)
    plt.legend()
    plt.title(
        maketitle(
            args,
            f'Speedup over {args.baseline.capitalize()} Benchmarked Inductor'))
    plt.xlabel('Workload Number')
    plt.ylabel(f'Speedup')
    plt.savefig(sys.stdout.buffer, format=args.format)


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
                     results_map[timestamp] else Datapoint('', '')).avg)
  if args.format == 'csv':
    full_titles = ['# Datetime(UTC)'] + full_titles
    print(','.join(full_titles))
    for j, utc in enumerate(x):
      print(','.join([str(utc)] + [str(y[i][j]) for i in range(len(labels))]))
  else:
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
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
        maketitle(
            args,
            f'Histogram of Speedup over {args.baseline.capitalize()} Benchmarked Inductor'
        ))
    plt.savefig(sys.stdout.buffer, format=args.format)


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
          results_map[timestamp] else Datapoint('', ''))
  if args.format == 'csv':
    print(','.join(['# Datetime(UTC)'] + [
        f"Speedup({title}/{args.baseline.capitalize()} Inductor),StdDev"
        for title in titles
    ]))
    for j, x in enumerate(x):
      print(','.join(
          map(str, [x] + [
              v for i in range(len(labels)) for v in (y[i][j].avg, y[i][j].std)
          ])))
  else:
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    ax.axhline(y=1.0, color='lightgray')
    for i in range(len(labels)):
      ax.errorbar(
          x, [v.avg for v in y[i]], [v.std for v in y[i]],
          marker=_markers[i],
          label=titles[i],
          elinewidth=_fig_elinewidth,
          capsize=_fig_capsize)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Geomean Speedup")
    plt.title(
        maketitle(
            args,
            f'Speedup over {args.baseline.capitalize()} Benchmarked Inductor'))
    plt.savefig(sys.stdout.buffer, format=args.format)


def pr_results(results_map: Dict[str, Any], args):
  timestamps = list(results_map.keys())
  timestamps.sort()

  if args.format != 'csv' and not has_matplotlib:
    sys.exit(f'Fatal: cannot find matplotlib, needed for {args.format} output.')

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
      '--baseline',
      default='oldest',
      choices=['oldest', 'latest'],
      help='Inductor baseline to be used for computing speedups.')
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
      "--fig-height",
      type=float,
      default=6.75,
      help="Figure height (inches)",
  )
  parser.add_argument(
      "--fig-width",
      type=float,
      default=9.0,
      help="Figure width (inches)",
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
      "--format",
      default='csv',
      choices=['csv', 'png', 'svg'],
      help='Output format')
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

  parser.add_argument(
      '--metric',
      default='exec',
      choices=['exec', 'compile'],
      help='Metric to extract.')
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

  for filename in filenames:
    process_file(args, results_map, filename)
  process_results(args, results_map)
  if not results_map:
    sys.exit('no results found')
  pr_results(results_map, args)


if __name__ == '__main__':
  main()

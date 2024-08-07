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
from tabulate import tabulate
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

_title_map = {
    'eager': 'Eager',
    'inductor': 'Inductor',
    'openxla+dynamo': 'XLA+Dynamo',
    'openxla+lazytensor': 'XLA+LazyTensor',
}

_test_to_field_name = {
    'inference': 'eval',
    'training': 'train',
}
_fig_elinewidth = 0.5
_fig_capsize = 3

_markers = ('D', 'o', 's', 'v', 'x', '+')


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


def get_backend_name(dynamo: str, xla: str) -> str:
  if dynamo == 'inductor':
    return 'inductor'
  if xla == 'PJRT':
    assert dynamo == 'openxla' or dynamo == None
    xla_name = dynamo
    tracer = 'dynamo'
    if not dynamo:
      xla_name = 'openxla'
      tracer = 'lazytensor'
    return f'{xla_name}+{tracer}'
  assert dynamo == None and xla == None
  return 'eager'


def process_file(args, results_map: Dict[str, Any], filename: str):
  fields = {
      'experiment': [
          'accelerator_model', 'batch_size', 'dynamo', 'test', 'xla'
      ],
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
            print("lllllll: ", k)
            sys.exit(f'JSONL record does not contain key {k}.{kk}. JSONL: {r}')

      # Read in what we need.
      accelerator_model = clean_up_accelerator_model(
          r['experiment']['accelerator_model'])
      if accelerator_model != args.accelerator:
        continue
      model_name = r['model']['model_name']
      if skip_model(args, model_name):
        continue
      xla = r['experiment']['xla']
      dynamo = r['experiment']['dynamo']
      backend = get_backend_name(dynamo, xla)
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
      if backend not in results_map[timestamp]:
        results_map[timestamp][backend] = {}
      if (model_name not in results_map[timestamp][backend]):
        results_map[timestamp][backend][model_name] = {}
      if (batch_size not in results_map[timestamp][backend][model_name]):
        results_map[timestamp][backend][model_name][batch_size] = {}
      results_map[timestamp][backend][model_name][batch_size] = dp


# Speedup of a over baseline ("b"), with errors.
def compute_speedup(a: NamedTuple, b: NamedTuple) -> NamedTuple:
  rel_err_a = a.std / a.avg
  rel_err_b = b.std / b.avg
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
  base_backend = args.backends[0]
  if args.baseline == 'oldest':
    # A benchmark's baseline is the oldest `base_backend` perf number we have
    # for it. This way we can track perf improvements over time.
    for ts in timestamps:
      if base_backend not in results_map[ts]:
        continue
      populate_baseline(baseline, results_map[ts][base_backend])

  elif args.baseline == 'latest':
    # Pick only results from the latest timestamp.
    ts = timestamps[-1]
    if base_backend not in results_map[ts]:
      sys.exit(f'No {base_backend} results in the latest timestamp {ts}')
    populate_baseline(baseline, results_map[ts][base_backend])
  return baseline


def process_results(args, results_map: Dict[str, Any]):
  baseline = compute_baseline(args, results_map)
  for timestamp in results_map:
    acc_map = results_map[timestamp]

    for backend in sorted(_title_map.keys()):
      compute_speedups(acc_map, baseline, f'{backend}:speedups', backend)


def maketitle(args, title: str):
  if args.title:
    title += f'\n{args.title}'
  return title


def get_pr_titles(args):
  titles = [_title_map[t] for t in args.backends]
  data_labels = args.backends
  return [titles, data_labels]


def speedup_header(title: str, backend_name: str, args):
  if args.format == 'tab':
    return f'Speedup\n{title}\nover\n{args.baseline.capitalize()}\n{backend_name}'
  return f'Speedup({title}/{args.baseline.capitalize()} {backend_name})'


def modelname_header(model: str, args):
  if args.format == 'tab':
    return f'ModelName\n{model}'
  return f'ModelName({model})'


def percentile_header(title: str, p: str, args):
  if args.format == 'tab':
    return f'{title}\n{p}'
  return f'{title} {p}'


def pr_text(headers, rows, args):
  if args.format == 'csv':
    if headers:
      headers[0] = f'# {headers[0]}'
    print(','.join(headers))
    for row in rows:
      print(','.join([str(f) if f is not None else '' for f in row]))
  elif args.format == 'tab':
    print(
        tabulate(rows, headers=headers, tablefmt='fancy_grid', floatfmt='.2f'))


def pr_latest(results_map: Dict[str, Any], args, timestamps: List[str]):
  titles, data_labels = get_pr_titles(args)
  # speedups[backend] is the list of speedups vs. the baseline for that backend.
  # Speedups are sorted in ascending order so that speedups[backend] is
  # monotonically increasing. That is, due to the sorting, it is unlikely that
  # speedups["foo"][i] and speedups["bar"][i] will correspond to the same model.
  speedups = [[] for _ in titles]
  # model_names[backend][i] contains the model name corresponding to
  # speedups[backend][i].
  model_names = [[] for _ in titles]
  base_backend_name = _title_map[args.backends[0]]

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
  if all(not x for x in speedups):
    logger.warning(f'cannot find data for accelerator {args.accelerator}')
    return

  if args.format == 'csv' or args.format == 'tab':
    headers = ['Workload'] + [
        header for title in titles for header in [
            speedup_header(title, base_backend_name, args), 'StdDev',
            modelname_header(title, args)
        ]
    ]

    # Not all models run in all backends, so it is likely that
    # len(speedups["foo"]) != len(speedups["bar"]). We therefore pad the speedup
    # lists with "None" elements so that we have a "full table", i.e. all lists
    # have the same length. This makes it trivial to generate correct CSV or
    # tabular output.
    num_rows = max([len(l) for l in speedups])

    def pad_array(arr, desired_len, val):
      if len(arr) >= desired_len:
        return
      arr += [val] * (desired_len - len(arr))

    for i in range(len(titles)):
      pad_array(speedups[i], num_rows, Datapoint(None, None))
      pad_array(model_names[i], num_rows, None)

    rows = []
    for j in range(num_rows):
      rows += [[j] + [
          v for i in range(len(titles))
          for v in (speedups[i][j].avg, speedups[i][j].std, model_names[i][j])
      ]]
    pr_text(headers, rows, args)
  else:
    plt.figure(figsize=(args.fig_width, args.fig_height))
    plt.axhline(y=1.0, color='lightgray')
    for i in range(len(titles)):
      plt.errorbar([j for j in range(len(speedups[i]))],
                   [v.avg for v in speedups[i]], [v.std for v in speedups[i]],
                   label=titles[i],
                   marker=_markers[i % len(_markers)],
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
            f'Speedup over {args.baseline.capitalize()} Benchmarked {base_backend_name}'
        ))
    plt.xlabel('Workload Number')
    plt.ylabel(f'Speedup')
    plt.savefig(sys.stdout.buffer, format=args.format)


def pr_latest_grouped(results_map: Dict[str, Any], args, timestamps: List[str]):
  titles, backend_labels = get_pr_titles(args)
  acc_map = results_map[timestamps[-1]]
  base_backend_name = _title_map[args.backends[0]]

  # Iterate over all the results in the latest timestamp and store them
  # in a per-model dict, i.e.
  #   per_model[model_name][backend_label] = X.XX
  per_model = {}
  for backend in backend_labels:
    speedup_label = f'{backend}:speedups'
    model_label = f'{speedup_label}:model_name'
    for i, model_name in enumerate(acc_map.get(model_label, ())):
      if model_name not in per_model:
        per_model[model_name] = {}
      per_model[model_name][backend] = pr_round(acc_map[speedup_label][i])

  # Create a dict with (key: model_name, value: list of results), where the
  # list of results corresponds to each one of backend_labels. If a result
  # doesn't exist, its corresponding element is Datapoint(None, None).
  per_model_list = {}
  for model_name, per_backend in per_model.items():
    per_model_list[model_name] = []
    for backend in backend_labels:
      result = per_backend[backend] if backend in per_backend else Datapoint(
          None, None)
      per_model_list[model_name].append(result)

  if not per_model_list:
    logger.warning(f'cannot find data for accelerator {args.accelerator}')
    return

  # Append the GEOMEAN after all the sorted model names.
  sorted_model_names = sorted(per_model_list.keys(), key=str.casefold)
  sorted_model_names.append('GEOMEAN')
  per_model_list['GEOMEAN'] = []
  for backend in backend_labels:
    per_model_list['GEOMEAN'].append(
        pr_round(acc_map[f'{backend}:speedups:gmean']))

  if args.format == 'csv' or args.format == 'tab':
    headers = ['ModelName'] + [
        header for title in titles for header in
        [speedup_header(title, base_backend_name, args), 'StdDev']
    ]
    rows = []
    for model_name in sorted_model_names:
      rows += [[model_name] + [
          v for i in range(len(per_model_list[model_name]))
          for v in (per_model_list[model_name][i].avg,
                    per_model_list[model_name][i].std)
      ]]
    pr_text(headers, rows, args)
  else:
    # Plot a horizontal bar chart because we might have lots of models.
    fig, ax = plt.subplots(
        figsize=(args.fig_width, args.fig_height), layout='constrained')
    ax.axvline(x=1.0, color='lightgray')
    n = len(sorted_model_names)
    # Plot downwards, i.e. y(model_name='a') > y(model_name='z').
    y = [n - v for v in range(n)]
    # Leave some room between bar groups.
    bar_width = 0.70 / len(backend_labels)

    per_backend_list = {}
    for i, backend in enumerate(backend_labels):
      per_backend_list[backend] = []
      for model_name in sorted_model_names:
        per_backend_list[backend].append(per_model_list[model_name][i])

    i = 0
    for model_name, values in per_backend_list.items():
      offset = bar_width * i
      i += 1
      rects = ax.barh(
          [v - offset for v in y],
          [round(v.avg, 2) if v.avg is not None else np.nan for v in values],
          bar_width,
          label=model_name,
          xerr=[v.std if v.std is not None else np.nan for v in values])
      ax.bar_label(rects)
    plt.legend()
    plt.title(
        maketitle(
            args, f'Speedup over {args.baseline.capitalize()} '
            f'Benchmarked {base_backend_name}'))
    plt.xlabel(f'Speedup')
    plt.ylabel('Workload')
    # Make sure the ytick lands at the middle of the bar group.
    # Note that 'v' is at the middle of the first bar; we plot downwards
    # so we have to subtract from it.
    plt.yticks([v - (len(backend_labels) - 1) * bar_width / 2.0 for v in y],
               sorted_model_names)
    plt.margins(y=0)
    plt.xlim(left=0)
    plt.savefig(sys.stdout.buffer, format=args.format)


def pr_histogram(results_map: Dict[str, Any], args, timestamps: List[str]):
  titles, data_labels = get_pr_titles(args)
  percentiles = [f'p{p}' for p in (95, 50, 5)]
  labels = [f'{pfx}:speedups:{p}' for pfx in data_labels for p in percentiles]
  full_titles = [
      percentile_header(title, p, args) for title in titles for p in percentiles
  ]
  base_backend_name = _title_map[args.backends[0]]
  x = []
  y = [[] for i in range(len(labels))]
  for timestamp in timestamps:
    if labels[0] in results_map[timestamp]:
      x.append(datetime.utcfromtimestamp(float(timestamp)))
      for i, label in enumerate(labels):
        y[i].append((pr_round(results_map[timestamp][label]) if label
                     in results_map[timestamp] else Datapoint(None, None)).avg)
  if args.format == 'csv' or args.format == 'tab':
    headers = ['Datetime(UTC)'] + full_titles
    rows = []
    for j, utc in enumerate(x):
      rows += [[utc] + [y[i][j] for i in range(len(labels))]]
    pr_text(headers, rows, args)
  else:
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    ax.axhline(y=1.0, color='lightgray')
    linestyles = ('solid', 'dotted', 'dashed', 'dashdot')
    for i, label in enumerate(labels):
      style = int(i / len(percentiles))
      ax.plot(
          x,
          y[i],
          label=full_titles[i],
          marker=_markers[style % len(_markers)],
          linestyle=linestyles[style % len(linestyles)])
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Geomean Speedup")
    plt.title(
        maketitle(
            args,
            f'Histogram of Speedup over {args.baseline.capitalize()} Benchmarked {base_backend_name}'
        ))
    plt.savefig(sys.stdout.buffer, format=args.format)


def pr_gmean(results_map: Dict[str, Any], args, timestamps: List[str]):
  label = f'speedups:gmean'
  x = []
  titles, data_labels = get_pr_titles(args)
  labels = [f"{x}:speedups:gmean" for x in data_labels]
  base_backend_name = _title_map[args.backends[0]]
  y = [[] for _ in labels]
  for timestamp in timestamps:
    if all(label not in results_map[timestamp] for label in labels):
      continue
    x.append(datetime.utcfromtimestamp(float(timestamp)))
    for i, label in enumerate(labels):
      y[i].append(
          pr_round(results_map[timestamp][label]) if label in
          results_map[timestamp] else Datapoint(None, None))
  if args.format == 'csv' or args.format == 'tab':
    headers = ['Datetime(UTC)'] + [
        header for title in titles for header in
        [speedup_header(title, base_backend_name, args), 'StdDev']
    ]
    rows = []
    for j, x in enumerate(x):
      rows += [
          [x] +
          [v for i in range(len(labels)) for v in (y[i][j].avg, y[i][j].std)]
      ]
    pr_text(headers, rows, args)
  else:
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    ax.axhline(y=1.0, color='lightgray')
    for i in range(len(labels)):
      ax.errorbar(
          x, [v.avg if v.avg is not None else np.nan for v in y[i]],
          [v.std if v.std is not None else np.nan for v in y[i]],
          marker=_markers[i % len(_markers)],
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
            f'Speedup over {args.baseline.capitalize()} Benchmarked {base_backend_name}'
        ))
    plt.savefig(sys.stdout.buffer, format=args.format)


def pr_results(results_map: Dict[str, Any], args):
  timestamps = list(results_map.keys())
  timestamps.sort()

  if not has_matplotlib and (args.format == 'png' or args.format == 'svg'):
    sys.exit(f'Fatal: cannot find matplotlib, needed for {args.format} output.')

  if args.report == 'latest':
    return pr_latest(results_map, args, timestamps)
  elif args.report == 'latest_grouped':
    return pr_latest_grouped(results_map, args, timestamps)
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
      '--backends',
      type=str,
      action='extend',
      nargs='+',
      help=f'''List of backends to report on.
      Valid: {sorted(_title_map.keys())}.
      Note: the first element is used as the baseline backend.''')
  parser.add_argument(
      '--baseline',
      default='oldest',
      choices=['oldest', 'latest'],
      help='Baseline point in time to be used for computing speedups.')
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
      choices=['csv', 'png', 'svg', 'tab'],
      help='Output format')
  parser.add_argument('input_file', nargs='+')
  parser.add_argument(
      '--report',
      default='speedup',
      choices=['latest', 'latest_grouped', 'histogram', 'speedup'],
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
  if not args.backends:
    if args.test == 'inference':
      args.backends = ['inductor', 'openxla+dynamo']
    else:
      args.backends = ['inductor', 'openxla+dynamo']
  for backend in args.backends:
    if backend not in _title_map:
      sys.exit(f"error: argument --backends: invalid choice: '{backend}' "
               f"(choose from {sorted(_title_map.keys())})")

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

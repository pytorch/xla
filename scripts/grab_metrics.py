#!/usr/bin/env python
# Given a log file in which the XLA metrics report has been dumped, extracts the
# different metrics across multiple points and produces data in a format which
# can be graphed.
# Can also produce data which is a combination of other metric, using the
# --synth parameters:
#
#   --synth 'LiveDataHandles:CreateDataHandles - DestroyDataHandles'
#

from __future__ import print_function

import argparse
import collections
import re
import sys


def parse_metrics(lines):
  # Counter: CreateCompileHandles
  #  Value: 1631
  metrics = collections.defaultdict(list)
  metric = None
  for line in lines:
    if metric is not None:
      m = re.match(r'\s*Value: ([^\s]+)', line)
      if m:
        metrics[metric].append(m.group(1))
      metric = None
    else:
      m = re.match(r'Counter: ([^\s]+)', line)
      if m:
        metric = m.group(1)
  return metrics


def create_metric_report(args, metric, metric_data):
  print('[{}]'.format(metric))
  for i, v in enumerate(metric_data):
    print('{}\t{}'.format(i, v))


def process_synth(args, synth, metrics):
  name, expr = synth.split(':', 1)
  xvars = set()
  for m in re.finditer(r'[a-zA-Z_][a-zA-Z_0-9]*', expr):
    xvars.add(m.group(0))
  xvars = list(xvars)
  xmetrics = []
  for v in xvars:
    metric_data = metrics.get(v, None)
    if metric_data is None:
      raise RuntimeError('Unknown metric: {}'.format(v))
    xmetrics.append(metric_data)
  print('[{}]'.format(name))
  x = 0
  while True:
    env = {}
    for i, v in enumerate(xvars):
      metric_data = xmetrics[i]
      if x >= len(metric_data):
        break
      env[v] = float(metric_data[x])
    if len(env) < len(xvars):
      break
    y = eval(expr, env)
    print('{}\t{}'.format(x, y))
    x += 1


def create_report(args, metrics):
  if args.metric:
    metric_data = metrics.get(args.metric, None)
    if metric_data is None:
      raise RuntimeError('Unknown metric: {}'.format(args.metric))
    create_metric_report(args, args.metric, metric_data)
  else:
    for metric in metrics.keys():
      create_metric_report(args, metric, metrics[metric])
  for synth in args.synth:
    process_synth(args, synth, metrics)


def process_metrics(args):
  fd = sys.stdin if args.input is None else open(args.input, 'r')
  metrics = parse_metrics(fd)
  create_report(args, metrics)


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--input', type=str)
  arg_parser.add_argument('--metric', type=str)
  arg_parser.add_argument('--synth', action='append', type=str)
  args, files = arg_parser.parse_known_args()
  args.files = files
  process_metrics(args)

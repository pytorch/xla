#!/usr/bin/env python

from __future__ import print_function

import argparse
import copy
import getpass
import glob
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import tempfile

_QUIT = False
_DEFAULT_VMODULE = [
    'tensor=5', 'computation_client=5', 'xrt_computation_client=5',
    'aten_xla_type=1'
]


def term_handler(signum, frame):
  global _QUIT
  _QUIT = True
  print('Termination handler called!', file=sys.stderr)


def get_metrics_file_path(outdir):
  return os.path.join(outdir, 'metrics')


def get_graphs_file_path(outdir):
  return os.path.join(outdir, 'graphs')


def get_log_file_path(outdir):
  return os.path.join(outdir, 'logs')


def get_graphdir_path(outdir):
  return os.path.join(outdir, 'graphdir')


def get_metrics_imgdir_path(outdir):
  return os.path.join(outdir, 'metrics_imgdir')


def get_metrics_report_path(outdir):
  return os.path.join(outdir, 'metrics_report')


def get_graph_report_path(outdir):
  return os.path.join(outdir, 'graph_report')


def get_scripts_path():
  return os.path.dirname(os.path.realpath(__file__))


def get_first_file(path):
  if os.path.isfile(path):
    return path
  path += '.0'
  return path if os.path.isfile(path) else None


def build_vmodule(args, default):
  default = list(default)
  if args.vmodule:
    default += args.vmodule.split(',')
  return ','.join(default)


def show_env(env, fd=sys.stdout):
  print('XLA Environment:', file=fd)
  for k, v in env.items():
    if re.match(r'(XLA_|XRT_|TF_)', k):
      print('  {}={}'.format(k, v), file=fd)


def create_env(args):
  env = copy.copy(os.environ)
  env['XLA_IR_DEBUG'] = '1'
  env['XLA_HLO_DEBUG'] = '1'
  env['TF_CPP_LOG_THREAD_ID'] = '1'
  env['XLA_DUMP_FATAL_STACK'] = '1'
  env['TF_CPP_VMODULE'] = build_vmodule(args, _DEFAULT_VMODULE)
  env['TF_CPP_MIN_LOG_LEVEL'] = '0'
  env['XLA_SAVE_TENSORS_FILE'] = get_graphs_file_path(args.outdir)
  if args.hlo:
    env['XLA_SAVE_TENSORS_FMT'] = 'hlo'
  env['XLA_METRICS_FILE'] = get_metrics_file_path(args.outdir)
  show_env(env)
  return env


def grab_graphs(args):
  graphs_file = get_first_file(get_graphs_file_path(args.outdir))
  if graphs_file is not None:
    grab_graph_path = os.path.join(get_scripts_path(), 'grab_graphs.py')
    report = subprocess.check_output([
        grab_graph_path, '--graphdir={}'.format(get_graphdir_path(args.outdir)),
        '--collisions_check', graphs_file
    ]).decode('utf-8')
    with open(get_graph_report_path(args.outdir), 'w') as fd:
      fd.write(report)


def grab_metrics(args):
  metrics_file = get_first_file(get_metrics_file_path(args.outdir))
  if metrics_file is not None:
    grab_metrics_path = os.path.join(get_scripts_path(), 'grab_metrics.py')
    report = subprocess.check_output([
        grab_metrics_path,
        '--image_path={}'.format(get_metrics_imgdir_path(args.outdir)),
        metrics_file
    ]).decode('utf-8')
    with open(get_metrics_report_path(args.outdir), 'w') as fd:
      fd.write(report)


def create_temp_folder():
  seq = -1
  while True:
    dir_name = 'debug_run-{}-{}'.format(socket.gethostname(), getpass.getuser())
    if seq >= 0:
      dir_name += '-{}'.format(seq)
    temp_folder = os.path.join(tempfile.gettempdir(), dir_name)
    if not os.path.isdir(temp_folder):
      os.mkdir(temp_folder)
      return temp_folder
    seq += 1


def setup_outdir(args):
  if args.outdir is None:
    args.outdir = create_temp_folder()
    print('Writing run results to {}'.format(args.outdir), file=sys.stderr)
  elif os.path.isdir(args.outdir):
    raise RuntimeError('Output folder must not exist: {}'.format(args.outdir))
  else:
    os.mkdir(args.outdir)


def targz(folder, tarfile):
  dirbase = os.path.dirname(folder)
  dirname = os.path.basename(folder)
  if subprocess.call(['tar', '-C', dirbase, '-czf', tarfile, dirname]) != 0:
    raise RuntimeError('Failed to create folder {} archive into {}'.format(
        folder, tarfile))


def read_proc_output(logfd, offset, outfd=None):
  size = os.fstat(logfd).st_size
  if size > offset:
    data = os.pread(logfd, size - offset, offset)
    if outfd is not None:
      os.write(outfd, data)
    offset = size
  else:
    data = None
  return offset, data


def terminate_process(proc, term_wait=10):
  proc.terminate()
  try:
    proc.wait(timeout=term_wait)
  except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()


def run_and_monitor(args):
  env = create_env(args)
  logfile = get_log_file_path(args.outdir)
  logfd = os.open(logfile, os.O_RDWR | os.O_CREAT, mode=0o664)
  offset = 0
  proc = subprocess.Popen(
      args.cmdline, stdout=logfd, stderr=subprocess.STDOUT, env=env)

  while not _QUIT and proc.poll() is None:
    offset, data = read_proc_output(logfd, offset, outfd=sys.stdout.fileno())
    if data is None:
      time.sleep(1.0)

  terminate_process(proc)
  read_proc_output(logfd, offset, outfd=sys.stdout.fileno())
  os.close(logfd)


def process_output(args):
  grab_graphs(args)
  grab_metrics(args)
  if args.outfile:
    targz(args.outdir, args.outfile)
  if args.tidy:
    shutil.rmtree(args.outdir)


def run_binary(args):
  setup_outdir(args)
  signal.signal(signal.SIGINT, term_handler)
  signal.signal(signal.SIGTERM, term_handler)
  run_and_monitor(args)
  process_output(args)


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument(
      '--hlo', action='store_true', help='Grab IR graphs in HLO format')
  arg_parser.add_argument(
      '--tidy',
      action='store_true',
      help='Remove output folder after creating the tar.gz report file')
  arg_parser.add_argument(
      '--outdir',
      type=str,
      default=None,
      help='The temporary output folder (must not exist)')
  arg_parser.add_argument(
      '--outfile',
      type=str,
      default=None,
      help='The location of the tar.gz debug report file')
  arg_parser.add_argument(
      '--vmodule',
      type=str,
      default=None,
      help='Extra --vmodule files to be added. A list of comma-separated NAME=LEVEL'
  )
  arg_parser.add_argument('cmdline', nargs='+')

  args = arg_parser.parse_args()
  run_binary(args)

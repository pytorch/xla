#!/usr/bin/env python

from __future__ import print_function

import argparse
import distutils.sysconfig
import glob
import os
import subprocess


def find_torch_xla_site(site_path):
  dirs = glob.glob(os.path.join(site_path, 'torch_xla*'))
  # Get the most recent one.
  return sorted(dirs, key=os.path.getmtime)[-1]


def list_rpaths(path):
  if subprocess.call(['patchelf', '--shrink-rpath', path]) != 0:
    raise RuntimeError('Failed to shrink RPATH folders: {}'.format(path))
  return subprocess.check_output(['patchelf', '--print-rpath',
                                  path]).decode('utf-8').strip('\n').split(':')


def set_rpaths(path, rpaths):
  if subprocess.call(['patchelf', '--set-rpath', ':'.join(rpaths), path]) != 0:
    raise RuntimeError('Failed to set RPATH folders {}: {}'.format(
        rpaths, path))


def fixup_binary(args):
  site_path = distutils.sysconfig.get_python_lib()
  site_xla_path = find_torch_xla_site(site_path)
  rpaths = list_rpaths(args.binary)
  rpaths = [
      os.path.join(site_xla_path, 'torch_xla/lib'), site_xla_path,
      os.path.join(site_path, 'torch/lib'),
  ] + rpaths
  set_rpaths(args.binary, rpaths)


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument(
      'binary',
      type=str,
      metavar='BINARY',
      help='The path to the binary to be patched')
  args, files = arg_parser.parse_known_args()
  fixup_binary(args)

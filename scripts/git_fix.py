#!/usr/bin/env python3
"""Fixes the format of locally changed git files.
"""

import logging
import os
import re
import sys
from typing import Optional

# Root of the PyTorch/XLA repo.
_PTXLA_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')

logger = logging.getLogger(__name__)


def get_uncommitted_changed_added_files() -> set[str]:
  """Gets the list of changed or added files that are not committed yet.

  This includes untracked files (i.e. newly created files not added to git yet).
  """

  # Run `git status --porcelain | grep -E '^(M|A|\?\?)\s' | awk '{print $NF}'` to get the list.
  # How this works:
  #   - git status --porcelain : print the changed files prefixed by their statuses.
  #   - grep -E '^(M|A|\?\?)\s' : find changed (M), added (A), or untracked (??) files.
  #   - awk '{print $NF}' : print the last field of each line, which is the file path.
  files = os.popen(
      'git status --porcelain | grep -E \'^(M|A|\?\?)\s\' | awk \'{print $NF}\''
  ).read().strip().split('\n')
  if '' in files:
    files.remove('')
  return set(files)


def get_cplusplus_files(files: set[str]) -> set[str]:
  """Filters the given list of files and returns the C++ files."""

  return {
      f for f in files
      if os.path.splitext(f)[1] in ('.cc', '.h', '.cpp', '.cxx')
  }


def get_python_files(files: set[str]) -> set[str]:
  """Filters the given list of files and returns the Python files."""

  return {f for f in files if os.path.splitext(f)[1] == '.py'}


def main() -> None:
  logging.basicConfig(level=logging.INFO)

  os.chdir(_PTXLA_DIR)
  files = get_uncommitted_changed_added_files()

  # We don't use `fix_cplusplus_files(...) and fix_python_files(...)` as we
  # don't want shortcircuiting.
  success = fix_cplusplus_files(get_cplusplus_files(files))
  if not fix_python_files(get_python_files(files)):
    success = False

  if not success:
    sys.exit(1)


if __name__ == '__main__':
  main()

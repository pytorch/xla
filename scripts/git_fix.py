#!/usr/bin/env python3
"""Fixes the format of locally changed git files.
"""

import argparse
import os
import re
import sys

# Root of the PyTorch/XLA repo.
_PTXLA_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')

# Path of this script, relative to the repo root.
_SCRIPT_PATH = os.path.abspath(__file__)[len(_PTXLA_DIR) + 1:]

# Names of CLIs for formatting files.
_CLANG_FORMAT = 'clang-format-11'
_YAPF = 'yapf'

# Path of the git pre-push hook script, relative to the repo root.
_GIT_PRE_PUSH_HOOK_PATH = '.git/hooks/pre-push'


def get_uncommitted_changed_added_files() -> set[str]:
  """Gets the list of changed or added files that are not committed yet.

  This includes untracked files (i.e. newly created files not added to git
  yet).
  """

  # Each line is a filepath prefixed by its status.
  lines = os.popen('git status --porcelain').read().strip().split('\n')
  files = set()
  for line in lines:
    # Find changed (M), added (A), or untracked (??) files.
    if re.match(r'^\s*(M|A|\?\?)\s', line):
      # The last field of the line is the filepath.
      files.add(line.split()[-1])
  return files


def get_committed_changed_added_files() -> set[str]:
  """Gets the list of changed or added files that are committed locally.
   
  These are the files that are committed in the local branch but not jyet
  merged to the origin/master branch.
  """

  # Each line is a filepath.
  lines = os.popen(
      # --name-only : include filepaths only in the output.
      # --diff-filter=AM : include only added (A) or modified (M) files.
      # --no-renames : if a file is renamed, treat it as a deleted file and
      #     an added file, so that the added file will be included in the
      #     result.
      # origin/master...HEAD : compare the local branch HEAD with
      #     origin/master, showing changes that exist on your local branch
      #     but not on origin/master.
      'git diff --name-only --diff-filter=AM --no-renames origin/master...HEAD'
  ).read().strip().split('\n')
  return set(lines)


def get_cplusplus_files(files: set[str]) -> set[str]:
  """Filters the given list of files and returns the C++ files."""

  return {
      f for f in files
      if os.path.splitext(f)[1] in ('.cc', '.h', '.cpp', '.cxx')
  }


def get_python_files(files: set[str]) -> set[str]:
  """Filters the given list of files and returns the Python files."""

  return {f for f in files if os.path.splitext(f)[1] == '.py'}


def tool_is_installed(file_type: str, tool: str) -> bool:
  """Checks if the given tool is installed.

  Args:
    file_type: The type of the file (e.g. C++, python).
    tool: The name of the tool.

  Returns:
    True if the tool is already installed.  
  """

  # Is the tool already installed?
  if os.system(f'which {tool} > /dev/null') == 0:
    return True

  print(
      f'WARNING: {tool} is not installed. It\'s needed for formatting '
      f'{file_type} files. Please install it and retry.',
      file=sys.stderr)
  return False


def format_files(file_type: str, tool: str, format_command: str,
                 files: set[str]) -> bool:
  """Fixes the formatting of the given files.
  
  Args:
    file_type: The type of the file (e.g. C++, python).    
    tool: The name of the tool.
    format_command: The command to format the files.
    files: The files to format.
  Returns:
    True if the formatting was successful.
  """

  if not files:
    return True

  if not tool_is_installed(file_type, tool):
    return False

  command = f'{format_command} {" ".join(files)}'
  if os.system(command) == 0:
    print(
        f'Successfully formatted {file_type} files:\n' +
        '\n'.join(sorted('  ' + f for f in files)),
        file=sys.stderr)
    return True

  print(
      f'WARNING: Failed to format {file_type} files via command: {command}',
      file=sys.stderr)
  return False


def format_cplusplus_files(files: set[str]) -> bool:
  """Fixes the formatting of the given C++ files.
  
  Returns:
    True if the formatting was successful.
  """

  return format_files('C++', _CLANG_FORMAT, f'{_CLANG_FORMAT} -i -style=file',
                      files)


def format_python_files(files: set[str]) -> bool:
  """Fixes the formatting of the given Python files.

  Returns:
    True if the formatting was successful.
  """

  return format_files('python', _YAPF, f'{_YAPF} -i', files)


def set_git_push_hook() -> bool:
  """Sets up `git push` to automatially run this script before pushing.
  
  Returns:
    True if the set-up was successful.
  """

  if os.path.isfile(_GIT_PRE_PUSH_HOOK_PATH):
    print(
        f'git pre-push hook already exists at {_GIT_PRE_PUSH_HOOK_PATH}.',
        file=sys.stderr)
    # Ask the user if they want to overwrite it.
    overwrite = input(
        f'Are you sure you want to overwrite {_GIT_PRE_PUSH_HOOK_PATH}? [y/N] ')
    if overwrite.lower() != 'y':
      print('Skipping git pre-push hook setup.', file=sys.stderr)
      return False

  # Write the current script to the git pre-push hook.
  with open(_GIT_PRE_PUSH_HOOK_PATH, 'w') as f:
    f.write(f'''#!/bin/bash
# This hook is automatically set by `{_SCRIPT_PATH } --set_git_push_hook`.)
{_SCRIPT_PATH}
''')


def main() -> None:
  arg_parser = argparse.ArgumentParser(
      prog=_SCRIPT_PATH, description=__doc__)
  arg_parser.add_argument(
      '--set_git_push_hook',
      action='store_true',
      help='set up `git push` to automatically run this script before pushing')
  args = arg_parser.parse_args()

  os.chdir(_PTXLA_DIR)
  if args.set_git_push_hook:
    sys.exit(0 if set_git_push_hook() else 1)

  files = (
      get_uncommitted_changed_added_files() |
      get_committed_changed_added_files())

  # We don't use `format_cplusplus_files(...) and format_python_files(...)` as
  # we don't want shortcircuiting.
  success = format_cplusplus_files(get_cplusplus_files(files))
  if not format_python_files(get_python_files(files)):
    success = False

  if not success:
    sys.exit(1)


if __name__ == '__main__':
  main()

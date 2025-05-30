#!/usr/bin/env python3
"""Updates third party dependencies.

Usage:

  scripts/update_deps.py

    updates the versions of OpenXLA, libtpu, and JAX as used in
    PyTorch/XLA. In particular, it:

    - updates OpenXLA to the latest commit,
    - updates libtpu to the latest nightly build, and
    - updates JAX to the latest nightly build.
"""

import logging
import os
import platform
import re
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_PLATFORM = platform.machine()  # E.g. 'x86_64'.

# Root of the PyTorch/XLA repo.
_PTXLA_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')
# Scratch space.
_TMP_DIR = '/tmp/pytorch_xla_deps'
# Path to the repo's WORKSPACE file.
_WORKSPACE_PATH = os.path.join(_PTXLA_DIR, 'WORKSPACE')
# Path to the setup.py file.
_SETUP_PATH = os.path.join(_PTXLA_DIR, 'setup.py')

# Page listing libtpu nightly builds.
_LIBTPU_BUILDS_URL = 'https://storage.googleapis.com/libtpu-wheels/index.html'
# Page listing jax nightly builds.
_JAX_BUILDS_URL = 'https://storage.googleapis.com/jax-releases/jax_nightly_releases.html'


def clean_tmp_dir() -> None:
  """Cleans up the temp dir."""

  os.system(f'rm -rf {_TMP_DIR}')
  os.system(f'mkdir -p {_TMP_DIR}')


def get_last_xla_commit_and_date() -> tuple[str, str]:
  """Finds the latest commit in the master branch of https://github.com/openxla/xla.
  
  Returns:
    A tuple of the latest commit SHA and its date (YYYY-MM-DD).
  """

  # Get the latest commit in the master branch of openxla.
  clean_tmp_dir()
  os.system(
      f'git clone --depth=1 https://github.com/openxla/xla {_TMP_DIR}/xla')
  commit = os.popen(f'cd {_TMP_DIR}/xla && git rev-parse HEAD').read().strip()

  # Get the date of the commit, in the format of YYYY-MM-DD.
  date = os.popen(
      f'cd {_TMP_DIR}/xla && git show -s --format=%cd --date=short {commit}'
  ).read().strip()
  return commit, date


def update_openxla() -> bool:
  """Updates the OpenXLA version in the WORKSPACE file to the latest commit.
  
  Returns:
    True if the WORKSPACE file was updated, False otherwise.
  """

  commit, date = get_last_xla_commit_and_date()

  with open(_WORKSPACE_PATH, 'r') as f:
    ws_lines = f.readlines()

  # Update the `xla_hash = ...` line with the latest commit and date.
  found_xla_hash = False
  for i, line in enumerate(ws_lines):
    if re.match(r'^xla_hash\s*=', line):
      found_xla_hash = True
      ws_lines[i] = f"xla_hash = '{commit}'  # Committed on {date}.\n"
      break

  if not found_xla_hash:
    logger.error('Could not find xla_hash in WORKSPACE file.')
    return False

  with open(_WORKSPACE_PATH, 'w') as f:
    f.writelines(ws_lines)

  logger.info('Updated the OpenXLA version in WORKSPACE.')
  return True


def find_latest_nightly(html_lines: list[str],
                        build_re: str) -> Optional[tuple[str, str, str]]:
  """Finds the latest nightly build from the list of HTML lines.
  
  Args:
    html_lines: A list of HTML lines to search for the nightly build.
    build_re: A regular expression for matching the nightly build line.
      It must have 3 capture groups: the version, the date, and the name suffix.

  Returns:
    A tuple of the version, date, and suffix of the latest nightly build,
    or None if no build is found.
  """

  build_re = re.compile(build_re)
  latest_version, latest_date, latest_suffix = '', '', ''
  found_build = False
  for line in html_lines:
    m = build_re.match(line)
    if m:
      found_build = True
      version, date, suffix = m.groups()
      if date > latest_date:
        latest_version, latest_date, latest_suffix = version, date, suffix

  if found_build:
    return latest_version, latest_date, latest_suffix

  return None


def find_latest_libtpu_nightly() -> Optional[tuple[str, str, str]]:
  """Finds the latest libtpu nightly build for the current platform.
  
  Returns:
    A tuple of the version, date, and suffix of the latest libtpu nightly build,
    or None if no build is found.
  """

  # Read the libtpu nightly build page.
  clean_tmp_dir()
  os.system('curl -s {} > {}/libtpu_builds.html'.format(_LIBTPU_BUILDS_URL,
                                                        _TMP_DIR))
  with open(f'{_TMP_DIR}/libtpu_builds.html', 'r') as f:
    html_lines = f.readlines()

  # Search for lines like
  # <a href="...">libtpu/libtpu-0.0.16.dev20250530+nightly-py3-none-manylinux_2_31_x86_64.whl</a><br>
  return find_latest_nightly(
      html_lines,
      r'.*<a href=.*?>libtpu/libtpu-(.*?)\.dev(\d{8})\+nightly-(.*?)_' +
      _PLATFORM + r'\.whl</a>')


def find_latest_jax_nightly() -> Optional[tuple[str, str, str]]:
  """Finds the latest JAX nightly build.
  
  Returns:
    A tuple of the jax version, jaxlib version, and date of the latest JAX nightly build,
    or None if no build is found.
  """

  # Read the nightly jax build page.
  clean_tmp_dir()
  os.system('curl -s {} > {}/jax_builds.html'.format(_JAX_BUILDS_URL, _TMP_DIR))
  with open(f'{_TMP_DIR}/jax_builds.html', 'r') as f:
    html_lines = f.readlines()

  # Find lines like
  # <a href=...>jax/jax-0.6.1.dev20250428-py3-none-any.whl</a>
  jax_build = find_latest_nightly(
      html_lines, r'.*<a href=.*?>jax/jax-(.*?)\.dev(\d{8})-(.*)\.whl</a>')
  if not jax_build:
    logger.error(
        f'Could not find latest jax nightly build in {_JAX_BUILDS_URL}.')
    return None

  # Find lines like
  # <a href=...>nocuda/jaxlib-0.6.1.dev20250428-....whl</a>
  jaxlib_build = find_latest_nightly(
      html_lines,
      r'.*<a href=.*?>nocuda/jaxlib-(.*?)\.dev(\d{8})-(.*)\.whl</a>')
  if not jaxlib_build:
    logger.error(
        f'Could not find latest jaxlib nightly build in {_JAX_BUILDS_URL}.')
    return None

  jax_version, jax_date, _ = jax_build
  jaxlib_version, jaxlib_date, _ = jaxlib_build
  if jax_date != jaxlib_date:
    logger.error(
        f'The latest jax date {jax_date} != the latest jaxlib date {jaxlib_date} in {_JAX_BUILDS_URL}.'
    )
    return None

  return jax_version, jaxlib_version, jax_date


def update_libtpu() -> bool:
  """Updates the libtpu version in setup.py to the latest nightly build.
  
  Returns:
    True if the setup.py file was updated, False otherwise.
  """

  result = find_latest_libtpu_nightly()
  if not result:
    return False

  version, date, suffix = result

  with open(_SETUP_PATH, 'r') as f:
    setup_lines = f.readlines()

  # Update the lines for specifying the libtpu version.
  found_libtpu_version, found_libtpu_date, found_libtpu_wheel = False, False, False
  for i, line in enumerate(setup_lines):
    if re.match(r'_libtpu_version\s*=', line):
      found_libtpu_version = True
      setup_lines[i] = f"_libtpu_version = '{version}'\n"
    elif re.match(r'_libtpu_date\s*=', line):
      found_libtpu_date = True
      setup_lines[i] = f"_libtpu_date = '{date}'\n"
    else:
      m = re.match(r'(\s+)_libtpu_wheel_name\s*=.*nightly', line)
      if m:
        found_libtpu_wheel = True
        indent = m.group(1)
        setup_lines[i] = (
            indent +
            "_libtpu_wheel_name = f'libtpu-{_libtpu_version}.dev{_libtpu_date}+nightly-"
            + suffix + "_{platform_machine}'\n")

  if not found_libtpu_version:
    logger.error('Could not find _libtpu_version in setup.py.')
  if not found_libtpu_date:
    logger.error('Could not find _libtpu_date in setup.py.')
  if not found_libtpu_wheel:
    logger.error('Could not find _libtpu_wheel_name in setup.py.')

  with open(_SETUP_PATH, 'w') as f:
    f.writelines(setup_lines)

  success = found_libtpu_version and found_libtpu_date and found_libtpu_wheel
  if success:
    logger.info('Updated the libtpu version in setup.py.')
  return success


def update_jax() -> bool:
  """Updates the jax/jaxlib versions in setup.py to the latest nightly build.
  
  Returns:
    True if the setup.py file was updated, False otherwise.
  """

  result = find_latest_jax_nightly()
  if not result:
    return False

  jax_version, jaxlib_version, date = result

  with open(_SETUP_PATH, 'r') as f:
    setup_lines = f.readlines()

  # Update the lines for specifying jax/jaxlib versions.
  found_jax_version, found_jaxlib_version, found_jax_date = False, False, False
  for i, line in enumerate(setup_lines):
    if re.match(r'_jax_version\s*=', line):
      found_jax_version = True
      setup_lines[i] = f"_jax_version = '{jax_version}'\n"
    elif re.match(r'_jaxlib_version\s*=', line):
      found_jaxlib_version = True
      setup_lines[i] = f"_jaxlib_version = '{jaxlib_version}'\n"
    elif re.match(r'_jax_date\s*=', line):
      found_jax_date = True
      setup_lines[i] = f"_jax_date = '{date}'  # Date for jax and jaxlib.\n"

  if not found_jax_version:
    logger.error('Could not find _jax_version in setup.py.')
  if not found_jaxlib_version:
    logger.error('Could not find _jaxlib_version in setup.py.')
  if not found_jax_date:
    logger.error('Could not find _jax_date in setup.py.')

  with open(_SETUP_PATH, 'w') as f:
    f.writelines(setup_lines)

  success = found_jax_version and found_jaxlib_version and found_jax_date
  if success:
    logger.info('Updated the jax/jaxlib versions in setup.py.')
  return success


def main() -> None:
  logging.basicConfig(level=logging.INFO)

  openxla_updated = update_openxla()
  libtpu_updated = update_libtpu()
  jax_updated = update_jax()

  if not (openxla_updated and libtpu_updated and jax_updated):
    sys.exit(1)


if __name__ == '__main__':
  main()

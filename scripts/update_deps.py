#!/usr/bin/env python3
"""Updates third party dependencies.

Usage:

  scripts/update_deps.py
  scripts/update_deps.py --use_latest

    By default, updates to the latest stable JAX release and its corresponding
    OpenXLA and libtpu versions.

    With --use_latest, updates to the latest nightly builds of OpenXLA,
    libtpu, and JAX.
"""

import argparse
import json
import logging
import os
import platform
import re
import sys
import urllib.request
from html.parser import HTMLParser

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
# New JAX package index URLs (PEP 503 compliant)
_JAX_INDEX_URL = 'https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/'
_JAX_PROJECT_URL = _JAX_INDEX_URL + 'jax/'
_JAXLIB_PROJECT_URL = _JAX_INDEX_URL + 'jaxlib/'

_TORCH_COMMIT_FORMAT = "# %cs%n%H"
_TORCH_COMMIT_FILE = os.path.join(_PTXLA_DIR, ".torch_commit")


class PEP503Parser(HTMLParser):
  """Parser for PEP 503 simple repository API pages.

  This parser extracts all links and their link text from the
  HTML content of a PEP 503 index page.
  """

  links: list[tuple[str, str]]
  """List of (href, text) tuples for all links found."""

  _current_link: str | None
  """The current link being processed."""

  _current_text: str
  """The text content of the current link being processed."""

  def __init__(self):
    super().__init__()
    self.links = []
    self._current_link = None
    self._current_text = ""

  def handle_starttag(self, tag: str, attrs: list[tuple[str,
                                                        str | None]]) -> None:
    """Handles the start of an HTML tag.

    Starts processing a link if the tag is an anchor (<a>).
    """
    if tag == 'a':
      href = None
      for attr, value in attrs:
        if attr == 'href':
          href = value
          break
      if href:
        self._current_link = href
        self._current_text = ""

  def handle_data(self, data: str) -> None:
    """Handles the text data within an HTML tag.

    If currently processing a link, appends the data to the current text.
    """
    if self._current_link:
      self._current_text += data

  def handle_endtag(self, tag: str) -> None:
    """Handles the end of an HTML tag.

    If the tag is an anchor (<a>), adds the link and its text to the list.
    """
    if tag == 'a' and self._current_link:
      self.links.append((self._current_link, self._current_text.strip()))
      self._current_link = None
      self._current_text = ""


def clean_tmp_dir() -> None:
  """Cleans up the temp dir."""

  os.system(f'rm -rf {_TMP_DIR}')
  os.system(f'mkdir -p {_TMP_DIR}')


def get_xla_commit_and_date(commit: str | None = None) -> tuple[str, str]:
  """Find a date, commit pair from https://github.com/openxla/xla.
  If commit is specified, use that commit.
  If no commit is specified take the latest commit from main.

  Returns:
    A tuple of the commit SHA and its date (YYYY-MM-DD).
  """

  clean_tmp_dir()
  if commit is None:
    # Clone the repository to a depth of 1 (just the main branch).
    os.system(
        f'git clone --depth=1 https://github.com/openxla/xla {_TMP_DIR}/xla')
    commit = os.popen(f'cd {_TMP_DIR}/xla && git rev-parse HEAD').read().strip()
    date = os.popen(
        f'cd {_TMP_DIR}/xla && git show -s --format=%cd --date=short {commit}'
    ).read().strip()
    logger.info(f'Found latest XLA commit {commit} on date {date}')
  else:
    # Clone the repository history, but no blobs to save space.
    os.system(
        f'git clone --bare --filter=blob:none https://github.com/openxla/xla.git {_TMP_DIR}/xla.git'
    )
    date = os.popen(
        f'git --git-dir={_TMP_DIR}/xla.git show -s --format=%cd --date=short {commit}'
    ).read().strip()
    if not date:
      logging.error(f"Unable to local XLA commit {commit}")
    logger.info(f'Given XLA commit {commit}, determined date {date}')

  return commit, date


def get_latest_stable_jax_info() -> tuple[str, str, str] | None:
  """Gets info about the latest stable JAX release from GitHub.

  Returns:
    A tuple of (JAX version, JAX release date, XLA commit hash).
  """
  url = 'https://api.github.com/repos/google/jax/releases/latest'
  try:
    with urllib.request.urlopen(url) as response:
      data = json.loads(response.read().decode())
  except Exception as e:
    logger.error(f'Failed to fetch {url}: {e}')
    return None

  tag_name = data['tag_name']  # e.g., "jax-v0.4.28"
  jax_version = tag_name.replace('jax-v', '')  # e.g., "0.4.28"

  published_at = data['published_at']  # e.g., "2024-04-26T22:58:34Z"
  release_date = published_at.split('T')[0]  # e.g., "2024-04-26"

  # The XLA commit is in third_party/xla/revision.bzl in the JAX repo.
  workspace_bzl_url = f'https://raw.githubusercontent.com/google/jax/{tag_name}/third_party/xla/revision.bzl'
  try:
    with urllib.request.urlopen(workspace_bzl_url) as response:
      workspace_content = response.read().decode()
  except Exception as e:
    logger.error(f'Failed to fetch {workspace_bzl_url}: {e}')
    return None

  match = re.search(r'XLA_COMMIT = "([a-f0-9]{40})"', workspace_content)
  if not match:
    logger.error(f'Could not find XLA_COMMIT in {workspace_bzl_url}.')
    return None
  xla_commit = match.group(1)

  return jax_version, release_date, xla_commit


def update_openxla(commit: str | None = None) -> bool:
  """Updates the OpenXLA version in the WORKSPACE file.

  Returns:
    True if the WORKSPACE file was updated, False otherwise.
  """
  commit, date = get_xla_commit_and_date(commit=commit)

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


def find_latest_nightly(
    html_lines: list[str],
    build_re: str,
    target_date: str | None = None) -> tuple[str, str, str] | None:
  """Finds the latest nightly build from the list of HTML lines.

  Args:
    html_lines: A list of HTML lines to search for the nightly build.
    build_re: A regular expression for matching the nightly build line.
      It must have 3 capture groups: the version, the date, and the name suffix.
    target_date: If specified, find the latest build on or before this date
      (YYYYMMDD). Otherwise, find the latest build overall.

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
      if target_date is None:
        if date > latest_date:
          latest_version, latest_date, latest_suffix = version, date, suffix
      elif date <= target_date and date > latest_date:
        latest_version, latest_date, latest_suffix = version, date, suffix

  if found_build:
    return latest_version, latest_date, latest_suffix

  return None


def find_libtpu_build(
    target_date: str | None = None) -> tuple[str, str, str] | None:
  """Finds a libtpu nightly build for the current platform.

  Args:
    target_date: If specified, find build for this date. Otherwise, find latest.

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
      _PLATFORM + r'\.whl</a>', target_date)


def fetch_pep503_page(url: str) -> list[tuple[str, str]]:
  """Fetches and parses a PEP 503 index page.

  Args:
    url: The URL of the PEP 503 index page.

  Returns:
    A list of (href, text) tuples for all links on the page.
  """
  try:
    with urllib.request.urlopen(url) as response:
      html = response.read().decode('utf-8')

    parser = PEP503Parser()
    parser.feed(html)
    return parser.links
  except Exception as e:
    logger.error(f'Failed to fetch {url}: {e}')
    return []


def find_latest_jax_nightly() -> tuple[str, str, str] | None:
  """Finds the latest JAX nightly build using the new package index.

  Returns:
    A tuple of the jax version, jaxlib version, and date of the latest JAX nightly build,
    or None if no build is found.
  """

  def parse_version_date(url: str, pattern: str) -> list[tuple[str, str]]:
    links = fetch_pep503_page(url)
    if not links:
      logger.error(f'Could not fetch packages from {url}')
      return []
    compiled = re.compile(pattern)
    results = []
    for href, text in links:
      filename = text if text else href.split('/')[-1].split('#')[0]
      m = compiled.match(filename)
      if m:
        version, date = m.groups()
        results.append((version, date))
    return results

  # Find JAX libraries.
  #
  # Look for patterns like: jax-0.6.1.dev20250428-py3-none-any.whl
  #   Group 1: Represents the JAX version (formatted as a series of digits and dots).
  #   Group 2: Represents the build date (an 8-digit string typically in YYYYMMDD format).
  jax_versions_dates = parse_version_date(
      _JAX_PROJECT_URL, r'jax-([\d.]+)\.dev(\d{8})-py3-none-any\.whl')
  if not jax_versions_dates:
    logger.error(f"Could not fetch JAX packages from {_JAX_PROJECT_URL}")
    return None

  # Fetch jaxlib libraries
  #
  # Look for patterns like: jaxlib-0.6.1.dev20250428-cp310-cp310-manylinux2014_x86_64.whl
  #   Group 1: Represents the jaxlib version (formatted as a series of digits and dots).
  #   Group 2: Represents the build date (an 8-digit string typically in YYYYMMDD format).
  jaxlib_versions_dates = parse_version_date(
      _JAXLIB_PROJECT_URL, r'jaxlib-([\d.]+)\.dev(\d{8})-.*\.whl')
  if not jaxlib_versions_dates:
    logger.error(f"Could not fetch jaxlib packages from {_JAXLIB_PROJECT_URL}")
    return None

  latest_jax_version = ''
  latest_jax_date = ''
  for version, date in jax_versions_dates:
    if date > latest_jax_date:
      latest_jax_version = version
      latest_jax_date = date

  if not latest_jax_version:
    logger.error(
        f'Could not find any JAX nightly builds. Tried parsing {_JAX_PROJECT_URL}'
    )
    return None

  latest_jaxlib_version = ''
  for version, date in jaxlib_versions_dates:
    # Only consider jaxlib builds from the same date as JAX
    if date == latest_jax_date and version > latest_jaxlib_version:
      latest_jaxlib_version = version

  if not latest_jaxlib_version:
    logger.error(
        f'Could not find jaxlib nightly build for date {latest_jax_date}. Tried parsing {_JAXLIB_PROJECT_URL}'
    )
    return None

  logger.info(
      f'Found JAX {latest_jax_version} and jaxlib {latest_jaxlib_version} from {latest_jax_date}'
  )
  return latest_jax_version, latest_jaxlib_version, latest_jax_date


def update_libtpu(target_date: str | None = None) -> bool:
  """Updates the libtpu version in setup.py.

  Returns:
    True if the setup.py file was updated, False otherwise.
  """

  result = find_libtpu_build(target_date)
  if not result:
    if target_date:
      logger.error(f'Could not find libtpu build for date {target_date}.')
    else:
      logger.error('Could not find latest libtpu nightly build.')
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


def update_jax(use_latest: bool) -> bool:
  """Updates the jax/jaxlib versions in setup.py.

  Returns:
    True if the setup.py file was updated, False otherwise.
  """
  if use_latest:
    result = find_latest_jax_nightly()
    if not result:
      return False
    jax_version, jaxlib_version, date = result
  else:
    jax_info = get_latest_stable_jax_info()
    if not jax_info:
      return False
    jax_version, release_date, _ = jax_info
    jaxlib_version = jax_version
    date = release_date.replace('-', '')

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


def update_pytorch(use_latest: bool) -> bool:
  clean_tmp_dir()

  torch_temp_dir = os.path.join(_TMP_DIR, "pytorch")
  branch = "main" if use_latest else "viable/strict"

  cmd_clone = [
      "git",
      "clone",
      "--branch",
      branch,
      "--depth=1",
      "https://github.com/pytorch/pytorch",
      torch_temp_dir,
  ]
  os.system(" ".join(cmd_clone))

  cmd_commit_show = [
      "git",
      f"--git-dir={torch_temp_dir}/.git",
      "show",
      "--no-patch",
      f"--pretty=format:\"{_TORCH_COMMIT_FORMAT}\"",
  ]
  commit = os.popen(" ".join(cmd_commit_show)).read().strip()

  with open(_TORCH_COMMIT_FILE, "w") as f:
    f.write(commit)

  return True


def main() -> None:
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser(
      description="Updates third party dependencies.")
  parser.add_argument(
      '--use_latest',
      action='store_true',
      default=False,
      help='Update to latest nightly versions instead of latest stable versions.'
  )
  args = parser.parse_args()

  if args.use_latest:
    logger.info('Updating to latest nightly versions...')
    openxla_updated = update_openxla()
    libtpu_updated = update_libtpu()
    jax_updated = update_jax(use_latest=True)
    pytorch_updated = update_pytorch(use_latest=True)
    if not (openxla_updated and libtpu_updated and jax_updated and pytorch_updated):
      sys.exit(1)
  else:
    logger.info('Updating to latest stable versions...')
    jax_info = get_latest_stable_jax_info()
    if not jax_info:
      sys.exit(1)

    jax_version, jax_release_date, xla_commit = jax_info
    logger.info(
        f'Found latest stable JAX release {jax_version} from {jax_release_date}, with XLA commit {xla_commit}'
    )

    openxla_updated = update_openxla(xla_commit)
    libtpu_updated = update_libtpu(
        target_date=jax_release_date.replace('-', ''))
    jax_updated = update_jax(use_latest=False)

    pytorch_updated = update_pytorch(use_latest=False)

    if not (openxla_updated and libtpu_updated and jax_updated and pytorch_updated):
      sys.exit(1)


if __name__ == '__main__':
  main()

#!/usr/bin/env python3
"""Updates the local git repos to match upstream repos.

For convenience, you can set up a `git sync` alias to run this script by:

  git config alias.sync '!scripts/git_sync.py'

(you only need to do this once).
"""

import argparse
import logging
import os
import platform
import re
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Names of the repos.
_PYTORCH_REPO = 'pytorch'
_VISION_REPO = 'vision'
_TORCH_XLA_REPO = 'torch_xla'

# Roots of the local repos.
_PTXLA_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')
_PYTORCH_DIR = os.path.abspath(_PTXLA_DIR + '/..')
_VISION_DIR = os.path.abspath(_PYTORCH_DIR + '/../vision')

# Path of this script, relative to the torch_xla repo root.
_SCRIPT_REL_PATH = os.path.abspath(__file__)[len(_PTXLA_DIR) + 1:]


def sync_repo(repo: str) -> bool:
  """Updates the default branch of the local pytorch repo to match upstream.
  
  Args:
    repo: The repo to sync. Must be one of "pytorch", "vision", and "torch_xla".

  Returns:
    True if the local pytorch repo was successfully updated, False otherwise.
  """

  logger.info(f'Syncing the {repo} repo...')

  # Change to the repo root directory.
  os.chdir(_PYTORCH_DIR if repo == _PYTORCH_REPO else _VISION_DIR if repo ==
           _VISION_REPO else _PTXLA_DIR)

  # If there are uncommited changes, ask the user to commit/stash them and
  # retry.
  status = os.system('git status --porcelain')
  if status != 0:
    logger.error(
        f'The local {repo} repo has uncommited changes. Please commit or stash them and retry.'
    )
    return False

  # If there are untracked files, ask the user to commit/stash them and retry.
  untracked_files = os.popen(
      'git ls-files --others --exclude-standard').read().strip().split('\n')
  if untracked_files:
    logger.error(
        f'The local {repo} repo has untracked files ({", ".join(sorted(untracked_files))}). Do you want to commit/stash them and retry?'
    )
    return False

  # Check out the default branch.
  default_branch = 'master' if repo == _TORCH_XLA_REPO else 'main'
  if os.system(f'git checkout {default_branch}') != 0:
    logger.error(
        f'Failed to checkout the {default_branch} branch of the {repo} repo.')
    return False

  # Make sure the remotes are set up correctly:
  #
  # - There must be an "origin" remote, pointing to where the local repo is
  #   cloned from. This can be either the official repo but a fork made by
  #   the user.
  # - If the user created the local repo by cloning their own fork, there
  #   must also be an "upstream" remote pointing to the official repo. In
  #   this case, "origin" would point to the user's fork.
  remotes = os.popen('git remote').read().strip().split('\n')
  if 'origin' not in remotes:
    logger.error(
        f'No remote named "origin" found for the local {repo} repo. Please add one and retry.'
    )
    return False

  # Which remote is the official repo?
  official_repo_remote = 'upstream' if 'upstream' in remotes else 'origin'

  # Pull the latest changes from the official repo.
  if os.system(f'git pull {official_repo_remote} {default_branch}') != 0:
    logger.error(
        f'Failed to pull the latest changes from the "{official_repo_remote}" remote of the {repo} repo.'
    )
    return False

  # If the user used a fork to create this local repo, push the changes to
  # the fork to make it in sync with the official repo.
  if official_repo_remote != 'origin':
    logger.info(
        f'Pushing the changes to the "origin" remote of the {repo} repo. Please touch the security key when prompted...'
    )
    if os.system(f'git push origin {default_branch}') != 0:
      logger.error(
          'Failed to push the changes to the "origin" remote of the {repo} repo.'
      )
      return False

  logger.info(
      f'Successfully updated the {default_branch} branch of the {repo} repo to match upstream.'
  )
  return True


def main() -> None:
  logging.basicConfig(level=logging.INFO)

  # Abort if the user is root.
  if os.geteuid() == 0:
    logger.error(
        'Please do not run this script as root (e.g. inside the dev container), as it will mess up the git permissions. Run it outside of the dev container instead.'
    )
    sys.exit(1)

  arg_parser = argparse.ArgumentParser(
      prog=_SCRIPT_REL_PATH, description=__doc__)
  arg_parser.add_argument(
      '--base_repo',
      '-b',
      type=str,
      choices=(_PYTORCH_REPO, _VISION_REPO, _TORCH_XLA_REPO),
      default=_TORCH_XLA_REPO,
      help='sync the given repo and all repos that depend on it; the default is torch_xla',
  )

  args = arg_parser.parse_args()

  failed = False
  if args.base_repo == _PYTORCH_REPO:
    if not sync_repo(_PYTORCH_REPO):
      failed = True

  if args.base_repo in (_PYTORCH_REPO, _VISION_REPO):
    if os.path.isdir(_VISION_DIR) and not sync_repo(_VISION_REPO):
      failed = True

  if not sync_repo(_TORCH_XLA_REPO):
    failed = True

  if failed:
    logger.error('Failed to sync some repos.')
    sys.exit(1)

  logger.info('All repos synced successfully.')


if __name__ == '__main__':
  main()

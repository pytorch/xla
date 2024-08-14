#!/usr/bin/env bash
set -ex

NIGHTLY_VERSION=$(date '+%Y%m%d%H%M')
VERSION_UPDATE_PATTERN="s/^__version__\s*=\s*\"([^\"]+)\"/__version__ = \"\1.dev$NIGHTLY_VERSION\"/g;"

sed -r "$VERSION_UPDATE_PATTERN" torch_xla2/__init__.py --in-place
hatch build -t wheel

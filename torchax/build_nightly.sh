#!/usr/bin/env bash
set -ex

NIGHTLY_VERSION=$(date '+%Y%m%d%H%M')

# Update the version to <version>.devYYYYMMDDHHMM in __init__.py
VERSION_UPDATE_PATTERN="s/^__version__\s*=\s*\"([^\"]+)\"/__version__ = \"\1.dev$NIGHTLY_VERSION\"/g;"
sed -r "$VERSION_UPDATE_PATTERN" torchax/__init__.py --in-place

hatch build -t wheel

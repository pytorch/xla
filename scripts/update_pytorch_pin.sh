#! /bin/bash

# This script updates the PyTorch pin file: .torch_commit with the current
# commit of the local PyTorch repository. Before running, make sure you have
# checked out to the right branch (usually `viable/strict`).

set -e

SCRIPTS_DIR=$(dirname $(realpath "$0"))
PYTORCH_XLA_DIR=$(dirname "$SCRIPTS_DIR")
PYTORCH_DIR=$(dirname "$PYTORCH_XLA_DIR")
TORCH_COMMIT_FILE="$PYTORCH_XLA_DIR/.torch_commit"

FORMAT="# commit %H%n# Author: %an <%ae>%n# Author Date:    %ad%n# Committer Date: %cd%n#%n#    %s%n#%n%H"

cd "$PYTORCH_DIR"

BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "viable/strict" ]]; then
    echo "WARNING: updating PyTorch to a branch different than 'viable/strict': $BRANCH."
fi

git show --no-patch --pretty=format:"$FORMAT" > "$TORCH_COMMIT_FILE"

echo "INFO: wrote to: $TORCH_COMMIT_FILE"
echo ""
cat "$TORCH_COMMIT_FILE"

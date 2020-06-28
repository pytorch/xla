#!/usr/bin/env bash

set -ex

cd /tmp/pytorch/xla

source ./xla_env

echo "Building docs"
pushd docs
./docs_build.sh
popd

echo "Pushing to public"
git config --global user.email "torchxla@gmail.com"
git config --global user.name "torchxlabot"
GH_PAGES_BRANCH=gh-pages
GH_PAGES_DIR=gh-pages-tmp
CURRENT_COMMIT=`git rev-parse HEAD`
BRANCH_NAME=`git rev-parse --abbrev-ref HEAD`
if [[ "$BRANCH_NAME" == release/* ]]; then
  SUBDIR_NAME=$BRANCH_NAME
else
  SUBDIR_NAME="master"
fi
pushd /tmp
git clone --quiet -b "$GH_PAGES_BRANCH" https://github.com/pytorch/xla.git "$GH_PAGES_DIR"
pushd $GH_PAGES_DIR
rm -rf $SUBDIR_NAME
mkdir -p $SUBDIR_NAME
cp -fR /tmp/pytorch/xla/docs/build/* $SUBDIR_NAME
git_status=$(git status --porcelain)
if [[ $git_status ]]; then
  echo "Doc is updated... Pushing to public"
  echo "${git_status}"
  sudo apt-get -qq update
  sudo apt-get -qq install expect
  git add .

  COMMIT_MSG="Update doc from commit $CURRENT_COMMIT"
  git commit -m "$COMMIT_MSG"
  set +x
/usr/bin/expect <<DONE
spawn git push origin "$GH_PAGES_BRANCH"
expect "Username*"
send "torchxlabot\n"
expect "Password*"
send "$::env(GITHUB_TORCH_XLA_BOT_TOKEN)\n"
expect eof
DONE
  set -x
else
  echo "Nothing changed in documentation."
fi
popd
popd



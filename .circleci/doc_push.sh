#!/bin/bash

set -ex

cd /tmp/pytorch/xla

source ./xla_env

echo "Building docs"
pushd docs
./docs_build.sh
popd

echo "Pushing to public"
git config --global user.email "circleci.ossci@gmail.com"
git config --global user.name "CircleCI"
GH_PAGES_BRANCH=gh-pages
GH_PAGES_DIR=gh-pages-tmp
pushd /tmp
git clone --quiet -b "$GH_PAGES_BRANCH" https://github.com/pytorch/xla.git "$GH_PAGES_DIR"
pushd $GH_PAGES_DIR
cp -fR /tmp/pytorch/xla/docs/build/* .
git_status=$(git status --porcelain)
if [[ $git_status ]]; then
  echo "Doc is updated... Pushing to public"
  echo "${git_status}"
  git add .
  
  CURRENT_COMMIT=`git rev-parse HEAD`
  COMMIT_MSG="Update doc from commit $CURRENT_COMMIT"
  git commit -m "$COMMIT_MSG"
  git push origin "$GH_PAGES_BRANCH" 
else
  echo "Nothing changed in documentation."
fi
popd
popd



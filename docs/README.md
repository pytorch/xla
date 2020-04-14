## Publish documentation for a new release.

CircleCI job `pytorch_xla_linux_debian11_and_push_doc` is specified to run on `release/*` branches, but it was not
run on release branches due to "Only build pull requests" setting. Turning off "Only build pull requests" will result
in much larger volumes in jobs which is often unnecessary. We're waiting for [this feature request](https://ideas.circleci.com/ideas/CCI-I-215)
to be implemented so that we could override this setting on some branches.

Before the feature is available on CircleCi side, we'll use a manual process to publish documentation for release.
[Documentation for master branch](http://pytorch.org/xla/master/) is still updated automatically by the CircleCI job.
But we'll need to manually commit the new versioned doc and point http://pytorch.org/xla to the documentation of new
stable release.

Take 1.5 release as example:
```
# Build pytorch/pytorch:release/1.5 and pytorch/xla:release/1.5 respectively.
# In pytorch/xla/docs
./docs_build.sh
git clone -b gh-pages https://github.com/pytorch/xla.git /tmp/xla
cp -r build/* /tmp/xla/release/1.5
cd /tmp/xla
# Update `redirect_url` in index.md
git add .
git commit -m "Publish 1.5 documentation."
git push origin gh-pages
```

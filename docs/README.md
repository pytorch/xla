## Publish documentation for a new release.

CI job `pytorch_xla_linux_debian11_and_push_doc` is specified to run on `release/*` branches, but it was not
run on release branches due to "Only build pull requests" setting. Turning off "Only build pull requests" will result
in much larger volumes in jobs which is often unnecessary. We're waiting for [this feature request](https://ideas.circleci.com/ideas/CCI-I-215)
to be implemented so that we could override this setting on some branches.

Before the feature is available on CircleCi side, we'll use a manual process to publish documentation for release.
[Documentation for master branch](http://pytorch.org/xla/master/) is still updated automatically by the CI job.
But we'll need to manually commit the new versioned doc and point http://pytorch.org/xla to the documentation of new
stable release.

Take 2.3 release as example:
```
# Build pytorch/pytorch:release/2.3 and pytorch/xla:release/2.3 respectively.
# In pytorch/xla/docs
./docs_build.sh
git clone -b gh-pages https://github.com/pytorch/xla.git /tmp/xla
cp -r build/* /tmp/xla/release/2.3
cd /tmp/xla
# Update `redirect_url` in index.md
git add .
git commit -m "Publish 2.3 documentation."
git push origin gh-pages
```
## Adding new docs

To add a new doc please create a `.md` file under this directory. To make this doc show up in our [doc pages](https://pytorch.org/xla/master/index.html), please add a `.rst` file under `source/`.

## Adding images in the doc
Please add your imges to both `asserts` and `source/assets/` for images to properlly show up in the markdown files as well as our [doc pages](https://pytorch.org/xla/master/index.html).

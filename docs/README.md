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
Please add your imges to both `_static/img/` and `source/_static/img/` for images to properlly show up in the markdown files as well as our [doc pages](https://pytorch.org/xla/master/index.html).

## Runnable tutorials

Stylistically, review the [pytorch tutorial contributing guide](https://github.com/pytorch/tutorials/blob/main/CONTRIBUTING.md).

* Save your tutorial as `*_tutorial.py`.
* Follow the formatting so that the tooling can automatically convert to a jupyter notebook

We do not yet have an automated build system for runnable tutorials that matches 
PyTorch. For now, include manual instructions and check-in the output ipynb and MD files. 


Note that the existing .gitattributes in this directory will prevent diffs and merges
on the ipynb, which is illegible json. 

TODO: Automate the docs build system for runnable tutorials. 

Add your runnable tutorial to the list below with build instructions, necessary environments,
and steps to manually verify correctness.

### source/tutorials/precision_tutorial.py

Run on a TPU machine. 

One time installs not in requirements.txt

conda install -c conda-forge pandoc

Run every time: 


py2nb source/tutorials/precision_tutorial.py
jupyter nbconvert --to notebook --execute --inplace source/tutorials/precision_tutorial.ipynb 
# Ignore SIGTERM
# Manually verify precision_tutorial.ipynb that the final line shows a delta, e.g.
./docs_build.sh
# Download build/ to your local machine and visually inspect the precision_tutorial.html in a browser. 
# Look for issues like headers that didn't render and mathjax that didn't render. 
# Check that the final code cell shows different numbers:
```
    Z_ref: FORMAT:0b SIGN:0 EXPONENT:01111111 MANTISSA:01110000101000111101100 VALUE=1.440000057220459
    Z:     FORMAT:0b SIGN:0 EXPONENT:01111111 MANTISSA:01110000101000111101101 VALUE=1.4400001764297485
```

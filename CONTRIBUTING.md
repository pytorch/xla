# Contribute To PyTorch/XLA

We appreciate all contributions. If you are planning to contribute a bug fix for
an open issue, please comment on the thread and we're happy to provide guidance.
You are welcome to pick issues with [good first issue](https://github.com/pytorch/xla/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
and [help wanted](https://github.com/pytorch/xla/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
labels to get started.

If you plan to contribute new features or extensions to this repository, first
open an issue and discuss the feature with us. Sending a PR without discussion
might result in a rejected PR, because we might be taking the repository in a
different direction.

## Setting up the Workspace

Please follow the steps below in order:

### Prerequisite

To work on PyTorch/XLA, you'll need a powerful Linux machine with plenty of
CPUs and RAM. Make sure you have `git` and `docker` installed on this machine.
If you don't, follow https://github.com/git-guides/install-git and
https://docs.docker.com/engine/install/ to install them.

### Forking the Git Repos

In order to create PRs later, we need to first fork the Git repos we'll be
working with:

1.  Go to https://github.com/pytorch/pytorch and fork it as `pytorch`.
1.  Go to https://github.com/pytorch/vision and fork it as `vision`.
1.  Go to https://github.com/pytorch/xla and fork it as `pytorch-xla`. Note
    the change of project name: we want to avoid confusion with the OpenXLA
    project that PyTorch/XLA depends on, which is also named `xla`.

### Cloning the Forked Repos

Next, we need to clone the forked repos locally so that we can make changes.

On your Linuc machine, decide a directory as your workspace. Make sure that
this directory and all of its ancestors are publically readable. Then run
the following commands on this machine:

```bash
# Make sure that all new files and directories are publically readable.
# Otherwise you may have permission issues when building the code in bazel's
# sandbox mode.
umask 022

# Create the workspace directory if you haven't.
export WORKSPACE_DIR=<absolute-path-to-your-workspace>
mkdir -p $WORKSPACE_DIR

# Clone the repos.
cd $WORKSPACE_DIR
git clone --recursive git@github.com:<your-github-user-name>/pytorch.git
git clone --recursive git@github.com:<your-github-user-name>/vision.git
git clone --recursive git@github.com:<your-github-user-name>/pytorch-xla.git pytorch/xla
```

### Setting up Remote Tracking

From time to time, we'll need to bring our forked repos up to date with the
official (aka, upstream) repos. Therefore we'll need to tell Git where to
find these upstream repos. We only need to do this once:

```bash
# Set up remote tracking for pytorch.
cd $WORKSPACE_DIR/pytorch
git remote add upstream https://github.com/pytorch/pytorch.git
# Set up remote tracking for vision.
cd $WORKSPACE_DIR
git remote add upstream https://github.com/pytorch/vision.git
# Set up remote tracking for pytorch/xla.
cd $WORKSPACE_DIR/pytorch/xla
git remote add upstream https://github.com/pytorch/xla.git
```

### Setting VSCode Configurations

```bash
cd $WORKSPACE_DIR
ln -s pytorch/xla/.devcontainer/ .devcontainer
ln -s pytorch/xla/contrib/vscode/ .vscode
ln -s pytorch/xla/.style.yapf .style.yapf
ln -s pytorch/xla/.clang-format .clang-format
```

## Building from Source

We recommend you use our prebuilt Docker image to start your development work
using either VS Code or a local container:

### Visual Studio Code Dev Container

WARNING: DO NOT run `git` commands that may change the repo's state inside
the container. Doing so will mess up the permission of Git's internal files
as you run as `root` inside the container. Instead, run all mutating `git`
commands on your Linux machine directly, outside of the container.

1.  Start VS Code and ensure you have the [`Remote Development` Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
    installed. It includes the [`Remote - SSH`](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) and
    [`Dev Containers`](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
    extensions.

1.  From VS Code, connect to your remote Linux machine and open your workspace
    directory:

    1. New Window > Connect to... > Connect to Host ... > type the remote
       address.
    1. Open... > select the workspace directory on your remote machine.
    1. When asked "Do you trust the authors of the files in this folder?",
       click on "Yes, I trust the authors".
    1. When asked if you want to reopen in dev container, click on "Yes".
       If you are not prompted to reopen in a container, in the VS Code command
       pallete, type `Dev Containers: Reopen in Container` to open your
       workspace in one of our pre-built Docker containers.
    1. Select the correct container based on the accellarators on your machine.
       Use `tpu-contributor` if you are unsure of which to use.
       If you're a Googler, use `tpu-internal`, which is set up for
       [bazel remote build caching](https://github.com/pytorch/xla/blob/master/docs/source/contribute/bazel.md#remote-caching)
       for faster builds.

1.  Make sure VSCode discovers the `pytorch/xla` repo so that diff highlighting
    works (by default VSCode cannot discover it as it's nested inside the
    `pytorch` repo):

    1. Go to File > Add Folder to Workspace..., and add the `pytorch/xla`
       folder.
    1. In the repository list, you should now see 3 repos: `xla` (for
       `pytorch/xla`), `pytorch`, and `vision`.

1.  Open a new terminal window in VS Code. Since you are running as root in this
    container, mark the repository directories as safe. The commands below assume
    your workspace directory is `torch`, update the commands to use your workspace
    directory.

    ```bash
    git config --global --add safe.directory /workspaces/torch/pytorch
    git config --global --add safe.directory /workspaces/torch/pytorch/xla
    git config --global --add safe.directory /workspaces/torch/vision
    ```

1.  In the terminal window, run the following commands to build PyTorch,
    TorchVision, and PyTorch/XLA:

    ```bash
    # Uninstall any existing torch torch-xla torchvision installation
    # Run multiple times if needed
    pip uninstall torch torch-xla torchvision libtpu-nightly
    # pytorch/xla requires pytorch wheel to be presented under pytorch/dist
    cd pytorch
    python setup.py bdist_wheel
    python setup.py install
    cd ../vision
    python setup.py develop
    cd ../pytorch/xla
    python setup.py develop
    # Optional: if you're using TPU, install libtpu
    pip install torch_xla[tpu] \
      -f https://storage.googleapis.com/libtpu-wheels/index.html \
      -f https://storage.googleapis.com/libtpu-releases/index.html
    ```

1.  If you are running on a TPU VM, ensure `torch` and `torch_xla` were built and
    installed correctly:

    ```bash
    python -c 'import torch_xla as xla; print(xla.device())'
    # Output: xla:0
    ```

1.  Set up `clangd` so that C++ code completion/navigation works:

    1. Install `clangd`: open any C++ source file in VS Code to trigger a
       prompt to install `clangd` in the dev container. Accept the request.
       Restart VS Code for the change to take effect.

    1. Generate the compilation database so that `clangd` knows how to compile
       the C++ files:

       ```bash
       # Run this from a terminal in VS Code, in the pytorch/xla directory
       # of the workspace.
       scripts/update_compile_commands.py
       ```

       This should create the `build/compile_commands.json` file, which
       describes how each C++ source file is compiled. The script may take
       several minutes the first time. You may need to rerun the script
       if build rules or file structures have changed. However, subsequent
       runs are usually much faster.

**Subsequent builds**: after building the packages from source code for the
first time, you may need to build everything again, for example, after a
`git pull`. You can run `scripts/build_developer.sh` which will rebuild PyTorch,
TorchVision, and PyTorch/XLA.

### Manually build in Docker container

* Setup Development Docker Image

  ```shell
  docker pull us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:tpu
  docker run --privileged --network=host --name ptxla -it -d -e "TERM=xterm-256color" us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:tpu
  docker exec --privileged -it ptxla /bin/bash
  ```
  All of the code below will be assumed to be run within the docker.

* Clone the _PyTorch_ repo as per [instructions](https://github.com/pytorch/pytorch#from-source).

  ```Shell
  git clone --recursive https://github.com/pytorch/pytorch
  cd pytorch/
  ```

* Clone the _PyTorch/XLA_ repo:

  ```Shell
  git clone --recursive https://github.com/pytorch/xla.git
  ```

* Build PyTorch
  ```Shell
  # pytorch/xla requires pytorch wheel to be presented under pytorch/dist
  python setup.py bdist_wheel
  python setup.py develop
  ```
* Build PyTorch/XLA
  ```Shell
  cd xla/
  python setup.py develop
  ```

### Additional steps for GPU

Please refer to this [guide](https://github.com/pytorch/xla/blob/master/plugins/cuda/README.md).

## Before Creating a Pull Request

In `pytorch/xla` repo we enforce coding style for both C++ and Python files. Please try to format
your code before creating a pull request.

### C++ Style Guide

`pytorch/xla` uses `clang-format-11` with a customized style config.
If your PR touches the C++ source files, please run the following command before submitting a PR.

```Shell
# How to install: sudo apt install clang-format-11
# If your PR only changes foo.cpp, run the following in xla/ folder
clang-format-11 -i -style=file /PATH/TO/foo.cpp
# To format all cpp files, run the following in xla/ folder
find -name '*.cpp' -o -name '*.h' -o -name '*.cc' | xargs clang-format-11 -i -style=file
```

### Python Style Guide

`pytorch/xla` uses `yapf`(specially version 0.40.2 in case it's not backward compatible) with a customized style config.
If your PR touches the Python source files, please run the following command before submitting a PR.

```Shell
# How to install: pip install yapf==0.40.2
yapf --recursive -i *.py test/ scripts/ torch_xla/ benchmarks/ torchax/
```

### Running the Tests

To run the tests, follow __one__ of the options below:

* Run on local CPU:

  ```Shell
  export PJRT_DEVICE=CPU
  ```

* Run on Cloud TPU:

  ```Shell
  export PJRT_DEVICE=TPU
  ```

* Run on GPU:

  ```Shell
  export PJRT_DEVICE=CUDA GPU_NUM_DEVICES=${NUM_GPU}
  ```

For more detail on configuring the runtime, please refer to [this doc](https://github.com/pytorch/xla/blob/master/docs/pjrt.md#quickstart)

If you are planning to be building from source and hence using the latest _PyTorch/TPU_ code base,
it is suggested for you to select the _Nightly_ builds when you create a Cloud TPU instance.

Then run `test/run_tests.sh` and `test/cpp/run_tests.sh` to verify the setup is working.

### Useful materials
1. [OP Lowering Guide](https://github.com/pytorch/xla/blob/master/OP_LOWERING_GUIDE.md)
2. [CODEGEN MIGRATION GUIDE](https://github.com/pytorch/xla/blob/master/CODEGEN_MIGRATION_GUIDE.md)
3. [Dynamo Integration Guide](https://github.com/pytorch/xla/blob/master/docs/dynamo.md)

### Sharp Edges

* If local changes aren't visible, uninstall existing pytorch/xla with `pip uninstall torch_xla` and `pip uninstall torch`, then rebuild PyTorch and PyTorch/XLA with `python setup.py develop` or `python setup.py install`.
* PJRT errors when running on TPU such as `The PJRT plugin has PJRT API version 0.34. The framework PJRT API version is 0.40`. You need to update your `libtpu.so` and ensure it's in your `LD_LIBRARY_PATH` environmental directory. You can download a new `libtpu.so` at [Google Cloud](https://storage.googleapis.com/libtpu-wheels/index.html), which are sorted by date. Download the newest one and install it at `pip install libtpu...whl`.

## Creating a Pull Request

On your Linux machine (not inside the dev container), create a local branch,
commit your local changes to it, and push the change to GitHub:

```bash
# Assuming that WORKSPACE_DIR is your workspace directory.
cd $WORKSPACE_DIR/pytorch/xla
git checkout -b my-branch
# ... make changes
git add foo/bar.cpp
git commit -m "Implement feature X."
# Push the committed local changes to GitHub.
# You only need to run the next line once.
git config --global push.autoSetupRemote true
git push
```

The last command will print a link for creating a PR. Open the link to create
the PR.

## Updating Forked Repos

From time to time, you'll need to bring your forked repos up to date with
the original (aka upstream) repos. You can do this one repo at a time
by running the following commands on your Linux machine (not inside the
dev container).

First, for the `pytorch` repo:

```bash
cd $WORKSPACE_DIR/pytorch
# Fetch the latest changes from upstream.
git fetch upstream
git checkout main 
# Merge the changes from upstream/main into your local branch.
git merge upstream/main
# Update submodules to match the latest changes.
git submodule update --recursive 
# Push the updated branch to your fork on GitHub.
git push origin main
```

Next, for the `vision` repo:

```bash
cd $WORKSPACE_DIR/vision
git fetch upstream
git checkout main 
git merge upstream/main
git push origin main
```

Finally, for the `pytorch/xla` repo (note that the primary branch is called
`master` instead of `main` in this repo):

```bash
cd $WORKSPACE_DIR/pytorch/xla
git fetch upstream
git checkout master
git merge upstream/master
git push origin master
```

## Updating Local Branch with Upstream Changes

While you work on a PR, other PRs may be merged into the upstream repo's
default branch, and you may want to make sure your PR works with them.
In this case, you'll want to rebase your commits on top of the upstream
commits. You can do this by:

```bash
cd $WORKSPACE_DIR/pytorch/xla
git checkout your-branch-name
# Update the remote-tracking branches for upstream.
git fetch upstream
# Rebase commits in your PR on top of the upstream master branch.
git rebase upstream/master
# If the above command fails due to merge conflicts, follow the error messages
# to resolve the conflicts.
# When you are done, push the updated branch to your fork on GitHub. This will
# update the PR.
git push --force-with-lease origin your-branch-name
```
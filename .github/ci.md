# CI Overview

PyTorch and PyTorch/XLA use CI to lint, build, and test each PR that is
submitted. All CI tests should succeed before the PR is merged into master.
PyTorch CI pins PyTorch/XLA to a specific commit. On the other hand, PyTorch/XLA
CI pulls PyTorch from master unless a pin is manually provided. This README will
go through the reasons of these pins, how to pin a PyTorch/XLA PR to an upstream
PyTorch PR, and how to coordinate a merge for breaking PyTorch changes.

## Usage

### Pinning PyTorch PR in PyTorch/XLA PR

Sometimes a PyTorch/XLA PR needs to be pinned to a specific PyTorch PR to test
new features, fix breaking changes, etc. Since PyTorch/XLA CI pulls from PyTorch
master by default, we need to manually provide a PyTorch pin. In a PyTorch/XLA
PR, PyTorch can be manually pinned by creating a `.torch_pin` file at the root
of the repository. The `.torch_pin` should have the corresponding PyTorch PR
number prefixed by "#". Take a look at
[example here](https://github.com/pytorch/xla/pull/7313). Before the PyTorch/XLA
PR gets merged, the `.torch_pin` must be deleted.

### Coordinating merges for breaking PyTorch PRs

When PyTorch PR introduces a breaking change, its PyTorch/XLA CI tests will
fail. Steps for fixing and merging such breaking PyTorch change is as following:

1. Create a PyTorch/XLA PR to fix this issue with `.torch_pin` and rebase with
   master to ensure the PR is up-to-date with the latest commit on PyTorch/XLA.
   Once this PR is created, it'll create a commit hash that will be used in step
   2\. If you have multiple commits in the PR, use the last one's hash.
   **Important note: When you rebase this PR, it'll create a new commit hash and
   make the old hash obsolete. Be cautious about rebasing, and if you rebase,
   make sure you inform the PyTorch PR's author.**
1. Rebase (or ask the PR owner to rebase) the PyTorch PR with master. Update the
   PyTorch PR to pin the PyTorch/XLA to the commit hash created in step 1 by
   updating `pytorch/.github/ci_commit_pins/xla.txt`.
1. Once CI tests are green on both ends, merge PyTorch PR.
1. Remove the `.torch_pin` in PyTorch/XLA PR and merge. To be noted,
   `git commit --amend` should be avoided in this step as PyTorch CI will keep
   using the commit hash created in step 1 until other PRs update that manually
   or the nightly buildbot updates that automatically.
1. Finally, don't delete your branch until 2 days later. See step 4 for
   explanations.

### Running TPU tests on PRs

The `build_and_test.yml` workflow runs tests on the TPU in addition to CPU and
GPU. The set of tests run on the TPU is defined in `test/tpu/run_tests.sh`.

## CI Environment

Before the CI in this repository runs, we build a base dev image. These are the
same images we recommend in our VSCode `.devcontainer` setup and nightly build
to ensure consistency between environments. We produce variants with and without
CUDA, configured in `infra/ansible` (build config) and
`infra/tpu-pytorch-releases/dev_images.tf` (build triggers).

The CI runs in two environments:

1. Organization self-hosted runners for CPU and GPU: used for almost every step
   of the CI. These runners are managed by PyTorch and have access to the shared
   ECR repository.
1. TPU self-hosted runners: these are managed by us and are only available in
   the `pytorch/xla` repository. See the [_TPU CI_](#tpu-ci) section for more
   details.

## Build and test (`build_and_test.yml`)

We have two build paths for each CI run:

- `torch_xla`: we build the main package to support both TPU and GPU[^1], along
  with a CPU build of `torch` from HEAD. This build step exports the
  `torch-xla-wheels` artifact for downstream use in tests.
  - Some CI tests also require `torchvision`. To reduce flakiness, we compile
    `torchvision` from [`torch`'s CI pin][pytorch-vision-pin].
  - C++ tests are piggybacked onto the same build and uploaded in the
    `cpp-test-bin` artifact.
- `torch_xla_cuda_plugin`: the XLA CUDA runtime can be built independently of
  either `torch` or `torch_xla` -- it depends only on our pinned OpenXLA. Thus,
  this build should be almost entirely cached, unless your PR changes the XLA
  pin or adds a patch.

Both the main package build and plugin build are configured with ansible at
`infra/ansible`, although they run in separate stages (`stage=build_srcs` vs
`stage=build_plugin`). This is the same configuration we use for our nightly and
release builds.

The CPU and GPU test configs are defined in the same file, `_test.yml`. Since
some of the tests come from the upstream PyTorch repository, we check out
PyTorch at the same git rev as the `build` step (taken from
`torch_xla.version.__torch_gitrev__`). The tests are split up into multiple
groups that run in parallel; the `matrix` section of `_test.yml` corresponds to
in `.github/scripts/run_tests.sh`.

CPU tests run immediately after the `torch_xla` build completes. This will
likely be the first test feedback on your commit. GPU tests will launch when
both the `torch_xla` and `torch_xla_cuda_plugin` complete. GPU compilation is
much slower due to the number of possible optimizations, and the GPU chips
themselves are quite outdated, so these tests will take longer to run than the
CPU tests.

![CPU tests launch when  is complete](../docs/assets/ci_test_dependency.png)

![GPU tests also depend on CUDA plugin](../docs/assets/ci_test_dependency_gpu.png)

For the C++ test groups in either case, the test binaries are pre-built during
the build phase and packaged in `cpp-test-bin`. This will only be downloaded if
necessary.

\[^1\]: Note: both GPU and TPU support require their respective plugins to be
installed. This package will _not_ work on either out of the box.

### TPU CI

The TPU CI workflow is defined in `_tpu_ci.yml`. It runs only a subset of our
tests due to capacity constraints, defined in `test/tpu/run_tests.sh`. The
runners themselves are containers in GKE managed by
[ARC](https://github.com/actions/actions-runner-controller). The container image
is also based on our dev images, with some changes for ARC compatibility. The
Dockerfile for this image lives in `test/tpu/Dockerfile`.

The actual ARC cluster is defined in Terraform at `infra/tpu-pytorch/tpu_ci.tf`.

### Reproducing test failures

The best way to reproduce failures in the CI is to use the recommended container
configuration in `.devcontainer`. These use identical images/environments as the
CI.

If you cannot reproduce the failure or need to inspect the package built in a CI
run, you can download the `torch-xla-wheels` artifact for that run,
[either locally in your web browser or remotely with the `gh` CLI tool][artifacts].
C++ tests in particular can be quite slow to build. If you need to re-run these
yourself, download the `cpp-test-bin` artifact. You'll have to set some
additional environment variables for these to load the correct `torch` and
plugin binaries, so you should copy the variables we set in `_test.yml` before
running them.

### Generating docs

Our API documentation is generated automatically from the `torch_xla` package
with `sphinx`. The workflow to update our static site is defined in `_docs.yml`.
The workflow is roughly the following:

- Changes to `master` update the docs at `/master` on the `gh-pages` branch.
- Changes to a release branch update the docs under `/releases/rX.Y`.

By default, we redirect to the latest stable version, defined in
[`index.md`](https://github.com/pytorch/xla/blob/gh-pages/index.md).

We build preview docs for every CI, but only push to `gh-pages` for `master` and
release branches. To preview doc changes, download the `github-pages` artifact
locally and open `index.html` in your browser.

Changes to `gh-pages` are pushed by our bot account, `torchxlabot2`.

### FAQ and Troubleshooting

#### Why does PyTorch CI pin PyTorch/XLA?

As mentioned above, [PyTorch CI pins PyTorch/XLA][pytorch-pin-ptxla] to a "known
good" commit to prevent accidental changes from PyTorch/XLA to break PyTorch CI
without warning. PyTorch has hundreds of commits each week, and this pin ensures
that PyTorch/XLA as a downstream package does not cause failures in PyTorch CI.

#### Why does PyTorch/XLA CI pull from PyTorch master?

[PyTorch/XLA CI pulls PyTorch from master][pull-pytorch-master] unless a PyTorch
pin is manually provided. PyTorch/XLA is a downstream package to PyTorch, and
pulling from master ensures that PyTorch/XLA will stay up-to-date and works with
the latest PyTorch changes.

#### TPU CI is broken

If the TPU CI won't run, try to debug using the following steps:

On your cloudtop:

```
gcloud config set project tpu-pytorch
gcloud container clusters get-credentials tpu-ci --location=us-central2
```

Check to see if the runner pod is working:

```
kubectl get pods -n arc-runners
```

If it is working, check the logs:

```
kubectl logs -n arc-runners <runner-pod-name>
```

If there is no runner pod available, you can check the controller logs. First
find the controller pod name:

```
kubectl get pods -n arc-systems
```

The name should match actions-runner-controller-gha-rs-controller-\*. You can
then check the logs by running the following:

```
kubectl logs -n arc-systems <controller-pod-name>
```

If the ephemeralrunner spawning the runner pods is stuck in an error, you can
attempt the following to restart the ephemeralrunner and check the logs:

```
kubectl delete ephemeralrunners --all -A
kubectl logs -f -n arc-runners $(kubectl get pods -n arc-runners -l 'actions.github.com/scale-set-name=v4-runner-set' -o jsonpath='{.items[0].metadata.name}')
```

## Upstream CI image (`build_upstream_image.yml`)

We use different build tools than the upstream `torch` repository due to our
dependency on XLA, namely `bazel`. To ensure the upstream CI has the correct
tools to run XLA, we layer some additional tools and changes on top of our dev
image and push the result to upstream's ECR instance. The upstream CI image is
defined in `.github/upstream`.

If you are making a breaking change to the image, bump the image version tag in
`build_upstream_image.yml` first and then send a PR to `pytorch/pytorch` to
update the tag on their side
([example](https://github.com/pytorch/pytorch/pull/125319)).

Note: the upstream CI still relies on some legacy scripts in `.circleci` rather
than our Ansible config. Don't update these without checking if they break the
upstream CI first! TODO: finally delete these.

<!-- xrefs -->

[artifacts]: https://docs.github.com/en/actions/managing-workflow-runs/downloading-workflow-artifacts
[pull-pytorch-master]: https://github.com/pytorch/xla/blob/f3415929683880192b63b285921c72439af55bf0/.circleci/common.sh#L15
[pytorch-pin-ptxla]: https://github.com/pytorch/pytorch/blob/master/.jenkins/pytorch/common_utils.sh#L119
[pytorch-vision-pin]: https://github.com/pytorch/pytorch/blob/main/.github/ci_commit_pins/vision.txt

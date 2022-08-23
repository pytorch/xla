# CircleCI Overview
PyTorch and PyTorch/XLA use CircleCI to lint, build, and test each PR that is submitted. All CircleCI tests should succeed before the PR is merged into master. PyTorch CircleCI pins PyTorch/XLA to a specific commit. On the other hand, PyTorch/XLA CircleCI pulls PyTorch from master unless a pin is manually provided. This README will go through the reasons of these pins, how to pin a PyTorch/XLA PR to an upstream PyTorch PR, and how to coordinate a merge for breaking PyTorch changes.

## Why does PyTorch CircleCI pin PyTorch/XLA?
As mentioned above, [PyTorch CircleCI pins PyTorch/XLA](https://github.com/pytorch/pytorch/blob/master/.jenkins/pytorch/common_utils.sh#L119) to a "known good" commit to prevent accidental changes from PyTorch/XLA to break PyTorch CircleCI without warning. PyTorch has hundreds of commits each week, and this pin ensures that PyTorch/XLA as a downstream package does not cause failures in PyTorch CircleCI. 

## Why does PyTorch/XLA CircleCI pull from PyTorch master?
[PyTorch/XLA CircleCI pulls PyTorch from master](https://github.com/pytorch/xla/blob/f3415929683880192b63b285921c72439af55bf0/.circleci/common.sh#L15) unless a PyTorch pin is manually provided. PyTorch/XLA is a downstream package to PyTorch, and pulling from master ensures that PyTorch/XLA will stay up-to-date and works with the latest PyTorch changes.

## Pinning PyTorch PR in PyTorch/XLA PR
Sometimes a PyTorch/XLA PR needs to be pinned to a specific PyTorch PR to test new featurues, fix breaking changes, etc. Since PyTorch/XLA CircleCI pulls from PyTorch master by default, we need to manually provided a PyTorch pin. In a PyTorch/XLA PR, PyTorch an be manually pinned by creating a `.torch_pin` under `/torch_patches`. The `.torch_pin` should have the corresponding PyTorch PR number prefixed by "#". Take a look at [example here](https://github.com/pytorch/xla/pull/3792/commits/40f41fb98b0f2386d287eeac0bae86e873d4a9d8).

## Coodinating merges for breaking PyTorch PRs
When PyTorch PR introduces a breaking change, its PyTorch/XLA CircleCI tests will fail. Steps for fixing and merging such breaking changes is as following: 
1. PyTorch PR introduces a breaking change. 
2. PyTorch/XLA PR to fix this issue is created with `.torch_pin`. Once this PR is created, it'll create a commit hash in its branch.
3. Update the PyTorch PR to pin the PyTorch/XLA commit to the commit hash created in step 2.  
4. Once CircleCI tests are green on both ends, merge PyTorch PR. 
5. Remove the `.torch_pin` in PyTorch/XLA PR and merge.

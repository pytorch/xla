# Run a model under torchax

Supporting a new model in torchax means
having this model run using torchax and succeeds.

A model usually consists of executing a list of torch ops
on a set of tensors (i.e. the parameters and inputs) and
produce a new tensor(s). These ops should just work.

However, there are cases that the model doesn't run on
torchax, because:

1. Some op it needs is not implemented.
2. Some op it needs is implemented incorrectly
3. There are some non-torch-op code that interacts with torchax in a non-friendly matter.

Here we present few steps to attempt to fix the related issues. Using dlrm model as
example.

This assumes that you already installed torchax with `pip install -e .` locally.
Following the instructions in [README](../README.md)


### Get torchbench scripts

Following the instructions in https://github.com/pytorch-tpu/run_torchbench


### Run script from run_torchbench:

```bash
(xla2) hanq-macbookpro:run_torchbench hanq$ python models/dlrm.py
Traceback (most recent call last):
  File "/Users/hanq/git/qihqi/run_torchbench/models/dlrm.py", line 16, in <module>
    module = importlib.import_module(model_name)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torchbench-0.1-py3.10.egg/torchbenchmark/models/dlrm/__init__.py", line 15, in <module>
    from .tricks.qr_embedding_bag import QREmbeddingBag
ModuleNotFoundError: No module named 'torchbenchmark.models.dlrm.tricks'
```

Turns out I forgot to run `python install.py dlrm` in the benchmarks folder (cloned from pytorch/benchmark)


### Fixing missing ops:

Rerunning:
```bash
(xla2) hanq-macbookpro:run_torchbench hanq$ python models/dlrm.py
Traceback (most recent call last):
  File "/Users/hanq/git/qihqi/run_torchbench/models/dlrm.py", line 28, in <module>
    print(model(*example))
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/hanq/git/qihqi/run_torchbench/benchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py", line 355, in forward
    return self.sequential_forward(dense_x, lS_o, lS_i)
  File "/Users/hanq/git/qihqi/run_torchbench/benchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py", line 367, in sequential_forward
    ly = self.apply_emb(lS_o, lS_i, self.emb_l)
  File "/Users/hanq/git/qihqi/run_torchbench/benchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py", line 308, in apply_emb
    V = E(sparse_index_group_batch, sparse_offset_group_batch)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/modules/sparse.py", line 390, in forward
    return F.embedding_bag(input, self.weight, offsets,
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/functional.py", line 2360, in embedding_bag
    return handle_torch_function(
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/overrides.py", line 1619, in handle_torch_function
    result = mode.__torch_function__(public_api, types, args, kwargs)
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torchax/torchax/tensor.py", line 215, in __torch_function__
    return func(*args, **(kwargs or {}))
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/functional.py", line 2451, in embedding_bag
    ret, _, _, _ = torch.embedding_bag(
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torchax/torchax/tensor.py", line 230, in __torch_dispatch__
    return self.env.dispatch(func, types, args, kwargs)
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torchax/torchax/tensor.py", line 310, in dispatch
    raise OperatorNotFound(
torchax.tensor.OperatorNotFound: Operator with name aten::_embedding_bag has no lowering
```

Now let's implement this op.

Few tricks while implementing the ops:

1. Feel free to edit the script `models/dlrm.py` while debugging.
2. Useful options to set `env.config.debug_print_each_op = True` will print out each
   op that goes through the dispatcher.
3. Set `env.config.debug_accuracy_for_each_op = True` will in addition of running Jax
   op, it also runs it again in Torch CPU. Then it diffs the result. If the diff is too
   large, then it drops you into pdb for inspection.
4. After inspecting input / output / shapes of the op, maybe it's enough hint for
   you to fix this op. Or, if it's not, then it's adviced to save the inputs / outputs
   and write a unit test for it and iterate on that. Usually a unit test is faster
   to iterate than running a whole model.

After finishing `embedding_bag` badly, I reached the next op

```bash
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/modules/sparse.py", line 390, in forward
    return F.embedding_bag(input, self.weight, offsets,
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/functional.py", line 2451, in embedding_bag
    ret, _, _, _ = torch.embedding_bag(
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torchax/torchax/tensor.py", line 124, in __torch_dispatch__
    return func(*args, **(kwargs or {}))
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/_ops.py", line 594, in __call__
    return self_._op(*args, **kwargs)
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torchax/torchax/tensor.py", line 212, in __torch_function__
    return func(*args, **(kwargs or {}))
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/_ops.py", line 594, in __call__
    return self_._op(*args, **kwargs)
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torchax/torchax/tensor.py", line 227, in __torch_dispatch__
    return self.env.dispatch(func, types, args, kwargs)
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torchax/torchax/tensor.py", line 308, in dispatch
    raise OperatorNotFound(
torchax.tensor.OperatorNotFound: Operator with name aten::_embedding_bag_forward_only has no lowering
```

Turns out, that is the same operator. so adding the @op(torch.ops.aten._embedding_bag_forward_only)
on top of the same op works.

Now the resulting PR is: https://github.com/pytorch/xla/pull/7583

After this `python models/dlrm.py` runs.

NOTE:
The _embedding_bag implementation is actually very crude, just sufficient to make
the model pass.

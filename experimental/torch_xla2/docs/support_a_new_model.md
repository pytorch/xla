# Run a model under torch_xla2

Supporting a new model in torch_xla2 means
having this model run using torch_xla2 and succeeds.

A model usually consists of executing a list of torch ops
on a set of tensors (i.e. the parameters and inputs) and 
produce a new tensor(s). These ops should just work.

However, there are cases that the model doesn't run on
torch_xla2, because:

1. Some op it needs is not implemented.
2. Some op it needs is implemented incorrectly
3. There are some non-torch-op code that interacts with torch_xla2 in a non-friendly matter.

Here we present few steps to attempt to fix the related issues.

# Step 1. Attempt to run the model

To run a model under torch_xla2, the first step is to 
instantiate the model and run it under normal torch.
This usually means eager mode torch CPU. (NOTE: for large
 models, it's recommended to make a model of equal architecture but smaller, by setting fewer layers / dim sizes; OR, use GPU
so that it can run reasonably fast).

In this example, we will use `BERT_pytorch` model from 
torchbench.

## Install torchbench and instantiate a the model

```bash
git clone https://github.com/pytorch/benchmark.git torchbench
cd torchbench
pip install torchvision torchaudio
pip install -e .
```
Now, torchbench is installed, now we need to download 
the model.

```
python install.py BERT_pytorch
```

NOTE: if you run `python install.py` without positional args 
it will download ALL the 100+ models which can take sometime.

Now, let's verify that the model is there by importing it in python.

```python
import torchbenchmark.models.BERT_pytorch

model, sample_inputs = torchbenchmark.models.BERT_pytorch.Model(
  test='eval', device='cpu'
)

print(model(*sample_inputs))
```

If the above succeeds, then the model is ready.

# Attempt to run the model in torchxla2

To run the model in torch_xla2, we need to do 2 things:
1. Move the model's weight to XLA device (i.e. XLA tensors)
2. Move the sample_inputs to XLA device (i.e. XLA tensors)

The API for the above is the `to_xla` method on `Environment` class.
To get the current environment, one can use `torch_xla2.default_env()`.

i.e.

```python
xla_env = torch_xla2.default_env()
model2 = xla_env.to_xla(model)
sample_inputs = xla_env.to_xla(sample_inputs)
with xla_env:
  print(model2(*sample_inputs))
```

You might get something like this:
```bash
Traceback (most recent call last):
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torch_xla2/examples/torchbench_models/BERT_pytorch.py", line 13, in <module>
    benchmark = benchmark_cls(test="eval", device = "cpu") # test = train or eval device = cuda or cpu
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torchbench-0.1-py3.10.egg/torchbenchmark/util/model.py", line 39, in __call__
    obj = type.__call__(cls, *args, **kwargs)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torchbench-0.1-py3.10.egg/torchbenchmark/models/BERT_pytorch/__init__.py", line 174, in __init__
    bert = BERT(
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torchbench-0.1-py3.10.egg/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/bert.py", line 30, in __init__
    self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torchbench-0.1-py3.10.egg/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py", line 24, in __init__
    self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torchbench-0.1-py3.10.egg/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/token.py", line 6, in __init__
    super().__init__(vocab_size, embed_size, padding_idx=0)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/modules/sparse.py", line 145, in __init__
    self.reset_parameters()
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/modules/sparse.py", line 154, in reset_parameters
    init.normal_(self.weight)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/init.py", line 172, in normal_
    return torch.overrides.handle_torch_function(
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/overrides.py", line 1619, in handle_torch_function
    result = mode.__torch_function__(public_api, types, args, kwargs)
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torch_xla2/torch_xla2/tensor.py", line 210, in __torch_function__
    return func(*args, **(kwargs or {}))
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/init.py", line 175, in normal_
    return _no_grad_normal_(tensor, mean, std, generator)
  File "/Users/hanq/homebrew/Caskroom/miniconda/base/envs/xla2/lib/python3.10/site-packages/torch/nn/init.py", line 20, in _no_grad_normal_
    return tensor.normal_(mean, std, generator=generator)
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torch_xla2/torch_xla2/tensor.py", line 224, in __torch_dispatch__
    return self.env.dispatch(func, types, args, kwargs)
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torch_xla2/torch_xla2/tensor.py", line 297, in dispatch
    raise OperatorNotFound(
torch_xla2.tensor.OperatorNotFound: Operator with name aten::normal_ has no lowering
```
if the issue is with operators.

Sometimes it's helpful to see how did this operator is called.
Note that, many times, an operator being called can also be
unnexpected. 

We can turn on logging with 
`xla_env.config.debug_print_each_op` and it will print each operator that is being run.

The logs looks like this:

```
2024-06-16 15:03:13,726 - root - DEBUG - FUNCTION: aten::view
2024-06-16 15:03:13,726 - root - DEBUG - FUNCTION: aten::gelu
2024-06-16 15:03:13,729 - root - DEBUG - FUNCTION: aten::view
2024-06-16 15:03:13,729 - root - DEBUG - FUNCTION: aten::t
2024-06-16 15:03:13,729 - root - DEBUG -  FUNCTION: transpose
2024-06-16 15:03:13,729 - root - DEBUG -   DISPATCH: aten::transpose.int
2024-06-16 15:03:13,730 - root - DEBUG -    FUNCTION: permute
2024-06-16 15:03:13,730 - root - DEBUG -     DISPATCH: aten::permute
2024-06-16 15:03:13,731 - root - DEBUG - FUNCTION: aten::addmm
2024-06-16 15:03:13,737 - root - DEBUG - FUNCTION: aten::view
2024-06-16 15:03:13,739 - root - DEBUG - FUNCTION: aten::add.Tensor
2024-06-16 15:03:13,740 - root - DEBUG - FUNCTION: aten::slice.Tensor
2024-06-16 15:03:13,740 - root - DEBUG - FUNCTION: aten::select.int
2024-06-16 15:03:13,740 - root - DEBUG - FUNCTION: aten::t
2024-06-16 15:03:13,740 - root - DEBUG -  FUNCTION: transpose
2024-06-16 15:03:13,740 - root - DEBUG -   DISPATCH: aten::transpose.int
2024-06-16 15:03:13,740 - root - DEBUG -    FUNCTION: permute
2024-06-16 15:03:13,740 - root - DEBUG -     DISPATCH: aten::permute
2024-06-16 15:03:13,740 - root - DEBUG - FUNCTION: aten::addmm
2024-06-16 15:03:13,741 - root - DEBUG - FUNCTION: aten::_log_softmax
2024-06-16 15:03:13,741 - root - DEBUG - FUNCTION: aten::view
2024-06-16 15:03:13,741 - root - DEBUG - FUNCTION: aten::t
2024-06-16 15:03:13,741 - root - DEBUG -  FUNCTION: transpose
2024-06-16 15:03:13,741 - root - DEBUG -   DISPATCH: aten::transpose.int
2024-06-16 15:03:13,741 - root - DEBUG -    FUNCTION: permute
2024-06-16 15:03:13,741 - root - DEBUG -     DISPATCH: aten::permute
2024-06-16 15:03:13,764 - root - DEBUG - FUNCTION: aten::addmm
2024-06-16 15:03:13,788 - root - DEBUG - FUNCTION: aten::view
2024-06-16 15:03:13,790 - root - DEBUG - FUNCTION: aten::_log_softmax
```
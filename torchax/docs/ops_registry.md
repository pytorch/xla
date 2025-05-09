# Ops Registry

## Background

In the [How it works](how_it_works.md) doc, we mentioned 2 important pieces:

1. A mechanism to route `ATen` ops to implementation written in
   Jax or in PyTorch, and

2. The ops themselves.


Ops Registry is there to help us to organize the ops themselves.

An op implementation can written in terms of Jax, or in other PyTorch ops.
The latter is also known as "decompositions". For decompositions,
one need to be careful of not introducing circular dependencies.

Here we simply store the operator implementations in a dictionary,
which key the torch / Aten callable that we wish to override, and
value an instance of `Operator` class.

`Operator` class has this schema:

```python
@dataclasses.dataclass
class Operator:
    torch_op: TorchCallable
    func: Union[TorchCallable, JaxCallable]
    is_jax_function: bool
    is_user_defined: bool
    needs_env: bool
    is_view_op: bool
```

The `torch_op` is the corresponding torch callable, and `func` the implementation. `is_jax_function` is True if `func` is implemented using Jax, False if `func` is implemented using other torch ops. We can use this information to decide how to call it.

If `needs_env` is true, `func` will recieve an extra kwarg with name `env`.
This will be the "Environment" in which this op operate on. In particular,
the environment will contain the Jax random number generator key, that might be useful for ops like `aten::rand`.


# Optimizing Repeated Layers with `scan` and `scan_layers`

This is a guide for using `scan` and `scan_layers` in PyTorch/XLA.

## When should you use this

Consider using [`scan_layers`][scan_layers] if you have a model with
many homogenous (same shape, same logic) layers, for example LLMs. These models
can be slow to compile. `scan_layers` is a drop-in replacement for a for loop over
homogenous layers, such as a bunch of decoder layers. `scan_layers` traces the
first layer and reuses the compiled result for all subsequent layers, significantly
reducing the model compile time.

[`scan`][scan] on the other hand is a lower level higher-order-op modeled after
[`jax.lax.scan`][jax-lax-scan]. Its primary purpose is to implement
`scan_layers` under the hood. However, you may find it useful
to program loop logic where the loop itself has a first-class
representation in the compiler (specifically, the XLA `while` op).

## `scan_layers` example

Typically, a transformer model passes the input embedding through a sequence of
homogenous decoder layers:

```python
def run_decoder_layers(self, hidden_states):
  for decoder_layer in self.layers:
    hidden_states = decoder_layer(hidden_states)
  return hidden_states
```

When this function is lowered into an HLO graph, the for loop is unrolled into a
flat list of operations, resulting in long compile times. To reduce compile
times, replace the for loop with `scan_layers`, as shown in
[`decoder_with_scan.py`][decoder_with_scan]:

```python
def run_decoder_layers(self, hidden_states):
  from torch_xla.experimental.scan_layers import scan_layers
  return scan_layers(self.layers, hidden_states)
```

You can train this decoder model by running the following command from the root
directory of a `pytorch/xla` source checkout.

```sh
python3 examples/train_decoder_only_base.py scan.decoder_with_scan.DecoderWithScan
```

## `scan` example

[`scan`][scan] takes a combine function and applies that function over the leading
dimension of tensors while carrying along state:

```python
def scan(
    fn: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
) -> tuple[Carry, Y]:
  ...
```

Use it to loop over the leading dimension of tensors efficiently. If `xs`
is a single tensor, this function is roughly equal to the following Python code:

```python
def scan(fn, init, xs):
  ys = []
  carry = init
  for i in len(range(xs.size(0))):
    carry, y = fn(carry, xs[i])
    ys.append(y)
  return carry, torch.stack(ys, dim=0)
```

Under the hood, `scan` is implemented efficiently by lowering the loop
into an XLA `while` operation. This ensures that only one iteration of the loop
is compiled by XLA.

[`scan_examples.py`][scan_examples] contains some example code showing how to use
`scan`. In that file, `scan_example_cumsum` uses `scan` to implement a cumulative
sum. `scan_example_pytree` demonstrates how to pass PyTrees to `scan`.

You can run the examples with:

```sh
python3 examples/scan/scan_examples.py
```

The output should look something like the following:

```
Running example: scan_example_cumsum
Final sum: tensor([6.], device='xla:0')
History of sums tensor([[1.],
        [3.],
        [6.]], device='xla:0')


Running example: scan_example_pytree
Final carry: {'sum': tensor([15.], device='xla:0'), 'count': tensor([5.], device='xla:0')}
Means over time: tensor([[1.0000],
        [1.5000],
        [2.0000],
        [2.5000],
        [3.0000]], device='xla:0')
```

## Limitations

### AOTAutograd compatibility requirement

The functions/modules passed to `scan` and `scan_layers` must be AOTAutograd
traceable. In particular, as of PyTorch/XLA 2.6, `scan` and `scan_layers` cannot
trace functions with custom Pallas kernels. That means if your decoder uses,
for example flash attention, then it is incompatible with `scan`. We are working on
[supporting this important use case][flash-attn-issue].

### AOTAutograd overhead

Because `scan` uses AOTAutograd to figure out the backward pass of the input
function/module on every iteration, it is easy to become tracing-bound compared to
a for loop implementation. In fact, the  `train_decoder_only_base.py` example runs
slower under `scan` than with for loop as of PyTorch/XLA 2.6 due to this overhead.
We are working on [improving tracing speed][retracing-issue]. This is less of a
problem when your model is very large or has many layers, which are the situations
you would want to use `scan`.

## Compile time experiments

To demonstrate the compile time savings, we'll train a simple decoder with many
layers on a single TPU chip with for loops vs with `scan_layers`.

- Run the for loop implementation:

```sh
❯ python3 examples/train_decoder_only_base.py \
    --hidden-size 256 \
    --num-layers 50 \
    --num-attention-heads 4 \
    --num-key-value-heads 2 \
    --intermediate-size 2048 \
    --num-steps 5 \
    --print-metrics

...

Metric: CompileTime
  TotalSamples: 3
  Accumulator: 02m57s694ms418.595us
  ValueRate: 02s112ms586.097us / second
  Rate: 0.054285 / second
  Percentiles: 1%=023ms113.470us; 5%=023ms113.470us; 10%=023ms113.470us; 20%=023ms113.470us; 50%=54s644ms733.284us; 80%=01m03s028ms571.841us; 90%=01m03s028ms571.841us; 95%=01m03s028ms571.841us;
  99%=01m03s028ms571.841us
```

- Run the `scan_layers` implementation:

```sh
❯ python3 examples/train_decoder_only_base.py \
    scan.decoder_with_scan.DecoderWithScan \
    --hidden-size 256 \
    --num-layers 50 \
    --num-attention-heads 4 \
    --num-key-value-heads 2 \
    --intermediate-size 2048 \
    --num-steps 5 \
    --print-metrics

...

Metric: CompileTime
  TotalSamples: 3
  Accumulator: 29s996ms941.409us
  ValueRate: 02s529ms591.388us / second
  Rate: 0.158152 / second
  Percentiles: 1%=018ms636.571us; 5%=018ms636.571us; 10%=018ms636.571us; 20%=018ms636.571us; 50%=11s983ms003.171us; 80%=18s995ms301.667us; 90%=18s995ms301.667us; 95%=18s995ms301.667us;
  99%=18s995ms301.667us
```

The maximum compile time dropped from `1m03s` to `19s` by
switching to `scan_layers`.

## References

See https://github.com/pytorch/xla/issues/7253 for the design of `scan` and
`scan_layers` itself.

See the function doc comments of [`scan`][scan] and [`scan_layers`][scan_layers]
for details on how to use them.

<!-- xrefs -->

[scan]: https://github.com/pytorch/xla/blob/master/torch_xla/experimental/scan.py
[scan_layers]: https://github.com/pytorch/xla/blob/master/torch_xla/experimental/scan_layers.py
[flash-attn-issue]: https://github.com/pytorch/xla/issues/8633
[retracing-issue]: https://github.com/pytorch/xla/issues/8632
[jax-lax-scan]: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
[decoder_with_scan]: https://github.com/pytorch/xla/blob/master/examples/scan/decoder_with_scan.py
[scan_examples]: https://github.com/pytorch/xla/blob/master/examples/scan/scan_examples.py

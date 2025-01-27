# Guide for using `scan` and `scan_layers`

This is a guide for using `scan` and `scan_layers` in PyTorch/XLA.

## When should you use this

You should consider using [`scan_layers`][scan_layers] if you have a model with
many homogenous (i.e. same shape, same logic) layers, e.g. LLMs. These models
are slow to compile. `scan_layers` is a drop-in replacement for a for loop over
e.g. a bunch of decoder layers. The benefit is that `scan_layers` will trace the
first layer and reuse the compiled result for all subsequent layers,
significantly reducing the model compile time.

[`scan`][scan] on the other hand is a lower level higher-order-op modeled after
[`jax.lax.scan`][jax-lax-scan]. Its primary purpose is to help implement
`scan_layers` under the hood. However, you may find it useful if you would like
to program some sort of loop logic where the loop itself has a first-class
representation in the compiler (specifically, an XLA `While` op).

## `scan_layers` example

Typically, a transformer model passes the input embedding through a sequence of
homogenous decoder layers like this:

```python
def run_decoder_layers(self, hidden_states):
  for decoder_layer in self.layers:
    hidden_states = decoder_layer(hidden_states)
  return hidden_states
```

When this function is lowered into an HLO graph, the for loop is unrolled into a
flat list of operations, resulting in long compile times. To reduce compile
times, you just need to replace the for loop with a call to `scan_layers`, as can
be seen in [`decoder_with_scan.py`][decoder_with_scan]:

```python
def run_decoder_layers(self, hidden_states):
  from torch_xla.experimental.scan_layers import scan_layers
  return scan_layers(self.layers, hidden_states)
```

You may train this decoder model by running

```sh
python3 examples/train_decoder_only_base.py scan.decoder_with_scan.DecoderWithScan
```

from the root directory of a `pytorch/xla` source checkout.

## `scan` example



## Limitations

### AOTAutograd compatibility requirement

The functions/modules passed to `scan` and `scan_layers` must be AOTAutograd
traceable. In particular, as of PyTorch/XLA 2.6, `scan` and `scan_layers` cannot
trace functions with custom Pallas kernels in them. That means if your decoder
uses e.g. flash attention, then it's incompatible with `scan`. We are working on
[supporting this important use case][flash-attn-issue] in nightly and the next
release.

### AOTAutograd overhead

Another lesser problem is that because `scan` uses AOTAutograd to figure out the
backward pass of the input function/module on every iteration, it's easy to
become tracing bound compared to a for loop implementation. In fact, the 
`train_decoder_only_base.py` example runs slower under `scan` than with for loop
as of PyTorch/XLA 2.6 due to this overhead. We are working on
[improving tracing speed][retracing-issue]. This is less of a problem when your
model is very large or has many layers, which are the situations you would want
to use `scan` anyways.

## References

See https://github.com/pytorch/xla/issues/7253 for the design of `scan` and
`scan_layers` itself.

<!-- xrefs -->

[scan]: https://github.com/pytorch/xla/blob/master/torch_xla/experimental/scan.py
[scan_layers]: https://github.com/pytorch/xla/blob/master/torch_xla/experimental/scan_layers.py
[flash-attn-issue]: https://github.com/pytorch/xla/issues/8633
[retracing-issue]: https://github.com/pytorch/xla/issues/8632
[jax-lax-scan]: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
[decoder_with_scan]: /examples/scan/decoder_with_scan.py

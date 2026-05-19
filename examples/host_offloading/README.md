## Host offloading example

When doing reverse-mode automatic differentiation, many tensors are saved
during the forward pass to be used to compute the gradient during the backward pass.
Previously you could use `torch_xla.utils.checkpoint` to discard tensors that's easy
to recompute later, called "checkpointing" or "rematerialization". Now PyTorch/XLA
also supports a technique called "host offloading", i.e. moving the tensor to host
and moving them back, adding another tool in the arsenal to save memory. Use
`torch_xla.experimental.stablehlo_custom_call.place_to_host` to move a tensor to host
and `torch_xla.experimental.stablehlo_custom_call.place_to_device` to move a tensor
back to the device. For example, you can use this to move intermediate activations
to host during a forward pass, and move those activations back to device during
the corresponding backward pass.

Because the XLA graph compiler aggressively reorders operations, host offloading is
best used in combination with `scan`.

TODO(yifeit): Clean up the example in https://github.com/tengyifei/playground/blob/master/graph_transforms/offloading.py
and put that here.

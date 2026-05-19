import torch.utils._pytree as pytree
from torch.autograd import Function


# Taken from https://github.com/pytorch/pytorch/issues/96337
#
# The main purpose is to support autograd in the `scan` operator, which takes in
# PyTrees and outputs PyTrees. Builtin PyTorch autograd ignores tensors in
# non-trivial PyTrees such as dictionaries of tensors. This decorator adds
# arbitrary PyTree support by flattening the PyTree before handing to PyTorch and
# unflattening on the way back.
def pytreeify(cls):
  assert issubclass(cls, Function)

  orig_fw = cls.forward
  orig_bw = cls.backward
  orig_apply = cls.apply

  def new_apply(*inp):
    flat_inp, struct = pytree.tree_flatten(inp)
    out_struct_holder = []
    flat_out = orig_apply(struct, out_struct_holder, *flat_inp)
    assert flat_out is not None
    assert len(out_struct_holder) == 1
    return pytree.tree_unflatten(flat_out, out_struct_holder[0])

  def new_forward(ctx, struct, out_struct_holder, *flat_inp):
    inp = pytree.tree_unflatten(flat_inp, struct)
    out = orig_fw(ctx, *inp)
    flat_out, out_struct = pytree.tree_flatten(out)
    ctx._inp_struct = struct
    ctx._out_struct = out_struct
    out_struct_holder.append(out_struct)
    return tuple(flat_out)

  def new_backward(ctx, *flat_grad_outputs):
    grad_outputs = pytree.tree_unflatten(flat_grad_outputs, ctx._out_struct)
    if not isinstance(grad_outputs, tuple):
      grad_outputs = (grad_outputs,)
    grad_inputs = orig_bw(ctx, *grad_outputs)
    flat_grad_inputs, grad_inputs_struct = pytree.tree_flatten(grad_inputs)
    if grad_inputs_struct != ctx._inp_struct:
      raise RuntimeError("The backward generated an arg structure that doesn't "
                         "match the forward's input.")
    return (None, None) + tuple(flat_grad_inputs)

  cls.apply = new_apply
  cls.forward = new_forward
  cls.backward = new_backward
  return cls

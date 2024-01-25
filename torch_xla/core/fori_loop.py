import pickle
import sys
import threading
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.utils.utils as xu


class ForiLoopOp(object):
  """Creates a PyTorch operation with an XLA lowering function.

  Args:
    name (str): The name of the operation.
    opfn (callable): The function implementing the XLA lowering.
  """

  def __init__(self, name, opfn):
    self._name = name
    self._opfn = opfn
    self._opname = 'xla::_op_' + name
    self._lock = threading.Lock()
    self._computations = dict()

  def __call__(self, *args, **kwargs):
    """Perform the PyTorch operation based on XLA tensors.

    Args:
      args: The PyTorch XLA tensors which are inputs of the operation.
      kwargs: Keyword arguments passed to the lowering function. These are
        Python scalars and cannot be XLA tensors.
    Returns:
      The PyTorch tensors wrapping the values returned by XLA lowering function.
    """
    # add op_fn in this file too, so we could accept body and cond
    
    # get torch_func(args[0])'s HLO2
    # args[0] is torch_func name
    xm.mark_step() # clear all pending irs
    res = (args[0])(*example_input) # res = callable(*example_input)
    torch_func_hlo = torch_xla._XLAC._get_xla_tensors_hlo([res])

    # get all args[1:]'s computation, then generate HLO2
    # args[1:] are all parameters
    shapes = xb.tensor_shape(args[1:])
    key = pickle.dumps([shapes, kwargs])
    with self._lock:
      computation = self._computations.get(key, None)
      if computation is None:
        computation = xb.create_computation(self._name, self._opfn, shapes,
                                            **kwargs)
        self._computations[key] = computation
        if xu.getenv_as('XLA_OP_PRINT_COMPUTATIONS', bool, False):
          print(xb.get_computation_hlo(computation), file=sys.stderr)

    # replace body in HLO2 with HLO1 (test with replace directly)

    # transfer new generated HLO to XLAcomputation
    ## there we need one python binding function, and need C++ level implementation(or existing func)
    
    
    # then return generated XLAcomputation
    ### result = torch_xla._XLAC._xla_user_computation(self._opname, args,
    ###                                                computation)
    ### return result[0] if len(result) == 1 else result


def registerforiloop(name, opfn):
  """Registers a PyTorch operation with an XLA lowering function.

  Python based XLA operations should be preferably registered globally, in order
  to amortize the lowering cost, but this is not a requirement.

  Example::

    import torch
    import torch_xla
    import torch_xla.core.xla_op_registry as xor
    import torch_xla.core.xla_model as xm

    def slice_and_add(a, b, dimno=0):
      sa = a.slice_in_dim(start_index=0, limit_index=1, dimno=dimno)
      sb = b.slice_in_dim(start_index=1, limit_index=2, dimno=dimno)
      return sa + sb

    # Register XLA function globally for better lowering cost savings.
    SLICE_AND_ADD = xor.register('slice_and_add', slice_and_add)

    def user_computation_test():
      device = xm.xla_device()
      x = torch.randn(2, 2).to(device)
      y = torch.randn(2, 2).to(device)
      z = SLICE_AND_ADD(x, y, dimno=0)
      print(z.cpu())

  Args:
    name (str): The name of the operation.
    opfn (callable): The function implementing the XLA lowering.
  Returns:
    The `Op` object which can be called to perform the XLA driven PyTorch
    operation.
  """
  return ForiLoopOp(name, opfn)

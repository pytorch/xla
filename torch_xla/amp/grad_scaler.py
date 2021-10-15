import torch
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_op_registry as xor
import inspect


class GradScaler(torch.cuda.amp.GradScaler):
  """
  An torch_xla variant of torch.cuda.amp.GradScaler that helps perform the steps of gradient scaling
  conveniently.
  Args:
      init_scale (float, optional, default=2.**16):  Initial scale factor.
      growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
          :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
      backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
          :meth:`update` if inf/NaN gradients occur in an iteration.
      growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
          that must occur for the scale to be multiplied by ``growth_factor``.
      enabled (bool, optional, default=True):  If ``False``, disables gradient scaling. :meth:`step` simply
          invokes the underlying ``optimizer.step()``, and other methods become no-ops.
      use_zero_grad (bool, optional, default=False): If ``True``, enables the torch_xla specific zero gradients
          optimization that performs ``optimizer.step()`` with gradients set to zero instead of skipping it when
          inf/NaN gradients occur. This may improve the performance by removing the barrier in GradScaler.
  """

  def __init__(
      self,
      init_scale=2.0**16,
      growth_factor=2.0,
      backoff_factor=0.5,
      growth_interval=2000,
      enabled=True,
      use_zero_grad=False,
  ):
    super().__init__(
        init_scale=init_scale,
        growth_factor=growth_factor,
        backoff_factor=backoff_factor,
        growth_interval=growth_interval,
        enabled=enabled,
    )

    def get_scaling_factor(a):

      def if_true(a):
        return xb.Op.zero(a.builder())

      def if_false(a):
        return xb.Op.one(a.builder())

      cond = a != xb.Op.zero(a.builder())
      return cond.mkconditional((a,), if_true, if_false)

    self.get_scaling_factor = xor.register("get_scaling_factor",
                                           get_scaling_factor)
    self.use_zero_grad = use_zero_grad

  def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
    retval = None
    is_syncfree_optim = "found_inf" in inspect.signature(
        optimizer.step).parameters
    if is_syncfree_optim:
      found_inf = torch.stack(
          tuple(optimizer_state["found_inf_per_device"].values())).sum()
      kwargs['found_inf'] = found_inf
      retval = optimizer.step(*args, **kwargs)
    elif self.use_zero_grad:
      found_inf = torch.stack(
          tuple(optimizer_state["found_inf_per_device"].values())).sum()
      scaling_factor = self.get_scaling_factor(found_inf)
      for grad in xm._fetch_gradients(optimizer):
        grad.nan_to_num_()
        grad.mul_(scaling_factor)
      retval = optimizer.step(*args, **kwargs)
    else:
      xm.mark_step()
      if not sum(
          v.item() for v in optimizer_state["found_inf_per_device"].values()):
        retval = optimizer.step(*args, **kwargs)
    return retval

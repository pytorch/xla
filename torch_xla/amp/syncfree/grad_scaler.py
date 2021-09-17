import torch


class GradScaler(torch.cuda.amp.GradScaler):

  def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
    retval = None
    found_inf = torch.stack(
        tuple(optimizer_state["found_inf_per_device"].values())).sum()
    retval = optimizer.step(found_inf=found_inf)
    return retval

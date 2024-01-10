import torch


def do_bench(fn,
             warmup=25,
             rep=100,
             grad_to_none=None,
             quantiles=None,
             fast_flush=True,
             device=None,
             sync_fn=lambda: torch.cuda.synchronize(),
             return_mode="mean"):
  assert return_mode in ["min", "max", "mean", "median"]
  import torch
  """
    With small modifications implementation is taken from
    https://github.com/openai/triton/blob/a767ca41e189988740d35cbb9aecd873c4874a62/python/triton/testing.py#L79.
    Parts modified relate mostly to a custome syncing function used for XLA, and custom device.
    
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """

  fn()
  sync_fn()

  # We maintain a buffer of 256 MB that we clear
  # before each kernel call to make sure that the L2
  # doesn't contain any input data before the run
  if fast_flush:
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
  else:
    cache = torch.empty(int(256e6), dtype=torch.int8, device=device)

  # Estimate the runtime of the function
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  for _ in range(5):
    cache.zero_()
    fn()
  end_event.record()
  sync_fn()
  estimate_ms = start_event.elapsed_time(end_event) / 5

  # compute number of warmup and repeat
  n_warmup = max(1, int(warmup / estimate_ms))
  n_repeat = max(1, int(rep / estimate_ms))
  start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
  end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
  # Warm-up
  for _ in range(n_warmup):
    fn()
  # Benchmark
  fn_output = None
  for i in range(n_repeat):
    # we don't want `fn` to accumulate gradient values
    # if it contains a backward pass. So we clear the
    # provided gradients
    if grad_to_none is not None:
      for x in grad_to_none:
        x.grad = None
    # we clear the L2 cache before each run
    cache.zero_()
    # record time of `fn`
    start_event[i].record()
    fn_output = fn()
    end_event[i].record()
  # Record clocks
  sync_fn()
  times = torch.tensor(
      [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
      dtype=torch.float)
  if quantiles is not None:
    ret = torch.quantile(times, torch.tensor(quantiles,
                                             dtype=torch.float)).tolist()
    if len(ret) == 1:
      ret = ret[0]
    return ret
  return getattr(torch, return_mode)(times).item(), fn_output

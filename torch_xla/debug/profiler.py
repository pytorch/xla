import functools
import os
import threading
from typing import Union

import torch_xla
import torch_xla.core.xla_model as xm

_TRACER_MARKED_STEP: bool = False


def set_tracer_marked_step(value: bool):
  global _TRACER_MARKED_STEP
  _TRACER_MARKED_STEP = value


def get_tracer_marked_step() -> bool:
  return _TRACER_MARKED_STEP


def start_server(port: int, only_on_master: bool = True) -> object:
  """Start a profiler server on the client side on provided port.

  Users can then use the tensorboard profiler plugin
  (https://github.com/tensorflow/profiler) or the
  :func:`~torch_xla.debug.profiler.trace` as the client to request
  a profiler from this server.

  Args:
    port (int): the port to start the profiler server on. An exception is
      raised if the provided port is invalid or busy.
    only_on_master (bool): whether to only startup server from
      local master ordinal.
  Returns:
    A `ProfilerServer` instance that dictates the lifecycle of the profiler
    server. If this object is garbage collected, the profiler server is
    shut down.
  Raises:
    RuntimeError: Raised if the port is invalid or busy already.
  """
  if not only_on_master or xm.is_master_ordinal():
    return torch_xla._XLAC.profiler.start_server(port)


def trace(service_addr: str,
          logdir: str,
          duration_ms: int = 1000,
          num_tracing_attempts: int = 3,
          host_tracer_level: int = 2,
          device_tracer_level: int = 1,
          delay_ms: int = 0,
          timeout_s: int = 120,
          interval_s: int = 5):
  """Performs an on-demand profiling session on provided profiler servers.

  This method will block until it's done with profiling. Both single and
  multi-host profiling is supported. The output of the profiling requests
  are stored in the logdir specified.

  NOTE(b/177595210): 2VM TPU setup + profiler isn't currently supported
  so both the client VM and TPU cannot be profiled concurrently. Ex.
  service_addr = "localhost:9012,10.0.0.2:8466" does not currently work.

  Args:
    service_addr (str): comma delimited string of addresses of the profiling
      servers to profile. ex. "10.0.0.2:8466" or "localhost:9012".
    logdir (str): the path to write profiling output to. Both the profiler
      client and server must have access. ex. "gs://bucket/file/path".
    duration_ms (int): duration in milliseconds for tracing the server.
    num_tracing_attempts (int): number of trials to send profiling request
      in case of failures.
    host_tracer_level (int): CPU tracing level. Values are: 1 - critical info
      only, 2 - info, 3 - verbose.
      device_tracer_level (int): Device (TPU/GPU) tracing level. Values are: 1 -
      enabled, 0 - disabled.
    delay_ms (int): Specifies the services to start profiling delay_ms
      milliseconds after the current time.
    timeout_s (int): duration to continue retrying sending trace requests.
    interval_s (int): interval for trace request retries.
  """
  options = {
      'host_tracer_level': host_tracer_level,
      'device_tracer_level': device_tracer_level,
      'delay_ms': delay_ms,
  }
  torch_xla._XLAC.profiler.trace(
      service_addr,
      logdir,
      duration_ms=duration_ms,
      num_tracing_attempts=num_tracing_attempts,
      timeout_s=timeout_s,
      interval_s=interval_s,
      options=options)


def trace_detached(*args, **kwargs):
  """
  Wraps the :func:`~torch_xla.debug.profiler.trace` method to capture a profile
  in a background thread. See that method for the list of supported parameters
  and their semantics.
  """
  threading.Thread(target=trace, args=args, kwargs=kwargs).start()


class Trace(torch_xla._XLAC.profiler.TraceMe):
  """Context manager that produces a trace event for profiling.

  The traces generated can then be collected using the above profiling APIs.
  The profiling server first needs to be started up and then can be sampled
  either using Tensorboard profiler plugin
  (https://github.com/tensorflow/profiler) or the
  :func:`~torch_xla.debug.profiler.trace` method.

  Note: currently only supports PyTorch/XLA client side trace events. i.e.,
  the namespace won't group TPU worker side trace.

  Example usage:
  ```python
  server = xp.start_server(9012)

  with xp.Trace('fwd_context'):
    model(input)
    torch_xla.sync()
  ```
  """

  def __init__(self, name: str, **kwargs):
    self.name = name
    super().__init__(name, **kwargs)

  def __enter__(self):
    self.scope = torch_xla._XLAC.profiler.scope_pusher(self.name)
    super().__enter__()

  def __exit__(self, type, value, traceback):
    if getattr(self, 'scope', None):
      del self.scope
    super().__exit__(type, value, traceback)


class StepTrace(Trace):
  """Context manager that produces a step trace event for profiling.

  In addition to being regular traces, the generated traces will
  help provide per-step performance statistics.

  Note: currently only supports PyTorch/XLA client side trace events. i.e.,
  the namespace won't group TPU worker side trace.

  Example usage:
  ```python
  server = xp.start_server(9012)

  for step, (input, label) in enumerate(loader):
    with xp.StepTrace('train_step', step_num=step):
      model(input)
      ...
  ```
  """

  def __init__(self, name: str, **kwargs):
    super().__init__(name, _r=1, **kwargs)

  def __enter__(self):
    set_tracer_marked_step(True)
    super().__enter__()

  def __exit__(self, type, value, traceback):
    if getattr(self, 'scope', None):
      # In ir.cpp ResetScopeContext we ensure that we have no remaining scope
      # before marking step.
      del self.scope
    torch_xla.sync()
    super().__exit__(type, value, traceback)


def trace_me(scope: str):

  def decorator_trace_me(func):

    @functools.wraps(func)
    def wrapper_trace_me(*args, **kwargs):
      with Trace(scope):
        return func(*args, **kwargs)

    return wrapper_trace_me

  return decorator_trace_me


# The profiler implementation is based on JAX implementation
# https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py
class _ProfileState:

  def __init__(self):
    self.profile_session = None
    self.log_dir = None
    self.create_perfetto_link = False
    self.create_perfetto_trace = False
    self.lock = threading.Lock()

  def reset(self):
    _profile_state.profile_session = None
    _profile_state.create_perfetto_link = False
    _profile_state.create_perfetto_trace = False
    _profile_state.log_dir = None


_profile_state = _ProfileState()


def start_trace(log_dir: Union[os.PathLike, str]) -> None:
  """Starts a profiler trace.

  The trace will capture CPU, GPU, and/or TPU activity, including Python
  functions and PyTorch/XLA on-device operations. Use :func:`stop_trace` to end
  the trace and save the results to ``log_dir``.

  The resulting trace can be viewed with TensorBoard. Note that TensorBoard
  doesn't need to be running when collecting the trace.

  Only one trace may be collected at a time. A RuntimeError will be raised if
  :func:`start_trace` is called while another trace is running.

  Args:
    log_dir: The directory to save the profiler trace to (usually the
      TensorBoard log directory).
  """
  with _profile_state.lock:
    if _profile_state.profile_session is not None:
      raise RuntimeError("Profile has already been started. "
                         "Only one profile may be run at a time.")

    _profile_state.profile_session = torch_xla._XLAC.profiler.TslProfilerSessionWrapper(
    )
    _profile_state.log_dir = str(log_dir)


def stop_trace() -> None:
  """Stops the currently-running profiler trace.

  The trace will be saved to the ``log_dir`` passed to the corresponding
  :func:`start_trace` call. Raises a RuntimeError if a trace hasn't been started.
  """
  with _profile_state.lock:
    if _profile_state.profile_session is None:
      raise RuntimeError("No profile started")
    sess = _profile_state.profile_session
    sess.export(sess.stop(), str(_profile_state.log_dir))
    _profile_state.reset()

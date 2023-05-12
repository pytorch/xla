import functools
import torch_xla
import torch_xla.core.xla_model as xm
from contextlib import contextmanager # from jax # add install?
# from functools import wraps # from jax
import glob # from jax # add install?
import gzip # from jax # add install?
import http.server # from jax # add install?
import json # from jax # add install?
import logging # from jax # add install?
import os # from jax # add install?
import socketserver # from jax # add install?
import threading # from jax # add install?

from typing import Callable, Optional  # from jax # add install?

# from jax._src import traceback_util
# traceback_util.register_exclusion(__file__) # this is used for annotation, and will continue divedeep later according to https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras

## # actual use:
## from typing import Any, Callable, List, Optional, TypeVar, cast
## _exclude_paths: List[str] = [__file__, util.__file__]
## _exclude_paths.append(__file__)

## # traceback_util is included via this
## pytype_strict_library(
##     name = "traceback_util",
##     srcs = ["_src/traceback_util.py"],
##     visibility = [":internal"] + jax_visibility("traceback_util"),
##     deps = [
##         ":config",
##         ":util",
##         "//jax/_src/lib",
##     ],
## )


# from jax._src import xla_bridge
# from jax._src.lib import xla_client # "@xla//xla/python:xla_client",

# this block is used to create deps for use from xla_client
# symlink_files(
#     name = "xla_client",
#     srcs = ["@xla//xla/python:xla_client"],
#     dst = ".",
#     flatten = True,
# )
# # then how jax use `xla_client`:
# xla_client.profiler.ProfilerServer
# xla_client.profiler.ProfilerServer
# xla_client.profiler.start_server(port)
# xla_client.profiler.ProfilerSession()
# xla_client.profiler.TraceMe
# xla_client.heap_profile(xla_bridge.get_backend(backend))

# how ptxla include profiler
# torch_xla._XLAC.profiler.TraceMe
# in the BUILD file of torch_xla/BUILD
# "//third_party/xla_client:profiler",
from torch_xla._XLAC import xla_client

_profiler_server: Optional[xla_client.profiler.ProfilerServer] = None

logger = logging.getLogger(__name__)


# _TRACER_MARKED_STEP: bool = False


# def set_tracer_marked_step(value: bool):
#   global _TRACER_MARKED_STEP
#   _TRACER_MARKED_STEP = value


# def get_tracer_marked_step() -> bool:
#   return _TRACER_MARKED_STEP


# def start_server(port: int, only_on_master: bool = True) -> object:
#   """Start a profiler server on the client side on provided port.

#   Users can then use the tensorboard profiler plugin
#   (https://github.com/tensorflow/profiler) or the
#   :func:`~torch_xla.debug.profiler.trace` as the client to request
#   a profiler from this server.

#   Args:
#     port (int): the port to start the profiler server on. An exception is
#       raised if the provided port is invalid or busy.
#     only_on_master (bool): whether to only startup server from
#       local master ordinal.
#   Returns:
#     A `ProfilerServer` instance that dictates the lifecycle of the profiler
#     server. If this object is garbage collected, the profiler server is
#     shut down.
#   Raises:
#     RuntimeError: Raised if the port is invalid or busy already.
#   """
#   if not only_on_master or xm.is_master_ordinal():
#     return torch_xla._XLAC.profiler.start_server(port)


# def trace(service_addr: str,
#           logdir: str,
#           duration_ms: int = 1000,
#           num_tracing_attempts: int = 3,
#           host_tracer_level: int = 2,
#           device_tracer_level: int = 1,
#           delay_ms: int = 0,
#           timeout_s: int = 120,
#           interval_s: int = 5):
#   """Performs an on-demand profiling session on provided profiler servers.

#   This method will block until it's done with profiling. Both single and
#   multi-host profiling is supported. The output of the profiling requests
#   are stored in the logdir specified.

#   NOTE(b/177595210): 2VM TPU setup + profiler isn't currently supported
#   so both the client VM and TPU cannot be profiled concurrently. Ex.
#   service_addr = "localhost:9012,10.0.0.2:8466" does not currently work.

#   Args:
#     service_addr (str): comma delimited string of addresses of the profiling
#       servers to profile. ex. "10.0.0.2:8466" or "localhost:9012".
#     logdir (str): the path to write profiling output to. Both the profiler
#       client and server must have access. ex. "gs://bucket/file/path".
#     duration_ms (int): duration in milliseconds for tracing the server.
#     num_tracing_attempts (int): number of trials to send profiling request
#       in case of failures.
#     host_tracer_level (int): CPU tracing level. Values are: 1 - critical info
#       only, 2 - info, 3 - verbose.
#       device_tracer_level (int): Device (TPU/GPU) tracing level. Values are: 1 -
#       enabled, 0 - disabled.
#     delay_ms (int): Specifies the services to start profiling delay_ms
#       milliseconds after the current time.
#     timeout_s (int): duration to continue retrying sending trace requests.
#     interval_s (int): interval for trace request retries.
#   """
#   options = {
#       'host_tracer_level': host_tracer_level,
#       'device_tracer_level': device_tracer_level,
#       'delay_ms': delay_ms,
#   }
#   torch_xla._XLAC.profiler.trace(
#       service_addr,
#       logdir,
#       duration_ms=duration_ms,
#       num_tracing_attempts=num_tracing_attempts,
#       timeout_s=timeout_s,
#       interval_s=interval_s,
#       options=options)


# class Trace(torch_xla._XLAC.profiler.TraceMe):
#   """Context manager that produces a trace event for profiling.

#   The traces generated can then be collected using the above profiling APIs.
#   The profiling server first needs to be started up and then can be sampled
#   either using Tensorboard profiler plugin
#   (https://github.com/tensorflow/profiler) or the
#   :func:`~torch_xla.debug.profiler.trace` method.

#   Note: currently only supports PyTorch/XLA client side trace events. i.e.,
#   the namespace won't group TPU worker side trace.

#   Example usage:
#   ```python
#   server = xp.start_server(9012)

#   with xp.Trace('fwd_context'):
#     model(input)
#     xm.mark_step()
#   ```
#   """

#   def __init__(self, name: str, **kwargs):
#     self.name = name
#     super().__init__(name, **kwargs)

#   def __enter__(self):
#     self.scope = torch_xla._XLAC.profiler.scope_pusher(self.name)
#     super().__enter__()

#   def __exit__(self, type, value, traceback):
#     if getattr(self, 'scope', None):
#       del self.scope
#     super().__exit__(type, value, traceback)


# class StepTrace(Trace):
#   """Context manager that produces a step trace event for profiling.

#   In addition to being regular traces, the generated traces will
#   help provide per-step performance statistics.

#   Note: currently only supports PyTorch/XLA client side trace events. i.e.,
#   the namespace won't group TPU worker side trace.

#   Example usage:
#   ```python
#   server = xp.start_server(9012)

#   for step, (input, label) in enumerate(loader):
#     with xp.StepTrace('train_step', step_num=step):
#       model(input)
#       ...
#   ```
#   """

#   def __init__(self, name: str, **kwargs):
#     super().__init__(name, _r=1, **kwargs)

#   def __enter__(self):
#     set_tracer_marked_step(True)
#     super().__enter__()

#   def __exit__(self, type, value, traceback):
#     if getattr(self, 'scope', None):
#       # In ir.cpp ResetScopeContext we ensure that we have no remaining scope
#       # before marking step.
#       del self.scope
#     xm.mark_step()
#     super().__exit__(type, value, traceback)


# def trace_me(scope: str):

#   def decorator_trace_me(func):

#     @functools.wraps(func)
#     def wrapper_trace_me(*args, **kwargs):
#       with Trace(scope):
#         return func(*args, **kwargs)

#     return wrapper_trace_me

#   return decorator_trace_me



################################ JAX code
def start_server(port: int) -> xla_client.profiler.ProfilerServer:
  """Starts the profiler server on port `port`.

  Using the "TensorFlow profiler" feature in `TensorBoard
  <https://www.tensorflow.org/tensorboard>`_ 2.2 or newer, you can
  connect to the profiler server and sample execution traces that show CPU,
  GPU, and/or TPU device activity.
  """
  global _profiler_server
  if _profiler_server is not None:
    raise ValueError("Only one profiler server can be active at a time.")

  # # Make sure backends are initialized before creating a profiler
  # # session. Otherwise on Cloud TPU, libtpu may not be initialized before
  # # creating the tracer, which will cause the TPU tracer initialization to
  # # fail and no TPU operations will be included in the profile.
  # # NOTE(skyewm): I'm not sure this is necessary for start_server (is definitely
  # # is for start_trace), but I'm putting it here to be safe.
  # xla_bridge.get_backend()

  _profiler_server = xla_client.profiler.start_server(port)
  return _profiler_server


def stop_server():
  """Stops the running profiler server."""
  global _profiler_server
  if _profiler_server is None:
    raise ValueError("No active profiler server.")
  _profiler_server = None # Should destroy the profiler server


class _ProfileState:
  def __init__(self):
    self.profile_session = None
    self.log_dir = None
    self.create_perfetto_link = False
    self.create_perfetto_trace = False
    self.lock = threading.Lock()

_profile_state = _ProfileState()


def start_trace(log_dir, create_perfetto_link: bool = False,
                create_perfetto_trace: bool = False) -> None:
  """Starts a profiler trace.

  The trace will capture CPU, GPU, and/or TPU activity, including Python
  functions and JAX on-device operations. Use :func:`stop_trace` to end the trace
  and save the results to ``log_dir``.

  The resulting trace can be viewed with TensorBoard. Note that TensorBoard
  doesn't need to be running when collecting the trace.

  Only once trace may be collected a time. A RuntimeError will be raised if
  :func:`start_trace` is called while another trace is running.

  Args:
    log_dir: The directory to save the profiler trace to (usually the
      TensorBoard log directory).
    create_perfetto_link: A boolean which, if true, creates and prints link to
      the Perfetto trace viewer UI (https://ui.perfetto.dev). The program will
      block until the link is opened and Perfetto loads the trace.
    create_perfetto_trace: A boolean which, if true, additionally dumps a
      ``perfetto_trace.json.gz`` file that is compatible for upload with the
      Perfetto trace viewer UI (https://ui.perfetto.dev). The file will also be
      generated if ``create_perfetto_link`` is true. This could be useful if you
      want to generate a Perfetto-compatible trace without blocking the
      processs.
  """
  with _profile_state.lock:
    if _profile_state.profile_session is not None:
      raise RuntimeError("Profile has already been started. "
                         "Only one profile may be run at a time.")
    # Make sure backends are initialized before creating a profiler
    # session. Otherwise on Cloud TPU, libtpu may not be initialized before
    # creating the tracer, which will cause the TPU tracer initialization to
    # fail and no TPU operations will be included in the profile.
    xla_bridge.get_backend()

    _profile_state.profile_session = xla_client.profiler.ProfilerSession()
    _profile_state.create_perfetto_link = create_perfetto_link
    _profile_state.create_perfetto_trace = (
        create_perfetto_trace or create_perfetto_link)
    _profile_state.log_dir = log_dir


def _write_perfetto_trace_file(log_dir):
  # Navigate to folder with the latest trace dump to find `trace.json.jz`
  curr_path = os.path.abspath(log_dir)
  root_trace_folder = os.path.join(curr_path, "plugins", "profile")
  trace_folders = [os.path.join(root_trace_folder, trace_folder) for
      trace_folder in os.listdir(root_trace_folder)]
  latest_folder = max(trace_folders, key=os.path.getmtime)
  trace_jsons = glob.glob(os.path.join(latest_folder, "*.trace.json.gz"))
  if len(trace_jsons) != 1:
    raise ValueError(f"Invalid trace folder: {latest_folder}")
  trace_json, = trace_jsons

  logger.info("Loading trace.json.gz and removing its metadata...")
  # Perfetto doesn't like the `metadata` field in `trace.json` so we remove
  # it.
  # TODO(sharadmv): speed this up by updating the generated `trace.json`
  # to not include metadata if possible.
  with gzip.open(trace_json, "rb") as fp:
    trace = json.load(fp)
    del trace["metadata"]
  filename = "perfetto_trace.json.gz"
  perfetto_trace = os.path.join(latest_folder, filename)
  logger.info("Writing perfetto_trace.json.gz...")
  with gzip.open(perfetto_trace, "w") as fp:
    fp.write(json.dumps(trace).encode("utf-8"))
  return perfetto_trace

class _PerfettoServer(http.server.SimpleHTTPRequestHandler):
  """Handles requests from `ui.perfetto.dev` for the `trace.json`"""

  def end_headers(self):
    self.send_header('Access-Control-Allow-Origin', '*')
    return super().end_headers()

  def do_GET(self):
    self.server.last_request = self.path
    return super().do_GET()

  def do_POST(self):
    self.send_error(404, "File not found")

def _host_perfetto_trace_file(path):
  # ui.perfetto.dev looks for files hosted on `127.0.0.1:9001`. We set up a
  # TCP server that is hosting the `perfetto_trace.json.gz` file.
  port = 9001
  orig_directory = os.path.abspath(os.getcwd())
  directory, filename = os.path.split(path)
  try:
    os.chdir(directory)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('127.0.0.1', port), _PerfettoServer) as httpd:
      url = f"https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{port}/{filename}"
      print(f"Open URL in browser: {url}")

      # Once ui.perfetto.dev acquires trace.json from this server we can close
      # it down.
      while httpd.__dict__.get('last_request') != '/' + filename:
        httpd.handle_request()
  finally:
    os.chdir(orig_directory)

def stop_trace():
  """Stops the currently-running profiler trace.

  The trace will be saved to the ``log_dir`` passed to the corresponding
  :func:`start_trace` call. Raises a RuntimeError if a trace hasn't been started.
  """
  with _profile_state.lock:
    if _profile_state.profile_session is None:
      raise RuntimeError("No profile started")
    _profile_state.profile_session.stop_and_export(_profile_state.log_dir)
    if _profile_state.create_perfetto_trace:
      abs_filename = _write_perfetto_trace_file(_profile_state.log_dir)
      if _profile_state.create_perfetto_link:
        _host_perfetto_trace_file(abs_filename)
    _profile_state.profile_session = None
    _profile_state.create_perfetto_link = False
    _profile_state.create_perfetto_trace = False
    _profile_state.log_dir = None


@contextmanager
def trace(log_dir, create_perfetto_link=False, create_perfetto_trace=False):
  """Context manager to take a profiler trace.

  The trace will capture CPU, GPU, and/or TPU activity, including Python
  functions and JAX on-device operations.

  The resulting trace can be viewed with TensorBoard. Note that TensorBoard
  doesn't need to be running when collecting the trace.

  Only once trace may be collected a time. A RuntimeError will be raised if a
  trace is started while another trace is running.

  Args:
    log_dir: The directory to save the profiler trace to (usually the
      TensorBoard log directory).
    create_perfetto_link: A boolean which, if true, creates and prints link to
      the Perfetto trace viewer UI (https://ui.perfetto.dev). The program will
      block until the link is opened and Perfetto loads the trace.
    create_perfetto_trace: A boolean which, if true, additionally dumps a
      ``perfetto_trace.json.gz`` file that is compatible for upload with the
      Perfetto trace viewer UI (https://ui.perfetto.dev). The file will also be
      generated if ``create_perfetto_link`` is true. This could be useful if you
      want to generate a Perfetto-compatible trace without blocking the
      processs.
  """
  start_trace(log_dir, create_perfetto_link, create_perfetto_trace)
  try:
    yield
  finally:
    stop_trace()


class TraceAnnotation(xla_client.profiler.TraceMe):
  """Context manager that generates a trace event in the profiler.

  The trace event spans the duration of the code enclosed by the context.

  For example:

  >>> x = jnp.ones((1000, 1000))
  >>> with jax.profiler.TraceAnnotation("my_label"):
  ...   result = jnp.dot(x, x.T).block_until_ready()

  This will cause a "my_label" event to show up on the trace timeline if the
  event occurs while the process is being traced.
  """
  pass


class StepTraceAnnotation(TraceAnnotation):
  """Context manager that generates a step trace event in the profiler.

  The step trace event spans the duration of the code enclosed by the context.
  The profiler will provide the performance analysis for each step trace event.

  For example, it can be used to mark training steps and enable the profiler to
  provide the performance analysis per step:

  >>> while global_step < NUM_STEPS:                                           # doctest: +SKIP
  ...   with jax.profiler.StepTraceAnnotation("train", step_num=global_step):  # doctest: +SKIP
  ...     train_step()                                                         # doctest: +SKIP
  ...     global_step += 1                                                     # doctest: +SKIP

  This will cause a "train xx" event to show up on the trace timeline if the
  event occurs while the process is being traced by TensorBoard. In addition,
  if using accelerators, the device trace timeline will also show a "train xx"
  event. Note that "step_num" can be set as a keyword argument to pass the
  global step number to the profiler.

  """

  def __init__(self, name: str, **kwargs):
    super().__init__(name, _r=1, **kwargs)


def annotate_function(func: Callable, name: Optional[str] = None,
                      **decorator_kwargs):
  """Decorator that generates a trace event for the execution of a function.

  For example:

  >>> @jax.profiler.annotate_function
  ... def f(x):
  ...   return jnp.dot(x, x.T).block_until_ready()
  >>>
  >>> result = f(jnp.ones((1000, 1000)))

  This will cause an "f" event to show up on the trace timeline if the
  function execution occurs while the process is being traced by TensorBoard.

  Arguments can be passed to the decorator via :py:func:`functools.partial`.

  >>> from functools import partial

  >>> @partial(jax.profiler.annotate_function, name="event_name")
  ... def f(x):
  ...   return jnp.dot(x, x.T).block_until_ready()

  >>> result = f(jnp.ones((1000, 1000)))
  """

  name = name or getattr(func, '__qualname__', None)
  name = name or func.__name__
  @wraps(func)
  def wrapper(*args, **kwargs):
    with TraceAnnotation(name, **decorator_kwargs):
      return func(*args, **kwargs)
    return wrapper
  return wrapper



def device_memory_profile(backend: Optional[str] = None) -> bytes:
  """Captures a JAX device memory profile as ``pprof``-format protocol buffer.

  A device memory profile is a snapshot of the state of memory, that describes the JAX
  :class:`jax.DeviceArray` and executable objects present in memory and their
  allocation sites.

  For more information how to use the device memory profiler, see
  :doc:`/device_memory_profiling`.

  The profiling system works by instrumenting JAX on-device allocations,
  capturing a Python stack trace for each allocation. The instrumentation is
  always enabled; :func:`device_memory_profile` provides an API to capture it.

  The output of :func:`device_memory_profile` is a binary protocol buffer that
  can be interpreted and visualized by the `pprof tool
  <https://github.com/google/pprof>`_.

  Args:
    backend: optional; the name of the JAX backend for which the device memory
      profile should be collected.

  Returns:
    A byte string containing a binary `pprof`-format protocol buffer.
  """
  return xla_client.heap_profile(xla_bridge.get_backend(backend))


def save_device_memory_profile(filename, backend: Optional[str] = None) -> None:
  """Collects a device memory profile and writes it to a file.

  :func:`save_device_memory_profile` is a convenience wrapper around :func:`device_memory_profile`
  that saves its output to a ``filename``. See the
  :func:`device_memory_profile` documentation for more information.

  Args:
    filename: the filename to which the profile should be written.
    backend: optional; the name of the JAX backend for which the device memory
      profile should be collected.
  """
  profile = device_memory_profile(backend)
  with open(filename, "wb") as f:
    f.write(profile)

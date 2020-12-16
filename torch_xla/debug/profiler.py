from concurrent import futures
import torch_xla
import torch_xla.core.xla_env_vars as xenv
from typing import Optional


def start_server(port: int):
  """Start a profiler server on the client side on provided port.

  Users can then use the tensorboard profiler plugin
  (https://github.com/tensorflow/profiler) or the
  :func:`~torch_xla.debug.profiler.trace` as the client to request
  a profiler from this server.

  Args:
    port (int): the port to start the profiler server on.
  Returns:
    A `ProfilerServer` instance that dictates the lifecycle of the profiler
    server. If this object is garbage collected, the profiler server is
    shut down.
  """
  return torch_xla._XLAC.profiler.start_server(port)


def trace(service_addr: str,
          logdir: str,
          duration_ms: Optional[int] = 1000,
          num_tracing_attempts: Optional[int] = 3,
          host_tracer_level: Optional[int] = 2,
          device_tracer_level: Optional[int] = 1,
          delay_ms: Optional[int] = 0):
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
      options=options)

import os

from dataclasses import dataclass
from torch.profiler import profile, ProfilerActivity, schedule

WAIT = 2
WARMUP = 5
ACTIVE = 3
DEFAULT_SCHEDULE = schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE)
PROF_ITERS = WAIT + WARMUP + ACTIVE


@dataclass
class MicrobenchResults:
  test_name: str
  testing_speedup: float
  baseline_wall_ms: float
  testing_wall_ms: float

  def __str__(self):
    RED = "\u001B[31m"
    GREEN = "\u001B[32m"
    RESET = "\u001B[0m"

    def wrap_in_color(str, color=None):
      if color is None:
        return str
      return f"{color}{str}{RESET}"

    formatted_baseline_wall_ms = f"{self.baseline_wall_ms:.02f}ms"
    formatted_testing_wall_ms = f"{self.testing_wall_ms:.02f}ms"
    formatted_speedup = f"{self.testing_speedup:.02f}x"
    formatted_test_name = self.test_name
    color = None
    if self.testing_speedup < 0.95:
      color = RED
    if self.testing_speedup > 1.05:
      color = GREEN
    formatted_test_name = wrap_in_color(formatted_test_name, color)
    formatted_baseline_wall_ms = wrap_in_color(formatted_baseline_wall_ms,
                                               color)
    formatted_testing_wall_ms = wrap_in_color(formatted_testing_wall_ms, color)
    formatted_speedup = wrap_in_color(formatted_speedup, color)
    return f"{formatted_test_name}: speedup={formatted_speedup}; base={formatted_baseline_wall_ms}; test={formatted_testing_wall_ms}"


def microbench(test_name,
               baseline_fn,
               testing_fn,
               baseline_bench_fn,
               testing_bench_fn,
               baseline_sync_fn,
               testing_sync_fn,
               save_profile_to_dir=None):
  """Benchmarks testing function against baseline function.

  Args:
      test_name (str): Name of the microbenchmark run.
      baseline_fn (fn): Baseline function to benchmark against.
      testing_fn (fn): Test function to be benchmarked.
      baseline_bench_fn (fn): Benchmarking function for baseline.
      testing_bench_fn (fn): Benchmarking function for test.
      baseline_sync_fn (fn): Profiling sync function for baseline.
      testing_sync_fn (fn): Profiling sync function for test.
      return_mode (str, optional): Benchmarking should consist of multiple runs. Defaults to 'min'.
      save_profile_to_dir (_type_, optional): If not None, saves the profiling results for the slowest runs to the `save_profile_to_dir`. Defaults to None.

  Returns:
      MicrobenchResults: A structure holding `test_name`, speedups, and measured wall times for both functions.
  """

  assert baseline_bench_fn is not None, "Expect baseline_bench_fn to be defined"
  assert testing_bench_fn is not None, "Expect testing_bench_fn to be defined"

  baseline_wall_ms, _ = baseline_bench_fn(baseline_fn)
  testing_wall_ms, _ = testing_bench_fn(testing_fn)

  if save_profile_to_dir and baseline_wall_ms / testing_wall_ms < 0.95:
    filebase_path = save_profile_to_dir
    _perf_prof(
        _get_filepath(filebase_path, test_name, "baseline"), baseline_fn,
        baseline_sync_fn)
    _perf_prof(
        _get_filepath(filebase_path, test_name, "testing"), testing_fn,
        testing_sync_fn)

  return MicrobenchResults(
      test_name=test_name,
      testing_speedup=baseline_wall_ms / testing_wall_ms,
      baseline_wall_ms=baseline_wall_ms,
      testing_wall_ms=testing_wall_ms,
  )


def _dump_flame(filepath, prof):
  """Data can be interpreted with running ./flamegraph.pl --title "CUDA time" --countname "us." /tmp/profiler_stacks.txt > perf_viz.svg

  More about flamegraphs, and the flamegraph.pl script can be found in https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#visualizing-data-as-a-flame-graph.
  """
  os.makedirs(filepath, exist_ok=True)
  filedest = os.path.join(filepath, "flame.txt")
  prof.export_stacks(filedest)


def _dump_kernels(filepath, prof):
  os.makedirs(filepath, exist_ok=True)
  filedest = os.path.join(filepath, "kernel.txt")
  summary = prof.key_averages(group_by_input_shape=True).table(
      sort_by="cuda_time_total", row_limit=100)
  with open(filedest, mode='w', encoding='utf-8') as f:
    f.write(summary)


def _dump_trace(filepath, prof):
  os.makedirs(filepath, exist_ok=True)
  filedest = os.path.join(filepath, "trace.json")
  prof.export_chrome_trace(filedest)


def _perf_prof(filepath, fn, sync_fn):
  assert sync_fn is not None
  with profile(
      activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
      record_shapes=True,
      with_flops=True,
      profile_memory=True,
      with_stack=True,
      schedule=DEFAULT_SCHEDULE) as prof:
    for _ in range(PROF_ITERS):
      fn()
      prof.step()
    sync_fn()
  _dump_flame(filepath, prof)
  _dump_trace(filepath, prof)
  _dump_kernels(filepath, prof)


def _get_filepath(filebase, test_name, suffix):
  return os.path.join(filebase, test_name, suffix)

import argparse
import bench
from collections import OrderedDict
import copy
import json
import logging
import os
import subprocess
import sys
import time
import torch
import torch._dynamo.utils as dynamo_utils
import tiers
from typing import Optional
import torch_xla.debug.metrics as met
from tqdm import tqdm
from enum import Enum
from torch.profiler import profile, ProfilerActivity
import copy
from torch.autograd import DeviceType
from benchmark_model import ModelLoader
from verifier import VerificationCode, VerificationResult, verify
from enum import Enum
from torchbench_model import TorchBenchModelLoader
from benchmark_experiment import ExperimentLoader
from util import reset_rng_state, move_to_device, randomize_input, us_to_s, ns_to_s

import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp

logger = logging.getLogger(__name__)


class ExperimentRunner:

  def __init__(self, args):
    self._args = args

    self.experiment_loader = ExperimentLoader(self._args)

    # Choose model loader.
    if self._args.suite_name == "torchbench":
      self.model_loader = TorchBenchModelLoader(self._args)
    elif self._args.suite_name == "dummy":
      self.model_loader = ModelLoader(self._args)
    else:
      raise NotImplementedError

    self.output_dir = os.path.abspath(self._args.output_dirname)
    os.makedirs(self.output_dir, exist_ok=True)
    self.output_file = os.path.join(self.output_dir, self._args.output_basename)

  def run(self):
    is_main_process = self._args.experiment_config is None and \
      self._args.model_config is None
    if is_main_process:
      self.generate_and_run_all_configs()
    else:
      assert self._args.experiment_config is not None and \
        self._args.model_config is not None
      self.run_single_config()

  def generate_and_run_all_configs(self):
    assert self._args.experiment_config is None and \
      self._args.model_config is None

    # Collect fingerprints for configs to skip. These are configs for which we
    # already have results. The derived fingerprints uniquely identify the
    # benchmark configurations, currently a string.
    skip_fingerprints = set()
    if os.path.exists(self.output_file):
      if self._args.no_resume:
        os.unlink(self.output_file)
      else:
        with open(self.output_file, mode="r", encoding="utf-8") as f:
          jsonlines = f.read().splitlines()
        for ln in jsonlines:
          ln_dict = json.loads(ln)
          skip_fingerprints.add(
              self._get_config_fingerprint(ln_dict["experiment"],
                                           ln_dict["model"]))

    # Enumerate experiment and model configs and launch subprocesses.
    experiment_configs = self.experiment_loader.list_experiment_configs()
    model_configs = self.model_loader.list_model_configs()
    logger.info(
        f"Number of selected experiment configs: {len(experiment_configs)}")
    logger.info(f"Number of selected model configs: {len(model_configs)}")
    for model_cfg in tqdm(
        model_configs,
        desc="Running benchmark configs by model",
        disable=not self._args.progress_bar):
      for experiment_cfg in experiment_configs:

        # Log run and configs.
        logger.info(f"Run with --model-config={json.dumps(model_cfg)} "
                    f"--experiment-config={json.dumps(experiment_cfg)}")

        # Move on if dry running.
        if self._args.dry_run:
          continue

        # TODO: See if we can pass experiment_cfg to `load_experiment`.
        benchmark_experiment = self.experiment_loader.load_experiment(
            experiment_cfg)
        benchmark_model = self.model_loader.load_model(
            model_cfg, benchmark_experiment, dummy=True)

        # Skip already completed benchmark.
        fingerprint = self._get_config_fingerprint(
            benchmark_experiment.to_dict(), benchmark_model.to_dict())
        if fingerprint in skip_fingerprints:
          logger.info(f"SKIP already completed benchmark")
          continue

        # Check if we should execute or skip the current configuration.
        # A configuration SHOULD be skipped if and only if:
        #
        #   1. --no-skip was not specified; AND
        #
        #   2. the model is not compatible with the experiment configuration
        #
        # Otherwise, we should go ahead and execute it.
        if (not self._args.no_skip and not self.model_loader.is_compatible(
            benchmark_model, benchmark_experiment)):
          logger.warning("SKIP incompatible model and experiment configs.")
          self._save_results(benchmark_experiment.to_dict(),
                             benchmark_model.to_dict(), {"error": "SKIP"})
          continue

        # Compose child process environment.
        process_env = os.environ.copy()
        benchmark_experiment.update_process_env(process_env)
        try:
          benchmark_model.update_process_env(process_env)
        except ValueError as e:
          logger.error(f"ERROR preparing child env: {e}")
          self._save_results(benchmark_experiment.to_dict(),
                             benchmark_model.to_dict(), {"error": str(e)})
          continue

        # Setup HLO dumps.
        if self._args.dump_hlo:
          hlo_path = self._get_results_dir_path(experiment_cfg, model_cfg,
                                                "hlo")
          new_xla_flags = f"--xla_dump_to={hlo_path}"
          xla_flags = process_env.pop("XLA_FLAGS", None)
          if xla_flags is None:
            xla_flags = new_xla_flags
          else:
            xla_flags = f"{xla_flags} {new_xla_flags}"
          process_env["XLA_FLAGS"] = xla_flags

        # Launch subprocess.
        try:
          command = [sys.executable] + sys.argv + [
              f"--experiment-config={json.dumps(experiment_cfg)}"
          ] + [f"--model-config={json.dumps(model_cfg)}"] + [
              # Note: if "--timestamp foo" is already in sys.argv, we
              # harmlessly pass "--timestamp foo" again here.
              f"--timestamp={self._args.timestamp}"
          ]
          command_str = " ".join(command)
          logger.debug(f"Run `{command_str}`")
          child_process = subprocess.run(
              command,
              timeout=self._args.subprocess_timeout,
              env=process_env,
              check=True,
              capture_output=True,
              text=True,
          )
          self._fwd_captured_stdout_stderr(child_process.stdout,
                                           child_process.stderr)
        except subprocess.TimeoutExpired as e:
          self._fwd_captured_stdout_stderr(e.stdout, e.stderr)
          logger.error("TIMEOUT")
          self._save_results(benchmark_experiment.to_dict(),
                             benchmark_model.to_dict(), {"error": str(e)})
        except subprocess.CalledProcessError as e:
          self._fwd_captured_stdout_stderr(e.stdout, e.stderr)
          logger.error("ERROR in subprocess")
          self._save_results(benchmark_experiment.to_dict(),
                             benchmark_model.to_dict(), {"error": e.stderr})
        except subprocess.SubprocessError as e:
          logger.error("ERROR when launching child process")
          self._save_results(benchmark_experiment.to_dict(),
                             benchmark_model.to_dict(), {"error": str(e)})
        except ValueError as e:
          logger.error(f"ERROR {e}")
          self._save_results(benchmark_experiment.to_dict(),
                             benchmark_model.to_dict(), {"error": str(e)})

  # TODO: Use `_unique_basename` instead.
  def _get_config_fingerprint(self, experiment_config: OrderedDict,
                              model_config: OrderedDict) -> str:
    # Experiment `batch_size` may be altered by model in `set_up`, so we will
    # ignore that.
    return "-".join(
        list(map(str, model_config.values())) +
        [str(v) for k, v in experiment_config.items() if k != "batch_size"] +
        [str(self._args.batch_size)])

  def _fwd_captured_stdout_stderr(self, stdout_text: str, stderr_text: str):
    if not self._args.print_subprocess:
      return
    print(stdout_text, file=sys.stdout, end='', flush=True)
    print(stderr_text, file=sys.stderr, end='', flush=True)

  def _default_iter_fn(self, benchmark_experiment, benchmark_model,
                       input_tensor):
    tracing_time = None
    total_time_start = time.perf_counter()
    # Invoke iteration function and measure tracing time w/o waiting on the
    # result.
    if benchmark_experiment.xla:
      t_trace_start = time.perf_counter()
    output = benchmark_model.model_iter_fn(
        input_tensor, collect_full_output=self._args.collect_full_output)
    if benchmark_experiment.xla:
      tracing_time = time.perf_counter() - t_trace_start

    # Mark step.
    self._mark_step(benchmark_experiment)
    total_time = time.perf_counter() - total_time_start
    return output, total_time, tracing_time

  def _pure_wall_time_iter_fn(self, benchmark_experiment, benchmark_model,
                              input_tensor):
    device = xm.xla_device() if benchmark_experiment.xla else 'cuda'
    sync_fn = xm.wait_device_ops if benchmark_experiment.xla else torch.cuda.synchronize
    timing, output = bench.do_bench(
        lambda: benchmark_model.model_iter_fn(
            input_tensor, collect_full_output=self._args.collect_full_output),
        return_mode='min',
        sync_fn=sync_fn,
        device=device)
    return output, timing, None

  def run_single_config(self):

    # Load experiment and model.
    experiment_config = json.loads(self._args.experiment_config)
    model_config = json.loads(self._args.model_config)
    benchmark_experiment = self.experiment_loader.load_experiment(
        experiment_config)
    reset_rng_state(benchmark_experiment)
    benchmark_model = self.model_loader.load_model(model_config,
                                                   benchmark_experiment)

    # Repeat the experiment and accumulate metrics.
    last_output = None
    with benchmark_model.pick_grad():
      accumulated_metrics = OrderedDict()
      for repeat_iteration in range(self._args.repeat):
        metrics, last_output = self.run_once_and_gather_metrics(
            benchmark_experiment, benchmark_model, experiment_config,
            model_config, repeat_iteration)
        for k, v in metrics.items():
          if k not in accumulated_metrics:
            accumulated_metrics[k] = []
          accumulated_metrics[k].append(v)

    verify_res = verify(
        last_output,
        experiment_config,
        model_config,
        self.experiment_loader,
        self.model_loader,
        mean_rel_error_tolerance=0.02,  # allow max 2% difference w.r.t eager runtime
        noop=not self._args.verify)
    self._save_results(benchmark_experiment.to_dict(),
                       benchmark_model.to_dict(), accumulated_metrics,
                       verify_res)

  def run_once_and_gather_metrics(self, benchmark_experiment, benchmark_model,
                                  experiment_config, model_config,
                                  repeat_iteration: int):

    # Prepare inputs.
    reset_rng_state(benchmark_experiment)
    inputs_list = self._prepare_inputs(benchmark_model.example_inputs,
                                       self._args.randomize_input)

    # Reset state and sync.
    reset_rng_state(benchmark_experiment)
    self._mark_step(benchmark_experiment)
    self._synchronize(benchmark_experiment)
    met.clear_all()
    dynamo_utils.counters.clear()
    metrics = OrderedDict()

    # Start timers.
    t_start = time.perf_counter()
    if benchmark_experiment.xla:
      t_trace = 0

    def loop(pytorch_profile=None, iter_fn=None):
      nonlocal t_trace
      total_timing = 0
      for i in range(self._args.iterations_per_run):
        output, timing, trace = iter_fn(benchmark_experiment, benchmark_model,
                                        inputs_list[i])
        if trace is not None:
          t_trace += trace
        if timing is not None:
          total_timing += timing

        # Mark step.
        self._mark_step(benchmark_experiment)
        if pytorch_profile is not None:
          pytorch_profile.step()

      self._synchronize(benchmark_experiment)
      return output, total_timing

    # Execute all iterations (with) profiling.
    enable_pytorch_profiling = self._args.dump_pytorch_profiles or \
        self._args.profile_cuda_cpu or \
        self._args.profile_cuda_cpu_individual_ops
    enable_xla_profiling = self._args.profile_xla
    assert not (enable_pytorch_profiling and
                enable_xla_profiling), "More than one profiling path enabled."

    if enable_xla_profiling:
      logdir = self._get_results_dir_path(experiment_config, model_config,
                                          "xplane", "xla-profile")
      xp.trace_detached(
          'localhost:9012',
          logdir=logdir,
          duration_ms=self._args.profile_xla_duration_ms)
      output, _ = loop(iter_fn=self._default_iter_fn)
    elif enable_pytorch_profiling:
      if self._args.pure_wall_time:
        logger.warning(
            'Run with pure wall time, but also with profiling flags enabled. Falling back to a default wall time.'
        )
      with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU
                              ]) as pytorch_profile:
        output, _ = loop(pytorch_profile, iter_fn=self._default_iter_fn)
    else:
      if self._args.pure_wall_time:
        output, pure_wall_timing = loop(iter_fn=self._pure_wall_time_iter_fn)
      else:
        output, _ = loop(iter_fn=self._default_iter_fn)

    # Stop timers.
    t_end = time.perf_counter()

    # Calculate metrics.
    if self._args.pure_wall_time:
      logger.warning(
          'For measuring pure wall time tracing time equals wall time')

    if self._args.pure_wall_time:
      assert pure_wall_timing is not None
      metrics["total_time"] = pure_wall_timing / 1000  # convert ms to s
    else:
      metrics["total_time"] = t_end - t_start

    metrics[
        "per_iter_time"] = metrics["total_time"] / self._args.iterations_per_run
    if benchmark_experiment.xla:
      metrics["trace_total_time"] = t_trace
      metrics["trace_per_iter_time"] = t_trace / self._args.iterations_per_run

    # Dump PyTorch profile and/or extract metrics.
    if self._args.dump_pytorch_profiles:
      self._dump_pytorch_profile(pytorch_profile, experiment_config,
                                 model_config, repeat_iteration)
    if self._args.profile_cuda_cpu:
      self._collect_cuda_cpu_metrics(pytorch_profile, metrics)
    if self._args.profile_cuda_cpu_individual_ops:
      self._collect_cuda_cpu_metrics_individual_ops(benchmark_experiment,
                                                    metrics, pytorch_profile)

    # Dump Dynamo counters and collect metrics.
    if self._args.dump_dynamo_counters:
      self._dump_dynamo_counters(experiment_config, model_config,
                                 repeat_iteration)
    if self._args.collect_dynamo_counters:
      metrics["dynamo_counters"] = copy.deepcopy(dynamo_utils.counters)

    # Dump PyTorch/XLA metrics and extract some.
    if benchmark_experiment.xla:
      if self._args.dump_pytorch_xla_metrics:
        self._dump_pytorch_xla_metrics(experiment_config, model_config,
                                       repeat_iteration)
      for m in ("CompileTime", "ExecuteTime"):
        data = met.metric_data(m)
        data = data if data is not None else (0, 0, [])
        number, total_time, _ = data
        # Time is measured in nano-seconds
        metrics[f"xla_{m}_time_s"] = ns_to_s(total_time)
        metrics[f"xla_{m}_number"] = number

    # Additional experiment metrics can be added here.

    # Save output.
    if self._args.save_output and output is not None:
      output = move_to_device(output, "cpu")
      path = self._get_results_file_path(
          experiment_config, model_config, repeat_iteration, "output", ext="pt")
      torch.save(output, path)

    return metrics, output

  def _prepare_inputs(self, example_inputs, should_randomize_input):
    inputs_list = []
    for i in range(self._args.iterations_per_run):
      inputs = copy.deepcopy(example_inputs)
      if should_randomize_input:
        inputs = randomize_input(inputs)
      inputs_list.append(inputs)
    return inputs_list

  def _mark_step(self, benchmark_experiment):
    if benchmark_experiment.xla:
      xm.mark_step()

  def _synchronize(self, benchmark_experiment):
    if benchmark_experiment.xla:
      xm.wait_device_ops()
    elif benchmark_experiment.accelerator == "cuda":
      torch.cuda.synchronize()
    else:
      pass

  ##############################################################################
  # Helpers to save results and result files.                                  #
  ##############################################################################

  def _unique_basename(self, experiment_config: OrderedDict,
                       model_config: OrderedDict) -> str:

    def unique_basename_segment(x, max_len=32):
      s = str(x).replace(" ", "")
      if len(s) > max_len:
        s = str(hex(hash(s)))
      return s

    # Ignore batch_size as it may be altered by the model.
    sorted_items = sorted(experiment_config.items()) + sorted(
        model_config.items())
    skip_keys = set(["batch_size", "process_env"])
    segments = [
        unique_basename_segment(v)
        for k, v in sorted_items
        if k not in skip_keys
    ]
    return "-".join(segments)

  def _get_results_file_path(self,
                             experiment_config: OrderedDict,
                             model_config: OrderedDict,
                             partial_name: str,
                             ext: Optional[str] = "txt",
                             sub_dirname: Optional[str] = None) -> str:
    is_dir = ext is None
    model_name = model_config["model_name"]
    basename = self._unique_basename(experiment_config, model_config)
    filename = f"{partial_name}-{basename}"
    if not is_dir:
      filename += f".{ext}"
    path = os.path.abspath(os.path.join(self._args.output_dirname, model_name))
    if sub_dirname is not None:
      path = os.path.join(path, sub_dirname)
    path = os.path.join(path, filename)

    # Create (parent) directory.
    os.makedirs(path if is_dir else os.path.dirname(path), exist_ok=True)

    return path

  def _get_results_dir_path(self,
                            experiment_config: OrderedDict,
                            model_config: OrderedDict,
                            partial_name: str,
                            sub_dirname: Optional[str] = None) -> str:
    return self._get_results_file_path(
        experiment_config,
        model_config,
        partial_name,
        ext=None,
        sub_dirname=sub_dirname)

  def _save_results_file(self,
                         text: str,
                         experiment_config: OrderedDict,
                         model_config: OrderedDict,
                         partial_name: str,
                         ext: str = "txt",
                         sub_dirname: Optional[str] = None,
                         mode: str = "w"):
    path = self._get_results_file_path(experiment_config, model_config,
                                       partial_name, ext, sub_dirname)
    with open(path, mode, encoding="utf-8") as f:
      f.write(text)

  def _save_results(
      self,
      experiment_config: OrderedDict,
      model_config: OrderedDict,
      metrics,
      verification_result: Optional[VerificationResult] = VerificationResult(
          VerificationCode.CANNOT_PROCEED_WITH_VERIFICATION)):
    results = OrderedDict()
    results["model"] = model_config
    results["experiment"] = experiment_config
    results["repeat"] = self._args.repeat
    results["iterations_per_run"] = self._args.iterations_per_run
    results["metrics"] = metrics
    results["timestamp"] = self._args.timestamp
    results["verification_code"] = verification_result.result_code
    results["verification_mean_rel_error"] = verification_result.mean_rel_error
    with open(self.output_file, mode="a", encoding="utf-8") as f:
      json.dump(results, f, ensure_ascii=False)
      f.write("\n")

  ##############################################################################
  # Helpers to dump and analyze the PyTorch profile, PyTorch/XLA metrics, etc. #
  ##############################################################################

  def _dump_pytorch_profile(self, profile, experiment_config: OrderedDict,
                            model_config: OrderedDict, repeat_iteration: int):
    assert profile is not None, "Expect PyTorch profile"

    # Dump PyTorch trace.
    profile.export_chrome_trace(
        self._get_results_file_path(
            experiment_config,
            model_config,
            "trace",
            ext="json",
            sub_dirname=str(repeat_iteration)))

    # Dump PyTorch profile.
    text = profile.key_averages().table(
        sort_by="cuda_time_total", row_limit=500)
    self._save_results_file(
        text,
        experiment_config,
        model_config,
        "pytorch-profile",
        sub_dirname=str(repeat_iteration))
    self._save_results_file(
        text, experiment_config, model_config, "pytorch-profile", mode="a")

  def _collect_cuda_cpu_metrics(self, pytorch_profile, metrics):
    assert pytorch_profile is not None, "Expect profile"

    kernel_dump = pytorch_profile.profiler.total_average()
    total_cuda_time = 0
    total_cpu_time = kernel_dump.self_cpu_time_total

    # Avoid double counting CUDA time for inductor. Copied here, since the
    # interface is not really exposed via any interface. The alternative is
    # regex matching resulting string dump for CUDA kernel time. See
    # https://github.com/pytorch/pytorch/blob/2f3beb715c608a060934c237de402faa40ea211f/torch/autograd/profiler_util.py#L1025-L1037
    for evt in pytorch_profile.profiler.key_averages():
      if evt.device_type == DeviceType.CPU:
        # In legacy profiler, kernel info is stored in cpu events
        if evt.is_legacy:
          total_cuda_time += evt.self_cuda_time_total
      elif evt.device_type == DeviceType.CUDA:
        # In kineto profiler, there're events with the correct device type
        # (e.g. CUDA)
        total_cuda_time += evt.self_cuda_time_total

    metrics["total_cpu_time_s"] = us_to_s(total_cpu_time)
    metrics["total_cuda_time_s"] = us_to_s(total_cuda_time)
    metrics["per_iter_cpu_time_s"] = us_to_s(total_cpu_time /
                                             self._args.iterations_per_run)
    metrics["per_iter_cuda_time_s"] = us_to_s(total_cuda_time /
                                              self._args.iterations_per_run)

  def _collect_cuda_cpu_metrics_individual_ops(self, benchmark_experiment,
                                               metrics, pytorch_profile):
    assert pytorch_profile is not None, "Expect profile"
    logger.debug("Collect CUDA and CPU metrics for individual ops")

    def is_aten_op(op_name):
      return 'aten::' in op_name

    def get_xla_cpu_fallback_ops(met):
      return set(name for name in met.counter_names() if is_aten_op(name))

    extract_prof_info = lambda event: {
        "self_cpu_time_s": us_to_s(event.self_cpu_time_total),
        "self_cuda_time_s": us_to_s(event.self_cuda_time_total),
        "total_cpu_time_s": us_to_s(event.cpu_time_total),
        "total_cuda_time_s": us_to_s(event.cuda_time_total),
        "num_of_calls": event.count
    }

    if benchmark_experiment.xla:
      unlowered_ops = get_xla_cpu_fallback_ops(met)
      if not unlowered_ops:
        return
      if "xla_unlowered_ops" not in metrics:
        metrics["xla_unlowered_ops"] = dict()
      for event in pytorch_profile.key_averages():
        if event.key in unlowered_ops:
          metrics["xla_unlowered_ops"][event.key] = extract_prof_info(event)
    else:
      for event in pytorch_profile.key_averages():
        op_name = event.key
        if not is_aten_op(op_name):
          continue
        if "inductor_ops" not in metrics:
          metrics["inductor_ops"] = dict()
        metrics["inductor_ops"][op_name] = extract_prof_info(event)

  def _dump_dynamo_counters(self, experiment_config, model_config,
                            repeat_iteration: int):
    text = f"{json.dumps(dynamo_utils.counters)}\n"
    self._save_results_file(
        text,
        experiment_config,
        model_config,
        "dynamo-counters",
        sub_dirname=str(repeat_iteration))

  def _dump_pytorch_xla_metrics(self, experiment_config, model_config,
                                repeat_iteration):
    text = met.metrics_report()
    assert isinstance(text, str)
    self._save_results_file(
        text,
        experiment_config,
        model_config,
        "pytorch-xla-metrics",
        sub_dirname=str(repeat_iteration))


################################################################################
# CLI                                                                          #
################################################################################


def parse_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--suite-name",
      required=True,
      choices=["dummy", "torchbench"],
      help="Suite name for the model garden.",
  )
  parser.add_argument(
      "--filter",
      "-k",
      action="append",
      default=[],
      help="filter benchmarks with regexp")
  parser.add_argument(
      "--exclude",
      "-x",
      action="append",
      default=[],
      help="filter out benchmarks with regexp")
  parser.add_argument(
      "--filter-by-tier",
      type=int,
      action="append",
      default=[],
      help="filter benchmarks by predefined tier 1-3",
  )
  parser.add_argument(
      "--exclude-by-tier",
      type=int,
      action="append",
      default=[],
      help="filter out benchmarks by predefined tier 1-3",
  )

  class LogLevel(Enum):
    critical = logging.CRITICAL
    error = logging.ERROR
    warning = logging.WARNING
    info = logging.INFO
    debug = logging.DEBUG

    @staticmethod
    def parse(s: str):
      try:
        return LogLevel[s]
      except KeyError:
        raise ValueError()

    def __str__(self):
      return self.name

  parser.add_argument(
      "--log-level",
      default=LogLevel.info,
      choices=list(LogLevel),
      type=LogLevel.parse,
      help="Specify log level.",
  )
  parser.add_argument(
      "--accelerator",
      choices=["cpu", "cuda", "tpu"],
      action="append",
      help="Specify an accelerator to use.",
  )
  parser.add_argument(
      "--xla",
      choices=["None", "PJRT", "XRT"],
      action="append",
      help="Specify an xla option to use.",
  )
  parser.add_argument(
      "--dynamo",
      choices=["None", "inductor", "openxla_eval", "openxla"],
      action="append",
      help="Specify an xla option to use.",
  )
  parser.add_argument(
      "--test",
      choices=["eval", "train"],
      action="append",
      help="Specify a test to run.",
  )
  parser.add_argument(
      "--repeat",
      type=int,
      default=10,
      help="Number of times to repeat the timed run in a single experiment.",
  )
  parser.add_argument(
      "--iterations-per-run",
      type=int,
      default=1,
      help="Number of times to repeat the model iteration inside a timed run.",
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      help="""Batch size to be used. If not provided, it depends on the model
      suites to determine it.""",
  )
  parser.add_argument(
      "--total-partitions",
      type=int,
      default=1,
      choices=range(1, 10),
      help="""Total number of partitions we want to divide the benchmark suite
        into""",
  )
  parser.add_argument(
      "--partition-id",
      type=int,
      default=0,
      help="""ID of the benchmark suite partition to be run. Used to divide CI
        tasks""",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Do a dry run to only print the benchmark commands.",
  )
  parser.add_argument(
      "--print-subprocess",
      action="store_true",
      help="Forward subprocess stdout and stderr.",
  )
  parser.add_argument(
      "--subprocess-timeout",
      type=int,
      default=60 * 30,
      help="Timeout per launched config subprocess.",
  )
  parser.add_argument(
      "--progress-bar",
      action="store_true",
      help="Display progress bar.",
  )
  parser.add_argument(
      "--randomize-input",
      action="store_true",
      help="""Whether to randomize the input values. Dimensions will be kept
        the same.""",
  )
  parser.add_argument(
      "--collect-full-output",
      action="store_true",
      help="""Whether to collect full output for training. Set this to true if
        we want to verify the numerical correctness of gradients. But that may
        cause time measurement not accurate""",
  )
  parser.add_argument(
      "--save-output",
      action="store_true",
      help="Whether to save the model output to disk",
  )
  parser.add_argument(
      "--output-dirname",
      type=str,
      default="./output/",
      help="Overrides the directory to place output files.",
  )
  parser.add_argument(
      "--output-basename",
      type=str,
      default="results.jsonl",
      help="Overrides the basename of output files.",
  )
  parser.add_argument(
      "--no-resume",
      action="store_true",
      help="""By default, the runner would skip the finished experiments that
        exist in the output-basename file. If --no-resume is set, the previous
        output-basename file will be deleted and all experiment will run.""",
  )
  parser.add_argument(
      "--dump-hlo",
      action="store_true",
      help="""Dump HLO modules by passing `--xla_dump_to` as `XLA_FLAGS`""",
  )
  parser.add_argument(
      "--dump-dynamo-counters",
      action="store_true",
      help="""Dump dynamo counters.""",
  )
  parser.add_argument(
      "--collect-dynamo-counters",
      action="store_true",
      help="""Collect dynamo counters as part of the regular metrics.""",
  )
  parser.add_argument(
      "--dump-pytorch-profiles",
      action="store_true",
      help="""Dump PyTorch profiles in the output directory. This
        includes CPU/GPU times per operation and Chrome traces. See also
        https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html""",
  )
  parser.add_argument(
      "--dump-pytorch-xla-metrics",
      action="store_true",
      help="""Dump PyTorch/XLA metrics in the output directory. This includes
      compile time and various counters. See also
      https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#get-a-metrics-report""",
  )
  parser.add_argument(
      "--profile-cuda-cpu",
      action="store_true",
      help="""Collect CUDA and CPU times. To dump the entire profile, use
        `--dump-pytorch-profiles`.""",
  )
  parser.add_argument(
      "--profile-cuda-cpu-individual-ops",
      action="store_true",
      help="""Collect CUDA and CPU times per operation. This will also gather
        CPU fallbacks.""",
  )
  parser.add_argument(
      "--xla-flags",
      type=str,
      action="append",
      help="Flags to forward to XLA via `XLA_FLAGS` env var.",
  )
  parser.add_argument(
      "--disable-tf32",
      action="store_true",
      help="Whether to enable fast F32 multiplication in PyTorch.",
  )
  parser.add_argument(
      "--experiment-config",
      type=str,
      help="""JSON string defining the experiment configuration. When set an
        experiment is run with exactly this one configuration.""",
  )
  parser.add_argument(
      "--model-config",
      type=str,
      help="""JSON string defining the model configuration. When set an
        experiment is run with exactly this one configuration.""",
  )
  parser.add_argument(
      "--timestamp",
      default=time.time(),
      type=float,
      help="Timestamp (seconds since the epoch) to assign to the benchmarks.")
  parser.add_argument(
      "--pure-wall-time",
      action="store_true",
      help="Times wall time measurements with pure CUDA events. No kernel launch overhead.",
  )
  parser.add_argument(
      "--filter-by-single-graph",
      action="store_true",
      help="Runs the experiment with hard-failing when it detects there will be multiple graphs out of a single compiled region.",
  )
  parser.add_argument(
      "--verify",
      action="store_true",
      help="""If set, verifies the model output with PT Eager mode, and saves relative error to the output file."""
  )
  parser.add_argument(
      "--no-skip",
      action="store_true",
      help="Do not skip any model.",
  )
  parser.add_argument(
      "--profile-xla",
      action="store_true",
      default=False,
      help="Dumps the XLA profiler traces to the output directory via XPlane. It later can be opened by Tensorboard."
  )
  parser.add_argument(
      '--profile-xla-duration-ms',
      default=5 * 1000,
      type=int,
      help="Defines the duration of the profiling when `profile-xla` flag is set.",
  )
  return parser.parse_args(args)


def main():
  args = parse_args()

  # Expand filter/exclude by tier.
  tiers.append_filter_by_tier(args.filter, args.filter_by_tier)
  tiers.append_filter_by_tier(args.exclude, args.exclude_by_tier)
  args.filter = args.filter or [r"."]
  args.exclude = args.exclude or [r"^$"]

  logging.basicConfig(level=args.log_level.value, force=True)
  logger.debug(f"Parsed args: {args}")

  if not args.disable_tf32:
    logger.warning('Enabling fast F32 multiplication for PyTorch')
    torch.set_float32_matmul_precision('high')

  if args.profile_xla:
    logger.info(
        'Enabling XLA XPlane profiling. Do not benchmark with this option set.')
    server = xp.start_server(9012)

  runner = ExperimentRunner(args)
  runner.run()


if __name__ == "__main__":
  main()

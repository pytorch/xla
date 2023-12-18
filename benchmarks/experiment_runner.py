import argparse
from collections import OrderedDict
import copy
import csv
import io
import json
import logging
import os
import subprocess
import sys
import time
import torch
import tiers
from typing import Optional
import torch_xla.debug.metrics as met
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity
from torch.autograd import DeviceType
from benchmark_model import ModelLoader
from torchbench_model import TorchBenchModelLoader
from benchmark_experiment import ExperimentLoader
from util import reset_rng_state, move_to_device, randomize_input
import torch_xla.core.xla_model as xm

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
        experiment_cfg_wo_env = experiment_cfg.copy()
        process_env = experiment_cfg_wo_env.pop("process_env")
        logger.info(f"Run with --model-config={json.dumps(model_cfg)} "
                    f"--experiment-config={json.dumps(experiment_cfg_wo_env)}")

        # Move on if dry running.
        if self._args.dry_run:
          continue

        # TODO: See if we can pass experiment_cfg to `load_experiment`.
        benchmark_experiment = self.experiment_loader.load_experiment(
            experiment_cfg_wo_env)
        benchmark_model = self.model_loader.load_model(
            model_cfg, benchmark_experiment, dummy=True)

        # Skip already completed benchmark.
        fingerprint = self._get_config_fingerprint(
            benchmark_experiment.to_dict(), benchmark_model.to_dict())
        if fingerprint in skip_fingerprints:
          logger.info(f"SKIP already completed benchmark")
          continue

        # Skip unsupported config.
        if not self.model_loader.is_compatible(benchmark_model,
                                               benchmark_experiment):
          logger.warning("SKIP incompatible model and experiment configs.")
          self.save_results(benchmark_experiment, benchmark_model,
                            {"error": "SKIP"}, None)
          continue

        # Launch subprocess.
        try:
          # TODO: See if we can generalize this for all env vars. The experiment
          # config is currently a bit bloated.
          process_env = benchmark_model.extend_process_env(process_env)
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
          self.save_results(benchmark_experiment, benchmark_model,
                            {"error": str(e)}, None)
        except subprocess.CalledProcessError as e:
          self._fwd_captured_stdout_stderr(e.stdout, e.stderr)
          logger.error("ERROR in subprocess")
          self.save_results(benchmark_experiment, benchmark_model,
                            {"error": e.stderr}, None)
        except subprocess.SubprocessError as e:
          logger.error("ERROR when launching child process")
          self.save_results(benchmark_experiment, benchmark_model,
                            {"error": str(e)}, None)
        except ValueError as e:
          self._fwd_captured_stdout_stderr(e.stdout, e.stderr)
          logger.exception("ERROR")

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

  def run_single_config(self):
    experiment_config = json.loads(self._args.experiment_config)
    model_config = json.loads(self._args.model_config)
    benchmark_experiment = self.experiment_loader.load_experiment(
        experiment_config)
    reset_rng_state(benchmark_experiment)
    benchmark_model = self.model_loader.load_model(model_config,
                                                   benchmark_experiment)

    with benchmark_model.pick_grad():
      metrics = OrderedDict()
      outputs = []
      for repeat_iteration in range(self._args.repeat):
        run_metrics, output = self.timed_run(benchmark_experiment,
                                             benchmark_model, repeat_iteration)
        output = move_to_device(output, 'cpu')
        outputs.append(output)
        for key, val in run_metrics.items():
          if key not in metrics:
            metrics[key] = []
          metrics[key].append(val)

    # Additional experiment metrics can be added here.

    self.save_results(benchmark_experiment, benchmark_model, metrics, outputs)

  def _unique_basename(self, experiment_config: OrderedDict,
                       model_config: OrderedDict) -> str:

    def unique_basename_segment(x, max_len=32):
      s = str(x).replace(" ", "")
      if len(s) > max_len:
        s = str(hex(hash(s)))
      return s

    # Ignore batch_size as it may be altered by the model.
    segments = [
        unique_basename_segment(v)
        for k, v in experiment_config.items()
        if k != "batch_size"
    ] + [unique_basename_segment(v) for k, v in model_config.items()]
    return "-".join(segments)

  def _get_results_file_path(self,
                             experiment_config: OrderedDict,
                             model_config: OrderedDict,
                             partial_name: str,
                             ext: str = "txt",
                             sub_dirname: Optional[str] = None) -> str:
    model_name = model_config["model_name"]
    basename = self._unique_basename(experiment_config, model_config)
    filename = f"{partial_name}-{basename}.{ext}"
    path = os.path.abspath(os.path.join(self._args.output_dirname, model_name))
    if sub_dirname is not None:
      path = os.path.join(path, sub_dirname)
    path = os.path.join(path, filename)

    # Create parent directory.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return path

  def _dump_results_file(self,
                         text: str,
                         experiment_config: OrderedDict,
                         model_config: OrderedDict,
                         partial_name: str,
                         ext: str = "txt",
                         sub_dirname: Optional[str] = None,
                         mode: str = "w"):
    path = self._get_results_file_path(experiment_config, model_config,
                                       partial_name, ext, sub_dirname)
    with open(path, mode) as f:
      f.write(text)

  def save_results(self, benchmark_experiment, benchmark_model, metrics,
                   outputs):
    if self._args.save_output and outputs is not None:
      outputs_file_name = f"{benchmark_model.filename_str}-{benchmark_experiment.filename_str}.pt"
      torch.save(outputs, os.path.join(self.output_dir, outputs_file_name))
    else:
      outputs_file_name = None

    results = OrderedDict()
    results["model"] = benchmark_model.to_dict()
    results["experiment"] = benchmark_experiment.to_dict()
    results["repeat"] = self._args.repeat
    results["iterations_per_run"] = self._args.iterations_per_run

    results["metrics"] = metrics
    results["outputs_file"] = outputs_file_name
    results["timestamp"] = self._args.timestamp

    json_str = json.dumps(results, ensure_ascii=False)
    with open(self.output_file, mode="a", encoding="utf-8") as f:
      f.write(f"{json_str}\n")

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

  def prepare_inputs(self, example_inputs, should_randomize_input):
    inputs_list = []
    for i in range(self._args.iterations_per_run):
      inputs = copy.deepcopy(example_inputs)
      if should_randomize_input:
        inputs = randomize_input(inputs)
      inputs_list.append(inputs)
    return inputs_list

  def dump_profile_info(self, prof, benchmark_model, benchmark_experiment,
                        repeat_iteration: int):
    model_config = benchmark_model.to_dict()
    experiment_config = benchmark_experiment.to_dict()
    assert prof is not None, 'Expecting profiler to be defined!'
    if not self._args.profile_cuda_dump:
      logger.warning(
          'Profiling enabled, but dumping tracing/kernel summary disabled.')
      return

    # Dump pytorch trace.
    prof.export_chrome_trace(
        self._get_results_file_path(
            experiment_config,
            model_config,
            "trace",
            ext="json",
            sub_dirname=str(repeat_iteration)))

    # Dump pytorch profile.
    pytorch_profile = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=500)
    self._dump_results_file(
        pytorch_profile,
        experiment_config,
        model_config,
        "pytorch-profile",
        sub_dirname=str(repeat_iteration))
    self._dump_results_file(
        pytorch_profile,
        experiment_config,
        model_config,
        "pytorch-profile",
        mode="a")

  def collect_profile_to_metrics(self, prof, metrics):
    assert prof is not None, 'Expecting profiler to be defined!'
    if not self._args.profile_cuda_cpu_collect:
      logger.warning(
          'Profiling enabled, but collection of CPU/CUDA profiling info disabled.'
      )
      return

    kernel_dump = prof.profiler.total_average()
    total_cuda_time = 0
    total_cpu_time = kernel_dump.self_cpu_time_total

    # Avoid double counting CUDA time for inductor. Copied here, since the interface is not really exposed via any interface.
    # The alternative is regex matching resulting string dump for CUDA kernel time.
    # Source: https://github.com/pytorch/pytorch/blob/2f3beb715c608a060934c237de402faa40ea211f/torch/autograd/profiler_util.py#L1025-L1037
    for evt in prof.profiler.key_averages():
      if evt.device_type == DeviceType.CPU:
        # in legacy profiler, kernel info is stored in cpu events
        if evt.is_legacy:
          total_cuda_time += evt.self_cuda_time_total
      elif evt.device_type == DeviceType.CUDA:
        # in kineto profiler, there're events with the correct device type (e.g. CUDA)
        total_cuda_time += evt.self_cuda_time_total

    total_cpu_time /= 1000000
    total_cuda_time /= 1000000
    metrics["total_cpu_time_s"] = total_cpu_time
    metrics["total_cuda_time_s"] = total_cuda_time
    metrics[
        "per_iter_cpu_time_s"] = total_cpu_time / self._args.iterations_per_run
    metrics[
        "per_iter_cuda_time_s"] = total_cuda_time / self._args.iterations_per_run

  def get_xla_cpu_fallback_ops(self, met):
    return set(name for name in met.counter_names() if self.is_aten_op(name))

  def is_aten_op(self, op_name):
    return 'aten::' in op_name

  def collect_individual_ops(self, benchmark_experiment, metrics, prof):
    assert prof is not None, 'Expecting prof to be defined!'

    us_to_s = lambda x: x / 1000000
    extract_prof_info = lambda event: {
        "self_cpu_time_s": us_to_s(event.self_cpu_time_total),
        "self_cuda_time_s": us_to_s(event.self_cuda_time_total),
        "total_cpu_time_s": us_to_s(event.cpu_time_total),
        "total_cuda_time_s": us_to_s(event.cuda_time_total),
        "num_of_calls": event.count
    }

    if benchmark_experiment.xla:
      unlowered_ops = self.get_xla_cpu_fallback_ops(met)
      if not unlowered_ops:
        return
      if "xla_unlowered_ops" not in metrics:
        metrics["xla_unlowered_ops"] = dict()
      for event in prof.key_averages():
        if event.key in unlowered_ops:
          metrics["xla_unlowered_ops"][event.key] = extract_prof_info(event)
    else:
      for event in prof.key_averages():
        op_name = event.key
        if not self.is_aten_op(op_name):
          continue
        if "inductor_ops" not in metrics:
          metrics["inductor_ops"] = dict()
        metrics["inductor_ops"][op_name] = extract_prof_info(event)

  def timed_run(self, benchmark_experiment, benchmark_model,
                repeat_iteration: int):
    reset_rng_state(benchmark_experiment)

    inputs_list = self.prepare_inputs(benchmark_model.example_inputs,
                                      self._args.randomize_input)

    reset_rng_state(benchmark_experiment)
    self._mark_step(benchmark_experiment)
    self._synchronize(benchmark_experiment)

    # Clear XLA metrics before executing the model.
    met.clear_metrics()

    enable_prof = self._args.profile_cuda
    metrics = OrderedDict()
    t_start = time.perf_counter()
    if benchmark_experiment.xla:
      t_trace = 0

    def loop(prof=None):
      nonlocal t_trace
      for i in range(self._args.iterations_per_run):
        if benchmark_experiment.xla:
          t_trace_start = time.perf_counter()

        output = benchmark_model.model_iter_fn(
            inputs_list[i], collect_full_output=self._args.collect_full_output)

        if benchmark_experiment.xla:
          t_trace += time.perf_counter() - t_trace_start

        self._mark_step(benchmark_experiment)

        if prof:
          prof.step()
      self._synchronize(benchmark_experiment)
      return output

    if enable_prof:
      with profile(
          activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        output = loop(prof)
    else:
      output = loop()

    t_end = time.perf_counter()
    if enable_prof:
      self.dump_profile_info(prof, benchmark_model, benchmark_experiment,
                             repeat_iteration)
      self.collect_profile_to_metrics(prof, metrics)

    metrics["total_time"] = t_end - t_start
    metrics[
        "per_iter_time"] = metrics["total_time"] / self._args.iterations_per_run

    if benchmark_experiment.xla:
      metrics["trace_per_iter_time"] = t_trace / self._args.iterations_per_run

      def ns_to_s(ns):
        return ns * 1e-9

      for m in ("CompileTime", "ExecuteTime"):
        data = met.metric_data(m)
        data = data if data is not None else (0, 0, [])
        number, total_time, _ = data
        # Time is measured in nano-seconds
        metrics[f"xla_{m}_time_s"] = ns_to_s(total_time)
        metrics[f"xla_{m}_number"] = number

    if enable_prof:
      self.collect_individual_ops(benchmark_experiment, metrics, prof)

    return metrics, output


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

  def _parse_log_level(level: str):
    level = level.lower()
    if level == "critical":
      return logging.CRITICAL
    elif level == "error":
      return logging.ERROR
    elif level == "warning":
      return logging.WARNING
    elif level == "info":
      return logging.INFO
    elif level == "debug":
      return logging.DEBUG
    else:
      raise NotImplementedError

  parser.add_argument(
      "--log-level",
      default=logging.INFO,
      choices=[
          logging.CRITICAL,
          logging.ERROR,
          logging.WARNING,
          logging.INFO,
          logging.DEBUG,
      ],
      type=_parse_log_level,
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
      help="Total number of partitions we want to divide the benchmark suite into",
  )
  parser.add_argument(
      "--partition-id",
      type=int,
      default=0,
      help="ID of the benchmark suite partition to be run. Used to divide CI tasks",
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
      "--profile-cuda",
      action="store_true",
      help="""Whether to profile CUDA or not. Note this does not do much except
        for triggering a profiler. To get the profiling data use additionally
        --profile-cuda-dump""",
  )
  parser.add_argument(
      "--profile-cuda-dump",
      type=str,
      default="",
      help="Directory specifying where to dump profiling information (summary, and trace)",
  )
  parser.add_argument(
      "--profile-cuda-cpu-collect",
      action="store_true",
      help="Whether to collect CPU/GPU profiling information in the resulting file.",
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
      default=False,
      help="Whether to enable fast F32 multiplication in PyTorch.",
  )
  parser.add_argument(
      "--experiment-config",
      type=str,
      help="JSON string defining the experiment configuration. When set an experiment is run with exactly this one configuration.",
  )
  parser.add_argument(
      "--model-config",
      type=str,
      help="JSON string defining the model configuration. When set an experiment is run with exactly this one configuration.",
  )

  parser.add_argument(
      "--timestamp",
      default=time.time(),
      type=float,
      help="Timestamp (seconds since the epoch) to assign to the benchmarks.")

  return parser.parse_args(args)


def main():
  args = parse_args()

  # Expand filter/exclude by tier.
  tiers.append_filter_by_tier(args.filter, args.filter_by_tier)
  tiers.append_filter_by_tier(args.exclude, args.exclude_by_tier)
  args.filter = args.filter or [r"."]
  args.exclude = args.exclude or [r"^$"]

  logging.basicConfig(level=args.log_level, force=True)
  logger.debug(f"Parsed args: {args}")

  if not args.disable_tf32:
    logger.warning('Enabling fast F32 multiplication for PyTorch')
    torch.set_float32_matmul_precision('high')

  runner = ExperimentRunner(args)
  runner.run()


if __name__ == "__main__":
  main()

import argparse
from collections import OrderedDict
import copy
import csv
import io
import json
import logging
import numpy as np
import os
import subprocess
import sys
import time
import torch
import types

try:
  from .benchmark_model import ModelLoader
  from .torchbench_model import TorchBenchModelLoader
  from .benchmark_experiment import ExperimentLoader
  from .util import patch_torch_manual_seed, reset_rng_state, move_to_device, randomize_input
except ImportError:
  from benchmark_model import ModelLoader
  from torchbench_model import TorchBenchModelLoader
  from benchmark_experiment import ExperimentLoader
  from util import patch_torch_manual_seed, reset_rng_state, move_to_device, randomize_input

logger = logging.getLogger(__name__)


class ExperimentRunner:

  def __init__(self, args):
    self._args = args
    self.suite_name = self._args.suite_name

    self.experiment_loader = ExperimentLoader(self._args)

    if self.suite_name == "torchbench":
      self.model_loader = TorchBenchModelLoader(self._args)
    elif self.suite_name == "dummy":
      self.model_loader = ModelLoader(self._args)
    else:
      raise NotImplementedError

    self.output_dir = os.path.abspath(self._args.output_dirname)
    os.makedirs(self.output_dir, exist_ok=True)
    self.output_file = os.path.join(self.output_dir, self._args.output_basename)

  def run(self):
    if self._args.experiment_config and self._args.model_config:
      if self._args.dry_run:
        logger.info(f"Dry run with {[sys.executable] + sys.argv}")
        return
      experiment_config = json.loads(self._args.experiment_config)
      model_config = json.loads(self._args.model_config)
      self.run_single_experiment(experiment_config, model_config)
    else:
      assert not self._args.experiment_config and not self._args.model_config
      if os.path.exists(self.output_file):
        os.unlink(self.output_file)

      experiment_configs = self.experiment_loader.list_experiment_configs()
      model_configs = self.model_loader.list_model_configs()
      for model_config in model_configs:
        for experiment_config in experiment_configs:
          if self.model_loader.is_compatible(model_config, experiment_config):
            process_env = experiment_config.pop("process_env")
            experiment_config_str = json.dumps(experiment_config)
            model_config_str = json.dumps(model_config)
            experiment_config["process_env"] = process_env
            command = ([sys.executable] + sys.argv +
                       [f"--experiment-config={experiment_config_str}"] +
                       [f"--model-config={model_config_str}"])
            if self._args.dry_run:
              logger.info(f"Dry run with {command}")
              continue
            try:
              subprocess.check_call(
                  command,
                  timeout=60 * 20,
                  env=process_env,
              )
            except subprocess.TimeoutExpired as e:
              logger.error("TIMEOUT")
              self.record_failed_experiment(model_config_str, experiment_config_str, e)
            except subprocess.SubprocessError as e:
              logger.error("ERROR")
              self.record_failed_experiment(model_config_str, experiment_config_str, e)

          else:
            logger.warning("SKIP because of incompatible configs.")

  def run_single_experiment(self, experiment_config, model_config):
    benchmark_experiment = self.experiment_loader.load_experiment(
        experiment_config
    )
    reset_rng_state(benchmark_experiment)
    benchmark_model = self.model_loader.load_model(
        model_config, benchmark_experiment
    )

    timings = OrderedDict()
    results = []
    for i in range(self._args.repeat):
      timing, result = self.timed_run(
          benchmark_experiment, benchmark_model
      )
      result = move_to_device(result, 'cpu')
      results.append(result)
      for key, val in timing.items():
        if i == 0:
          timings[key] = np.zeros(self._args.repeat, np.float64)
        timings[key][i] = val
    # print(timings)

    self.save_results(benchmark_experiment, benchmark_model, timings, results)

  def save_results(self, benchmark_experiment, benchmark_model, timings, results):
    detail_file_name = f"{benchmark_model.filename_str}-{benchmark_experiment.filename_str}.pt"
    csv_headers = [
        "suite_name",
        "model_name",
        "accelerator",
        "xla",
        "test",
        "batch_size",
        "median_total",
        "median_average",
        "detail_results",
    ]

    csv_row = [
        benchmark_model.suite_name,
        benchmark_model.model_name,
        benchmark_experiment.accelerator,
        benchmark_experiment.xla,
        benchmark_experiment.test,
        benchmark_experiment.batch_size,
        np.median(timings["total"]).item(),
        np.median(timings["average"]).item(),
        detail_file_name,
    ]

    self.output_csv(csv_headers, csv_row)

    torch.save({"timings": timings, "results": results},
               os.path.join(self.output_dir, detail_file_name))

  def record_failed_experiment(self, model_config_str, experiment_config_str, e):
    headers = ["model_config", "experiment_config", "failure"]
    row = [model_config_str, experiment_config_str, e]
    file_path = os.path.join(self.output_dir, "failed_experiments.csv")
    self.output_csv(headers, row, file_path)

  def output_csv(self, headers, row, file_path=None):
    if not file_path:
      file_path = self.output_file
    existed = os.path.exists(file_path)
    output = csv.writer(
        io.TextIOWrapper(
            open(file_path, "ab", buffering=0),
            "utf-8",
            write_through=True,
        ),
        lineterminator="\n",
    )
    if not existed:
      output.writerow(headers)
    output.writerow([(f"{x:.8e}" if isinstance(x, float) else x) for x in row])

  def _mark_step(self, benchmark_experiment):
    if benchmark_experiment.xla:
      import torch_xla.core.xla_model as xm
      xm.mark_step()

  def _synchronize(self, benchmark_experiment):
    if benchmark_experiment.xla:
      import torch_xla.core.xla_model as xm
      xm.wait_device_ops()
    elif benchmark_experiment.accelerator == "gpu":
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

  def timed_run(self, benchmark_experiment, benchmark_model):
    reset_rng_state(benchmark_experiment)

    inputs_list = self.prepare_inputs(benchmark_model.example_inputs, self._args.randomize_input)

    reset_rng_state(benchmark_experiment)
    self._mark_step(benchmark_experiment)
    self._synchronize(benchmark_experiment)

    timing = OrderedDict()
    t_start = time.perf_counter()

    for i in range(self._args.iterations_per_run):
      result = benchmark_model.model_iter_fn(inputs_list[i], collect_full_result=self._args.collect_full_result)

      if benchmark_experiment.xla and self._args.iterations_per_run == 1:
        t_trace = time.perf_counter()

      self._mark_step(benchmark_experiment)

    self._synchronize(benchmark_experiment)

    t_end = time.perf_counter()

    timing["total"] = t_end - t_start
    timing["average"] = timing["total"] / self._args.iterations_per_run
    if benchmark_experiment.xla and self._args.iterations_per_run == 1:
      timing["trace"] = t_trace - t_start

    return timing, result


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suite-name",
        required=True,
        choices=["dummy", "torchbench"],
        help="Suite name for the model garden.",
    )

    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude", "-x", action="append", help="filter benchmarks with regexp"
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
        help="Batch size to be used. If not provided, it depends on the model suites to determine it.",
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
        "--randomize-input",
        action="store_true",
        help="Whether to randomize the input values. Dimensions will be kept the same.",
    )

    parser.add_argument(
        "--collect-full-result",
        action="store_true",
        help="""Whether to collect full result for training. Set this to true if we
        want to verify the numerical correctness of graidents. But that may
        cause time measurement not accurate""",
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
        default="results.csv",
        help="Overrides the basename of output files.",
    )

    parser.add_argument(
        "--experiment-config",
        type=str,
        help="JSON string of the experiment config dict.",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        help="JSON string of the model config dict.",
    )

    return parser.parse_args(args)


def main():
  args = parse_args()

  args.filter = args.filter or [r"."]
  args.exclude = args.exclude or [r"^$"]

  logger.info(args)
  runner = ExperimentRunner(args)
  runner.run()


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, force=True)
  main()
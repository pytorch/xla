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
from tqdm import tqdm

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

try:
  import torch_xla.core.xla_model as xm
except ImportError:
  # ignore the error if torch_xla is not installed
  pass

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
        logger.warning(f"Dry run with {[sys.executable] + sys.argv}")
        return
      experiment_config = json.loads(self._args.experiment_config)
      model_config = json.loads(self._args.model_config)
      self.run_single_experiment(experiment_config, model_config)
    else:
      assert not self._args.experiment_config and not self._args.model_config
      finished_experiments = set()
      if os.path.exists(self.output_file):
        if self._args.no_resume:
          os.unlink(self.output_file)
        else:
          with open(self.output_file, mode="r", encoding="utf-8") as f:
            jsonlines = f.read().splitlines()
          for jsonline in jsonlines:
            tmp = json.loads(jsonline)
            if self._args.experiment_name == "run_all":
              # the finished experiment batch_size may be altered by model set_up(),
              # so the dummy experiment will not match it
              tmp["experiment"]["batch_size"] = self._args.batch_size
            finished_experiments.add("-".join(
                str(item) for item in (list(tmp["model"].values()) +
                                       list(tmp["experiment"].values()))))

      experiment_configs = self.experiment_loader.list_experiment_configs()
      model_configs = self.model_loader.list_model_configs()
      logger.warning(
          f"Number of selected experiment configs: {len(experiment_configs)}")
      logger.warning(f"Number of selected model configs: {len(model_configs)}")
      for model_config in tqdm(
          model_configs,
          desc="model configs",
          disable=not self._args.progress_bar):
        for experiment_config in experiment_configs:
          process_env = experiment_config.pop("process_env")
          experiment_config_str = json.dumps(experiment_config)
          model_config_str = json.dumps(model_config)
          dummy_benchmark_experiment = self.experiment_loader.load_experiment(
              experiment_config, dummy=True)
          dummy_benchmark_model = self.model_loader.load_model(
              model_config, dummy_benchmark_experiment, dummy=True)
          experiment_config["process_env"] = process_env
          command = ([sys.executable] + sys.argv +
                     [f"--experiment-config={experiment_config_str}"] +
                     [f"--model-config={model_config_str}"])
          if self._args.dry_run:
            logger.warning(f"Dry run with {command}")
            continue
          if "-".join(
              str(item)
              for item in (list(dummy_benchmark_model.to_dict().values()) +
                           list(dummy_benchmark_experiment.to_dict().values())
                          )) in finished_experiments:
            continue
          if self.model_loader.is_compatible(dummy_benchmark_model,
                                             dummy_benchmark_experiment):
            try:
              completed_process = subprocess.run(
                  command,
                  timeout=60 * 20,
                  env=process_env,
                  check=True,
                  capture_output=True,
                  encoding="utf-8",
              )
            except subprocess.TimeoutExpired as e:
              logger.error("TIMEOUT")
              self.save_results(dummy_benchmark_experiment,
                                dummy_benchmark_model, {"error": str(e)}, None)
            except subprocess.CalledProcessError as e:
              logger.error("ERROR")
              self.save_results(dummy_benchmark_experiment,
                                dummy_benchmark_model, {"error": e.stderr},
                                None)
            except subprocess.SubprocessError as e:
              logger.error("ERROR")
              self.save_results(dummy_benchmark_experiment,
                                dummy_benchmark_model, {"error": str(e)}, None)
            else:
              if self._args.print_subprocess:
                logger.info(completed_process.stdout)
                logger.warning(completed_process.stderr)

          else:
            e = "SKIP because of incompatible model and experiment configs."
            logger.warning(e)
            self.save_results(dummy_benchmark_experiment, dummy_benchmark_model,
                              {"error": str(e)}, None)

  def run_single_experiment(self, experiment_config, model_config):
    benchmark_experiment = self.experiment_loader.load_experiment(
        experiment_config)
    reset_rng_state(benchmark_experiment)
    benchmark_model = self.model_loader.load_model(model_config,
                                                   benchmark_experiment)

    with benchmark_model.pick_grad():
      metrics = OrderedDict()
      outputs = []
      for i in range(self._args.repeat):
        run_metrics, output = self.timed_run(benchmark_experiment,
                                             benchmark_model)
        output = move_to_device(output, 'cpu')
        outputs.append(output)
        for key, val in run_metrics.items():
          # metrics from repeated runs are formed into lists in the metrics dict
          if i == 0:
            metrics[key] = []
          metrics[key].append(val)

    # additional experiment metrics can be added here

    self.save_results(benchmark_experiment, benchmark_model, metrics, outputs)

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

    self.output_jsonl(results)

  def output_jsonl(self, obj, file_path=None):
    if not file_path:
      file_path = self.output_file
    json_str = json.dumps(obj, ensure_ascii=False)
    with open(file_path, mode="a", encoding="utf-8") as f:
      f.write(f"{json_str}\n")

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
      xm.mark_step()

  def _synchronize(self, benchmark_experiment):
    if benchmark_experiment.xla:
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

    inputs_list = self.prepare_inputs(benchmark_model.example_inputs,
                                      self._args.randomize_input)

    reset_rng_state(benchmark_experiment)
    self._mark_step(benchmark_experiment)
    self._synchronize(benchmark_experiment)

    metrics = OrderedDict()
    t_start = time.perf_counter()
    if benchmark_experiment.xla:
      t_trace = 0

    for i in range(self._args.iterations_per_run):
      if benchmark_experiment.xla:
        t_trace_start = time.perf_counter()

      output = benchmark_model.model_iter_fn(
          inputs_list[i], collect_full_output=self._args.collect_full_output)

      if benchmark_experiment.xla:
        t_trace += time.perf_counter() - t_trace_start

      self._mark_step(benchmark_experiment)

    self._synchronize(benchmark_experiment)

    t_end = time.perf_counter()

    metrics["total_time"] = t_end - t_start
    metrics[
        "per_iter_time"] = metrics["total_time"] / self._args.iterations_per_run
    if benchmark_experiment.xla:
      metrics["trace_per_iter_time"] = t_trace / self._args.iterations_per_run

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
      "--filter", "-k", action="append", help="filter benchmarks with regexp")
  parser.add_argument(
      "--exclude", "-x", action="append", help="filter benchmarks with regexp")

  parser.add_argument(
      "--log-level",
      default="warning",
      choices=["info", "warning"],
      help="Specify the logging level.",
  )

  parser.add_argument(
      "--experiment-name",
      default="run_all",
      choices=["run_all"],
      help="Experiment name to run.",
  )

  parser.add_argument(
      "--accelerator",
      choices=["cpu", "gpu", "tpu"],
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
      choices=[
          "None", "inductor", "torchxla_trace_once", "aot_torchxla_trace_once"
      ],
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
      "--print-subprocess",
      action="store_true",
      help="Print subprocess stdout.",
  )

  parser.add_argument(
      "--progress-bar",
      action="store_true",
      help="Display progress bar.",
  )

  parser.add_argument(
      "--randomize-input",
      action="store_true",
      help="Whether to randomize the input values. Dimensions will be kept the same.",
  )

  parser.add_argument(
      "--collect-full-output",
      action="store_true",
      help="""Whether to collect full output for training. Set this to true if we
        want to verify the numerical correctness of graidents. But that may
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
        output-basename file will be deleted and all experiment will run""",
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

  if args.log_level == "info":
    log_level = logging.INFO
  elif args.log_level == "warning":
    log_level = logging.WARNING
  else:
    log_level = None
  logging.basicConfig(level=log_level, force=True)

  logger.info(args)
  runner = ExperimentRunner(args)
  runner.run()


if __name__ == "__main__":
  main()

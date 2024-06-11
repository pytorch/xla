import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import time

logger = logging.getLogger(__name__)


class ResultAnalyzer:

  def __init__(self, args):
    self._args = args
    self.timestamp = self._args.timestamp or time.time()
    self.output_dir = os.path.abspath(self._args.output_dirname)
    if not os.path.exists(self.output_dir):
      raise ValueError("The output directory does not exist.")
    self.output_file = os.path.join(self.output_dir,
                                    "metric_report." + self._args.output_format)

    self.database = os.path.abspath(self._args.database)

  def run_jsonl(self):
    jsonl_files = []
    for file in os.listdir(self.output_dir):
      if file.endswith(".jsonl"):
        jsonl_files.append(os.path.join(self.output_dir, file))

    all_test_runs = []
    for file in jsonl_files:
      test_runs = self.extract_metrics_jsonl(file)
      all_test_runs.extend(test_runs)

    with open(self.output_file, 'w+') as f:
      for test_run in all_test_runs:
        f.write(json.dumps(test_run))
        f.write("\n")

    print(f"Saving report to: {self.output_file}")

  def run_csv(self):
    jsonl_files = []
    for file in os.listdir(self.output_dir):
      if file.endswith(".jsonl"):
        jsonl_files.append(os.path.join(self.output_dir, file))

    metric_df = pd.DataFrame({
        "timestamp": pd.Series(dtype="int"),
        "suite_name": pd.Series(dtype="str"),
        "model_name": pd.Series(dtype="str"),
        "accelerator": pd.Series(dtype="str"),
        "accelerator_model": pd.Series(dtype="str"),
        "xla": pd.Series(dtype="str"),
        "xla_flags": pd.Series(dtype="str"),
        "dynamo": pd.Series(dtype="str"),
        "torch_xla2": pd.Series(dtype="str"),
        "keep_model_data_on_cuda": pd.Series(dtype="bool"),
        "test": pd.Series(dtype="str"),
        "batch_size": pd.Series(dtype="int"),
        "repeat": pd.Series(dtype="int"),
        "iterations_per_run": pd.Series(dtype="int"),
        "error_message": pd.Series(dtype="str"),
        "median_total_time": pd.Series(dtype="float"),
        "median_per_iter_time": pd.Series(dtype="float"),
        "xla_median_trace_per_iter_time": pd.Series(dtype="float"),
        "xla_compile_time": pd.Series(dtype="float"),
        "dynamo_compile_time": pd.Series(dtype="float"),
        "outputs_file": pd.Series(dtype="str"),
    })
    for file in jsonl_files:
      metric_df = self.extract_metrics_csv(file, metric_df)
    # additional processing of the metric_df can be done here
    self.export_metric_report(metric_df)

  def get_calculated_metrics(self, d, dataline):
    MAX_TOTAL_TIME = f"{np.max.__name__}_total_time"
    MEDIAN_TOTAL_TIME = f"{np.median.__name__}_total_time"

    for metric, raw_values in dataline["metrics"].items():
      values = np.asarray(raw_values, dtype="float")

      is_valid = (
          dataline["experiment"]["xla"] or metric != "trace_per_iter_time")

      for fn in (np.min, np.median, np.max):
        d[f"{fn.__name__}_{metric}"] = fn(values) if is_valid else -1

      # Remove first measurement.
      # Assumption: the first measurement has tracing + compilation times
      # embedded into it. Therefore, we remove it from our data for computing
      # the average and standard deviation.
      skip_head = values[1:]

      if len(skip_head) > 0:
        for fn in (np.mean, np.std):
          d[f"{fn.__name__}_{metric}"] = fn(skip_head) if is_valid else -1

    compile_time = d[MAX_TOTAL_TIME] - d[MEDIAN_TOTAL_TIME]
    d["dynamo_compile_time"] = compile_time if dataline["experiment"][
        "dynamo"] else -1
    d["xla_compile_time"] = compile_time if dataline["experiment"]["xla"] else -1
    return d

  # TODO: handle error message properly (database length restriction)
  # Do not use bool. This will mess up with the bigquery parsing.
  def extract_metrics_jsonl(self, file):
    with open(file, mode="r", encoding="utf-8") as f:
      jsonlines = f.read().splitlines()
    runs = []
    for jsonline in jsonlines:
      dataline = json.loads(jsonline)
      timestamp = dataline[
          "timestamp"] if "timestamp" in dataline else self.timestamp
      batch_size = dataline["experiment"]["batch_size"]
      batch_side_value = -1 if batch_size is None else batch_size
      xla = dataline["experiment"]["xla"]
      xla_value = "None" if xla is None else xla
      dynamo = dataline["experiment"]["dynamo"]
      dynamo_value = "None" if dynamo is None else dynamo
      torch_xla2 = dataline["experiment"]["torch_xla2"]
      torch_xla2_value = "None" if torch_xla2 is None else torch_xla2
      keep_model_data_on_cuda = dataline["experiment"][
          "keep_model_data_on_cuda"]
      keep_model_data_on_cuda_value = "None" if keep_model_data_on_cuda is None else str(
          keep_model_data_on_cuda)
      test = dataline["experiment"]["test"]
      test_value = "None" if test is None else test
      outputs_file = dataline["experiment"].get("outputs_file", None)
      outputs_file_value = "None" if outputs_file is None else outputs_file

      d = {
          "metrics": {
              "timestamp": int(timestamp),
              "batch_size": batch_side_value,
              "repeat": dataline["repeat"],
              "iterations_per_run": dataline["iterations_per_run"]
          },
          "dimensions": {
              "suite_name": dataline["model"]["suite_name"],
              "model_name": dataline["model"]["model_name"],
              "accelerator": dataline["experiment"]["accelerator_model"],
              "accelerator_model": dataline["experiment"]["accelerator_model"],
              "xla": xla_value,
              "dynamo": dynamo_value,
              "torch_xla2": torch_xla2_value,
              "keep_model_data_on_cuda": keep_model_data_on_cuda_value,
              "test": test_value,
              "outputs_file": outputs_file_value
          }
      }

      if "error" in dataline["metrics"] and not self._args.hide_errors:
        d["error_message"] = dataline["metrics"]["error"]

      if "error" not in dataline["metrics"]:
        d["dimensions"]["run_status"] = "success"
        d["metrics"] = self.get_calculated_metrics(d["metrics"], dataline)
      else:
        d["dimensions"]["run_status"] = "failure"
        d["metrics"]["median_total_time"] = -1
        d["metrics"]["median_per_iter_time"] = -1
        d["metrics"]["xla_median_trace_per_iter_time"] = -1
        d["metrics"]["xla_compile_time"] = -1
        d["metrics"]["dynamo_compile_time"] = -1

      runs.append(d)

    return runs

  def extract_metrics_csv(self, file, metric_df):
    with open(file, mode="r", encoding="utf-8") as f:
      jsonlines = f.read().splitlines()

    for jsonline in jsonlines:
      dataline = json.loads(jsonline)
      timestamp = dataline[
          "timestamp"] if "timestamp" in dataline else self.timestamp
      d = {
          "timestamp":
              timestamp,
          "suite_name":
              dataline["model"]["suite_name"],
          "model_name":
              dataline["model"]["model_name"],
          "accelerator":
              dataline["experiment"]["accelerator"],
          "accelerator_model":
              dataline["experiment"]["accelerator_model"],
          "xla":
              dataline["experiment"]["xla"],
          "xla_flags":
              dataline["experiment"]["xla_flags"],
          "dynamo":
              dataline["experiment"]["dynamo"],
          "torch_xla2":
              dataline["experiment"]["torch_xla2"],
          "keep_model_data_on_cuda":
              dataline["experiment"]["keep_model_data_on_cuda"],
          "test":
              dataline["experiment"]["test"],
          "batch_size":
              dataline["experiment"]["batch_size"],
          "repeat":
              dataline["repeat"],
          "iterations_per_run":
              dataline["iterations_per_run"],
          "error_message":
              None,
          "outputs_file":
              dataline["experiment"].get("outputs_file", ""),
      }

      if "error" in dataline["metrics"] and not self._args.hide_errors:
        d["error_message"] = dataline["metrics"]["error"]

      if "error" not in dataline["metrics"]:
        d = self.get_calculated_metrics(d, dataline)

      new_row = pd.Series(d)
      new_row.fillna(value=np.nan, inplace=True)
      metric_df = pd.concat([metric_df, new_row.to_frame().T],
                            ignore_index=True)

    return metric_df

  def export_metric_report(self, metric_df):
    metric_df.to_csv(
        self.output_file, mode="w", encoding="utf-8", header=True, index=False)

    if not os.path.exists(self.database):
      metric_df.to_csv(
          self.database, mode="w", encoding="utf-8", header=True, index=False)
    else:
      metric_df.to_csv(
          self.database, mode="a", encoding="utf-8", header=False, index=False)

  def run(self):
    if self._args.output_format == "jsonl":
      self.run_jsonl()
    elif self._args.output_format == "csv":
      self.run_csv()
    else:
      raise ValueError(f"Unsupported output format: {self._args.output_format}")


def parse_args(args=None):
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--output-format",
      default="csv",
      type=str,
      choices=["jsonl", "csv"],
      help="Specify the output format.",
  )

  parser.add_argument(
      "--log-level",
      default="warning",
      type=str,
      choices=["info", "warning"],
      help="Specify the logging level.",
  )

  parser.add_argument(
      "--experiment-name",
      default="run_all",
      type=str,
      choices=["run_all"],
      help="Experiment name to run.",
  )

  parser.add_argument(
      "--output-dirname",
      type=str,
      default="./output/",
      help="Overrides the directory to place output files.",
  )

  parser.add_argument(
      "--database",
      type=str,
      default="./output/database.csv",
      help="Path to the database.",  # for POC, database is a path to a csv file.
  )

  parser.add_argument(
      "--timestamp",
      type=float,
      help="User provided timestamp used if the input data does not have it.",
  )

  parser.add_argument(
      "--hide-errors",
      default=False,
      action="store_true",
      help="Hide errors to make the CSV more readable",
  )

  return parser.parse_args(args)


def main():
  args = parse_args()

  if args.log_level == "info":
    log_level = logging.INFO
  elif args.log_level == "warning":
    log_level = logging.WARNING
  else:
    log_level = None
  logging.basicConfig(level=log_level, force=True)

  logger.info(args)
  analyzer = ResultAnalyzer(args)
  analyzer.run()


if __name__ == "__main__":
  main()

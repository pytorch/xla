import argparse
from collections import OrderedDict
import copy
import csv
import io
import json
import logging
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import time
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ResultAnalyzer:

  def __init__(self, args):
    self._args = args
    self.timestamp = self._args.timestamp or time.time()
    self.output_dir = os.path.abspath(self._args.output_dirname)
    if not os.path.exists(self.output_dir):
      raise ValueError("The output directory does not exist.")
    self.output_file = os.path.join(self.output_dir, "metric_report.csv")

    self.database = os.path.abspath(self._args.database)

  def run(self):
    jsonl_files = []
    for file in os.listdir(self.output_dir):
      if file.endswith(".jsonl"):
        jsonl_files.append(os.path.join(self.output_dir, file))

    metric_df = pd.DataFrame({"timestamp": pd.Series(dtype="int"),
                              "suite_name": pd.Series(dtype="str"),
                              "model_name": pd.Series(dtype="str"),
                              "experiment_name": pd.Series(dtype="str"),
                              "accelerator": pd.Series(dtype="str"),
                              "accelerator_model": pd.Series(dtype="str"),
                              "xla": pd.Series(dtype="str"),
                              "dynamo": pd.Series(dtype="str"),
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
      metric_df = self.extract_metrics(file, metric_df)

    # additional processing of the metric_df can be done here

    self.export_metric_report(metric_df)

  def extract_metrics(self, file, metric_df):
    with open(file, mode="r", encoding="utf-8") as f:
      jsonlines = f.read().splitlines()

    for jsonline in jsonlines:
      tmp = json.loads(jsonline)
      d = {"timestamp": self.timestamp,
           "suite_name": tmp["model"]["suite_name"],
           "model_name": tmp["model"]["model_name"],
           "experiment_name": tmp["experiment"]["experiment_name"],
           "accelerator": tmp["experiment"]["accelerator"],
           "accelerator_model": tmp["experiment"]["accelerator_model"],
           "xla": tmp["experiment"]["xla"],
           "dynamo": tmp["experiment"]["dynamo"],
           "test": tmp["experiment"]["test"],
           "batch_size": tmp["experiment"]["batch_size"],
           "repeat": tmp["repeat"],
           "iterations_per_run": tmp["iterations_per_run"],
           "error_message": tmp["metrics"].get("error", None),
           "outputs_file": tmp["outputs_file"],
          }
      if "error" not in tmp["metrics"]:
        total_time = np.asarray(tmp["metrics"]["total_time"], dtype="float")
        d["median_total_time"] = np.median(total_time)
        per_iter_time = np.asarray(tmp["metrics"]["per_iter_time"], dtype="float")
        d["median_per_iter_time"] = np.median(per_iter_time)
        if tmp["experiment"]["xla"]:
          trace_per_iter_time = np.asarray(tmp["metrics"]["trace_per_iter_time"], dtype="float")
          d["xla_median_trace_per_iter_time"] = np.median(trace_per_iter_time)
          d["xla_compile_time"] = np.max(total_time) - np.median(total_time)
        if tmp["experiment"]["dynamo"]:
          d["dynamo_compile_time"] = np.max(total_time) - np.median(total_time)

      new_row = pd.Series(d)
      new_row.fillna(value=np.nan, inplace=True)
      metric_df = pd.concat([metric_df, new_row.to_frame().T], ignore_index=True)

    return metric_df

  def export_metric_report(self, metric_df):
    metric_df.to_csv(self.output_file, mode="w", encoding="utf-8", header=True, index=False)

    if not os.path.exists(self.database):
      metric_df.to_csv(self.database, mode="w", encoding="utf-8", header=True, index=False)
    else:
      metric_df.to_csv(self.database, mode="a", encoding="utf-8", header=False, index=False)

def parse_args(args=None):
  parser = argparse.ArgumentParser()

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
      type=int,
      help="User provided timestamp. If not provided, get the timestamp in analyzer",
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
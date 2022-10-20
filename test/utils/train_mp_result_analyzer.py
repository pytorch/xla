import argparse
import numpy as np
import re
from tabulate import tabulate


def analyse_rate(logs):
  rates = np.array([float(log['Rate']) for log in logs])
  # Drop rates that belong to warm up rounds.
  threashold = np.percentile(rates, 7)  # 7% is a magic number, don't ask why.
  rates = rates[rates > threashold]
  mean = np.mean(rates)
  std = np.std(rates)
  median = np.median(rates)
  ninetieth = np.percentile(rates, 90)
  print(
      tabulate([['Rate', mean, std, median, ninetieth]],
               headers=['Type', 'Mean', 'Std Dev', 'Median', '90th %']))


def parse_logs(file: str):
  """
  We are trying to parse all training logs and extract each field. An example trainig log is like the following:
  "| Training Device=xla:0/1 Epoch=1 Step=1160 Loss=0.00136 Rate=430.41 GlobalRate=365.59 Time=21:54:34"
  The result will be a dictionary that has each field as the key and the value as the value in string, e.g:
  "{'Device': 'xla:0/5', 'Epoch': '1', 'Step': '0', 'Loss': '6.89059', 'Rate': '5.35', 'GlobalRate': '5.35', 'Time': '21:48:13'}"
  """
  logs = []
  with open(file) as fp:
    for line in fp:
      pattern = r'^\| Training (Device=.+) (Epoch=.+) (Step=.+) (Loss=.+) (Rate=.+) (GlobalRate=.+) (Time=.+)'
      match = re.match(pattern, line)
      if match != None:
        log = {}
        for group in match.groups():
          keyvalue = group.split('=')
          log[keyvalue[0]] = keyvalue[1]
        logs.append(log)
  return logs


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="""
    This script is to print some statistical metrics out from a test_train_mp_xxx.py test. It
    currently only prints 'Mean, Std Dev, Median, 90th%' data for 'Rate' with
    warm up rounds results trimmed.""")
  parser.add_argument(
      'file', type=str, help="The location of the result.txt file.")
  args = parser.parse_args()

  logs = parse_logs(args.file)
  analyse_rate(logs)

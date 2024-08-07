import argparse
import datetime
import logging
import json
import os
import re
import subprocess
import sys

from enum import Enum

logger = logging.getLogger(__name__)


def get_info_from_result_file(results_dir: str) -> tuple[str, str, float]:
  results_file = os.path.join(results_dir, 'results.jsonl')
  if not os.path.exists(results_file):
    sys.exit(f"Results file {results_file} not found. "
             "Please run experiment_runner.py first.")
  accelerator_model = None
  with open(results_file, 'r') as f:
    first_line = f.readline()
    acc_match = re.search(r'"accelerator_model": "([^"]+)"', first_line)
    time_match = re.search(r'"timestamp": ([0-9.]+)', first_line)
    if acc_match and time_match:
      accelerator_model = acc_match.group(1)
      timestamp = float(time_match.group(1))
    else:
      sys.exit(f"Cannot find a timestamp and a matching accelerator "
               "in {results_file}.")
  logger.debug(f"Found accelerator_model='{accelerator_model}' and "
               f"timestamp={timestamp} in {results_file}.")
  return accelerator_model, timestamp


def set_up_llama_repo(workspace_dir: str) -> str:
  llama_dir = os.path.join(workspace_dir, 'llama-inference')
  if os.path.exists(llama_dir):
    logger.debug(f'llama_dir={llama_dir} already exists; no setting up to do.')
    return llama_dir

  logger.debug(f'Setting up llama repo at {llama_dir}.')
  subprocess.check_call([
      'git', 'clone', 'https://github.com/pytorch-tpu/llama.git', '--branch',
      'llama2-google-next-inference', llama_dir
  ])
  subprocess.check_call(
      ['pip', 'install', '-r',
       os.path.join(llama_dir, 'requirements.txt')])
  subprocess.check_call(['pip', 'install', '-e', llama_dir])

  # Create model JSON files
  model_configs = {
      '7b.json': {
          "dim": 4096,
          "multiple_of": 256,
          "n_heads": 32,
          "n_layers": 32,
          "norm_eps": 1e-05,
          "vocab_size": -1
      },
      '13b.json': {
          "dim": 5120,
          "multiple_of": 256,
          "n_heads": 40,
          "n_layers": 40,
          "norm_eps": 1e-05,
          "vocab_size": -1
      },
      '70b.json': {
          "dim": 8192,
          "multiple_of": 4096,
          "ffn_dim_multiplier": 1.3,
          "n_heads": 64,
          "n_kv_heads": 8,
          "n_layers": 80,
          "norm_eps": 1e-05,
          "vocab_size": -1
      }
  }
  for filename, config in model_configs.items():
    filepath = os.path.join(llama_dir, filename)
    with open(filepath, 'w') as f:
      json.dump(config, f)
      f.write("\n")
  return llama_dir


def parse_log_file(log_file: str):
  latencies = []
  with open(log_file, 'r') as f:
    for line in f:
      if ('Totally decoded ' not in line or 'tokens in' not in line or
          ' seconds' not in line):
        continue
      parts = line.strip().split()
      tokens = float(parts[2])
      seconds = float(parts[5])
      latency_per_token = seconds / tokens
      latencies.append(latency_per_token)
  logger.debug(f'{log_file}: Found latencies={latencies}')
  return latencies


def benchmark_has_already_run(results_file: str, model_name: str, xla: str,
                              dynamo: str, batch_size: int):
  with open(results_file, 'r') as f:
    for line in f:
      # Grep for relevant lines to avoid parsing the entire JSONL file.
      if f'"model_name": "{model_name}"' not in line:
        continue
      r = json.loads(line.rstrip('\n|\r'))
      # yapf: disable
      if all(
          r.get(k1, {}).get(k2) == v
          for (k1, k2, v) in [
              ('experiment', 'accelerator', 'cuda'),
              ('experiment', 'batch_size', batch_size),
              ('experiment', 'dynamo', dynamo),
              ('experiment', 'test', 'eval'),
              ('experiment', 'xla', xla),
              ('experiment', 'xla_flags', None),
              ('model', 'model_name', model_name),
          ]):
        return True
      # yapf: enable
  return False


def run_benchmarks(args, llama_dir: str, results_dir: str,
                   accelerator_model: str, timestamp: float):
  os.chdir(llama_dir)
  for size in ['7b', '13b', '70b']:
    params_json = 'params.json'
    if os.path.exists(params_json):
      os.remove(params_json)
    os.symlink(f'{size}.json', params_json)
    model_name = f"llama2.{size}"
    for dynamo in [None, 'inductor', 'openxla']:
      backend = dynamo if dynamo else 'lazytensor'
      xla = None if dynamo == 'inductor' else 'PJRT'
      summary = f"{model_name} eval {backend} batch {args.batch_size}"

      results_file = os.path.join(results_dir, 'results.jsonl')
      if benchmark_has_already_run(results_file, model_name, xla, dynamo,
                                   args.batch_size):
        logger.info(f"SKIP already completed benchmark -- {summary}")
        continue

      logger.info(f"RUN {summary}")
      log_file = os.path.join(results_dir,
                              f'llama-inference.{backend}.{size}.log')

      cmd = [
          'python', 'example_text_completion.py', '1', '--ckpt_dir', '.',
          '--tokenizer_path',
          os.path.join(llama_dir, 't5_tokenizer/spiece.model'), '--max_seq_len',
          '2048', '--max_gen_len', '1000', f'--max_batch_size',
          f'{args.batch_size}', '--mp', 'True', f'--repeat', f'{args.repeat}',
          f'--dynamo', f'"{dynamo}"' if dynamo else "''"
      ]

      run_env = os.environ.copy()
      if dynamo == 'inductor':
        run_env['CUDA_VISIBLE_DEVICES'] = '0'
        run_env['USE_CUDA'] = '1'
      else:
        run_env['PJRT_DEVICE'] = 'CUDA'
        run_env['GPU_NUM_DEVICES'] = '1'

      run_ok = True
      with open(log_file, 'w') as f:
        try:
          subprocess.check_call(cmd, stdout=f, stderr=f, env=run_env)
        except subprocess.CalledProcessError:
          logger.warning(f"Run failed -- see {log_file}.")
          run_ok = False

      result = {
          'model': {
              'suite_name': 'llama2',
              'model_name': model_name,
          },
          'experiment': {
              'accelerator': 'cuda',
              'accelerator_model': accelerator_model,
              'xla': xla,
              'xla_flags': None,
              'dynamo': dynamo,
              'test': 'eval',
              'batch_size': args.batch_size,
          },
          'repeat': args.repeat,
          'iterations_per_run': 1,
          'metrics': {
              # Filled in below.
          },
          'timestamp': timestamp,
      }
      if run_ok:
        latencies = parse_log_file(log_file)
        result['metrics']['total_time'] = latencies
      else:
        result['metrics']['error'] = f"Run failed -- see {log_file}."

      with open(results_file, mode="a", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")


def parse_args():
  # Helper class for --log-level flag.
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

  parser = argparse.ArgumentParser(description='Run Llama inference benchmarks')
  parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
  parser.add_argument(
      '--log-level',
      default=LogLevel.info,
      choices=list(LogLevel),
      type=LogLevel.parse,
      help='Log level')
  parser.add_argument(
      '--repeat', type=int, default=8, help='Number of repetitions')
  parser.add_argument(
      '--workspace_dir', type=str, required=True, help='Workspace directory.')
  args = parser.parse_args()

  return args


def main():
  args = parse_args()
  logging.basicConfig(level=args.log_level.value, force=True)
  args.workspace_dir = os.path.expanduser(args.workspace_dir)
  if not os.path.exists(args.workspace_dir):
    sys.exit(f"Workspace directory {args.workspace_dir} not found.")

  # Sanity check: we should already be inside the appropriate venv.
  workspace_dir = os.path.realpath(args.workspace_dir)
  logger.debug(f'workspace_dir realpath: {workspace_dir}')
  if sys.prefix != os.path.join(workspace_dir, 'env'):
    sys.exit(
        "Error: must run under the Python venv from the given --workspace_dir.")

  results_dir = os.path.join(workspace_dir, 'experiment_results')
  accelerator_model, timestamp = get_info_from_result_file(results_dir)
  llama_dir = set_up_llama_repo(workspace_dir)

  run_benchmarks(args, llama_dir, results_dir, accelerator_model, timestamp)


if __name__ == "__main__":
  main()

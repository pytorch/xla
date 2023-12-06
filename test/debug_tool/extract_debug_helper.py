import os
from typing import NamedTuple


def check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def extract_execution_cause(lines):
  causes = []
  for i in range(len(lines)):
    if 'Execution Cause' in lines[i].decode():
      causes.append(lines[i + 1].decode())
  return causes


def extract_compilation_cause(lines):
  causes = []
  for i in range(len(lines)):
    if 'Compilation Cause' in lines[i].decode():
      causes.append(lines[i + 1].decode())
  return causes


class GraphInfo(NamedTuple):
  hash: str
  num_input: int
  num_output: int


def extract_graph_infos(lines):
  infos = []
  for i in range(len(lines)):
    if 'Graph Info' in lines[i].decode():
      hash = lines[i + 1].decode().split('Graph Hash: ')[1].strip()
      num_input = lines[i +
                        2].decode().split('Number of Graph Inputs:')[1].strip()
      num_output = lines[i + 3].decode().split(
          'Number of Graph Outputs:')[1].strip()
      infos.append(GraphInfo(hash, int(num_input), int(num_output)))

  return infos


def extract_python_frames(lines):
  frames = []
  current_frame = ''
  record_frame = False
  for i in range(len(lines)):
    if 'Python Frame Triggered Execution' in lines[i].decode():
      record_frame = True
    elif 'Analysis: ----------------' in lines[i].decode():
      record_frame = False
      frames.append(current_frame)
      current_frame = ''
    if record_frame:
      current_frame += lines[i].decode()
  return frames

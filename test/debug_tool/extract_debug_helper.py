import os
import re
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
  name: str
  hash: str
  num_input: int
  num_output: int


class PostCompilationInfo(NamedTuple):
  input_size: str
  output_size: str
  aliased_size: str
  intermediate_size: str
  program_size: str


def extract_graph_infos(lines):
  infos = []
  graph_name = ""
  for i in range(len(lines)):
    if 'Graph Info' in lines[i].decode():
      i += 1
      if 'Graph Name' in lines[i].decode():
        graph_name = re.search(r"\s* Analysis:\s*Graph Name:\s*(.+)",
                               lines[i].decode()).group(1)
        i += 1
      hash = re.search(r"\s* Analysis:\s*Graph Hash:\s*(.+)",
                       lines[i].decode()).group(1)
      i += 1
      num_input = re.search(r"\s* Analysis:\s*Number of Graph Inputs:\s*(.+)",
                            lines[i].decode()).group(1)
      i += 1
      num_output = re.search(r"\s* Analysis:\s*Number of Graph Outputs:\s*(.+)",
                             lines[i].decode()).group(1)
      i += 1
      infos.append(GraphInfo(graph_name, hash, int(num_input), int(num_output)))

  return infos


def extract_post_compilation_analysis(lines):
  infos = []
  i = 0
  while i < len(lines):
    if 'Post Compilation Analysis' in lines[i].decode():
      input_size = lines[i + 1].decode().split('Graph input size: ')[1].strip()
      output_size = lines[i +
                          2].decode().split('Graph output size: ')[1].strip()
      aliased_size = lines[i +
                           3].decode().split('Aliased Input size: ')[1].strip()
      intermediate_size = lines[i + 4].decode().split(
          'Intermediate tensor size: ')[1].strip()
      program_size = lines[i + 5].decode().split(
          'Compiled program size: ')[1].strip()
      infos.append(
          PostCompilationInfo(input_size, output_size, aliased_size,
                              intermediate_size, program_size))
      i += 7
    i += 1
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

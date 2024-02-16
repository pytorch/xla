import sys


def extract_all_graphs_buffers(filename):
  """Extracts lines between 'start buffers' and 'end_buffers' in a log file."""

  all_graphs_buffers = []
  current_buffer = []
  recording = False

  with open(filename, 'r') as logfile:
    for line in logfile:
      if "start buffers" in line:
        recording = True
      elif "end buffers" in line:
        recording = False
        all_graphs_buffers.append(current_buffer)
        current_buffer = []
      elif recording:
        # extract buffer
        current_buffer.append(line.strip().split("pjrt_buffer=")[1][:-1])
  return all_graphs_buffers


def find_new_buffers(l1, idx, all_l):
  new_buffer_cnt = 0
  for buffer in l1:
    is_existing_buffer = False
    for i, l in enumerate(all_l):
      if idx == i:
        continue
      if buffer in l:
        is_existing_buffer = True
    if not is_existing_buffer:
      new_buffer_cnt += 1
  return new_buffer_cnt


if __name__ == "__main__":
  if len(sys.argv) > 1:
    log_file = sys.argv[1]
    all_graphs_buffers = extract_all_graphs_buffers(log_file)

    if all_graphs_buffers:
      print("number of graphs: {}".format(len(all_graphs_buffers)))
      new_buffers_cnt = 0
      for idx, buffers in enumerate(all_graphs_buffers):
        print("number of buffers: {}".format(len(buffers)))
        new_buffers = find_new_buffers(buffers, idx, all_graphs_buffers)
        print("number of new buffers: {}".format(new_buffers))
        new_buffers_cnt += new_buffers
      print("total_new_buffers: {}".format(new_buffers_cnt))
    else:
      print("No 'start buffers' / 'end_buffers' block found in the log file.")
  else:
    print("Need file name")

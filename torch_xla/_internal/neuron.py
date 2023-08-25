import os


def num_local_processes() -> int:
  if 'MASTER_ADDR' not in os.environ:
    logging.warning("MASTER_ADDR not setting, defaulting to localhost")
  os.environ['NEURON_RT_ROOT_COMM_ID'] = '{}:{}'.format(
      os.environ.get('MASTER_ADDR', 'localhost'), '62182')
  if "NEURONCORE_NUM_DEVICES" not in os.environ:
    logging.warning("NEURONCORE_NUM_DEVICES not set, defaulting to 1")
  num_processes = int(os.environ.get("NEURONCORE_NUM_DEVICES", "1"))
  os.environ['NEURON_PJRT_PROCESSES_NUM_DEVICES'] = ','.join(
      ['1' for _ in range(num_processes)])

  return num_processes


def initialize_env(local_rank):
  os.environ["NEURON_PJRT_PROCESS_INDEX"] = str(local_rank)
  os.environ["NEURON_RT_VISIBLE_CORES"] = str(local_rank)

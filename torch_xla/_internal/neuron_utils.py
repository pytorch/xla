import os
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


def convert_range(range_spec):
  try:
    lowerupper = list(map(int, range_spec.split("-")))
  except Exception as e:
    print(f"ERROR: Malformed range specs in NEURON_RT_VISIBLE_CORES;" +
          f"expecting <int> or <lower int>-<upper int> (got {range_spec})")
    raise e
  if len(lowerupper) > 2:
    raise ValueError(
        f"ERROR: Range specs in NEURON_RT_VISIBLE_CORES should be of " +
        f"the form <int> or <lower int>-<upper int> (got {range_spec})")
  if len(lowerupper) == 2:
    if lowerupper[0] > lowerupper[1]:
      raise ValueError(
          f"ERROR: Range specs in NEURON_RT_VISIBLE_CORES should " +
          f"be of the form <int> or <lower int>-<upper int> (got {range_spec})")
    lowerupper = range(lowerupper[0], lowerupper[1] + 1)
  return lowerupper


def get_visible_cores_list():
  import os

  range_list = os.environ.get("NEURON_RT_VISIBLE_CORES")
  cores_list = None
  if range_list:
    range_list = range_list.split(",")
    cores_list = []
    for i in range_list:
      new = convert_range(i)
      if (set(cores_list) & set(new)) != set():
        raise ValueError(
            "ERROR: Please ensure the ranges in NEURON_RT_VISIBLE_CORES are mutually exclusive."
        )
      cores_list.extend(new)
  return cores_list


def remap_visible_cores(local_rank, local_world_size):
  cores_list = get_visible_cores_list()
  count = len(cores_list)
  assert (local_world_size > 0), "Local world size should be non-zero"
  if count <= 1 and local_world_size == 1:
    # Allow user to pass NEURON_RT_VISIBLE_CORES for sinlge-core workload
    pass
  elif local_world_size != count:
    raise ValueError(
        f"LOCAL_WORLD_SIZE (torchrun) or PJRT_LOCAL_PROCESS_COUNT (xmp.spawn) value of {local_world_size} "
        +
        f"is not equal to count {count} from NEURON_RT_VISIBLE_CORES {cores_list}"
    )
  elif local_rank >= count:
    raise ValueError(
        f"LOCAL_RANK (torchrun) or PJRT_LOCAL_PROCESS_RANK (xmp.spawn) value of {local_rank} is higher than "
        + f"count {count} from NEURON_RT_VISIBLE_CORES {cores_list}")
  else:
    remapped_core = cores_list[local_rank]
    logger.warning(f"Remapping NEURON_RT_VISIBLE_CORES {cores_list} to " +
                   f"NEURON_RT_VISIBLE_CORES[LOCAL_RANK]={remapped_core}")
    os.environ['NEURON_RT_VISIBLE_CORES'] = str(remapped_core)

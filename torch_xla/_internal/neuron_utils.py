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


def test_get_visible_cores_list():
  import os
  import pytest
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1"
  assert (get_visible_cores_list() == [1])
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1,2,3"
  assert (get_visible_cores_list() == [1, 2, 3])
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3"
  assert (get_visible_cores_list() == [1, 2, 3])
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3,5-8"
  assert (get_visible_cores_list() == [1, 2, 3, 5, 6, 7, 8])
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1,3,5-8"
  assert (get_visible_cores_list() == [1, 3, 5, 6, 7, 8])
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3,5-8,3-5"
  with pytest.raises(ValueError):
    get_visible_cores_list()
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3,5-8-5"
  with pytest.raises(ValueError):
    get_visible_cores_list()
  os.environ["NEURON_RT_VISIBLE_CORES"] = "a-b,5-8-5"
  with pytest.raises(Exception):
    get_visible_cores_list()
  os.environ["NEURON_RT_VISIBLE_CORES"] = "a"
  with pytest.raises(Exception):
    get_visible_cores_list()


def test_remap_visible_cores():
  import os
  import pytest
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1"
  remap_visible_cores(0, 1)
  assert (os.environ['NEURON_RT_VISIBLE_CORES'] == "1")
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1,2,3"
  remap_visible_cores(1, 3)
  assert (os.environ['NEURON_RT_VISIBLE_CORES'] == "2")
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3"
  remap_visible_cores(2, 3)
  assert (os.environ['NEURON_RT_VISIBLE_CORES'] == "3")
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3,5-8"
  remap_visible_cores(5, 7)
  assert (os.environ['NEURON_RT_VISIBLE_CORES'] == "7")
  os.environ["NEURON_RT_VISIBLE_CORES"] = "1,3,5-8"
  remap_visible_cores(5, 6)
  assert (os.environ['NEURON_RT_VISIBLE_CORES'] == "8")
  with pytest.raises(ValueError):
    remap_visible_cores(5, 9)
  with pytest.raises(ValueError):
    remap_visible_cores(6, 6)

import torch
import torch_xla

import inspect
from collections import defaultdict

from torch.utils._python_dispatch import TorchDispatchMode

class_count = defaultdict(int)
instance_count = dict()


def GetInstancePlaceHolder(class_type, obj):
  global class_count
  global instance_count

  if (class_type, id(obj)) not in instance_count:
    class_count[class_type] += 1
    instance_count[(class_type, id(obj))] = class_count[class_type]

  place_holder = instance_count[(class_type, id(obj))]

  return f".{place_holder}"


def CheckIgnored(key):
  ignored_list = ("self", "_bootstrap", "_fix_up_module",
                  "_get_supported_file_loaders", "_setup", "_buffers",
                  "_parameters", "_non_persistent_buffers_set")

  return (key.startswith("__") and key.endswith("__")) or key in ignored_list


def Prefix(prefix, val):
  if len(prefix) > 0:
    return f"{prefix}.{val}"
  else:
    return f"{val}"


def ReverseSearchBreadthFirst(container, obj, debug=False):
  if container is None:
    return False

  queue = []
  visited = set()
  nested_name = ""
  max_depth = 5
  queue.append((0, nested_name, container))

  while len(queue):
    depth, prefix, candidate = queue.pop(0)

    if depth > max_depth or id(candidate) in visited:
      continue

    visited.add(id(candidate))

    if isinstance(candidate, dict):
      for k, v in candidate.items():
        if not isinstance(k, str):
          if debug:
            print(f"Found non string key {k}")
          break
        if CheckIgnored(k):
          continue
        nested_name = Prefix(prefix, k)
        if v is obj:
          if debug:
            print(f"Found {nested_name}")
          return True, nested_name
        elif debug:
          print(f"Miss {nested_name}")
        if id(v) not in visited and depth < max_depth:
          queue.append((depth + 1, nested_name, v))
    elif isinstance(candidate, (list, tuple)):
      for i, v in enumerate(candidate):
        nested_name = Prefix(prefix, i)
        if v is obj:
          if debug:
            print(f"Found {nested_name}")
          return True, nested_name
        elif debug:
          print(f"Miss {nested_name}")
        if id(v) not in visited and depth < max_depth:
          queue.append((depth + 1, nested_name, v))
    elif hasattr(candidate, "__class__"):
      # Ignore class wich overrides __getattr__ and
      # generates error
      if type(candidate).__name__ == "_ClassNamespace":
        continue
      for att in ("_modules", "__dict__"):
        if hasattr(candidate, att):
          v = getattr(candidate, att)
          if id(v) not in visited and depth < max_depth:
            queue.append((depth + 1, nested_name, v))
    else:
      print("No action")

  return False, None


def FindMemberVariable(frame, obj):
  parent_frame = frame.f_back
  found = False
  variable_name = None

  for lframe in inspect.getouterframes(parent_frame):
    if lframe.frame.f_code.co_nlocals <= 0:
      continue
    self_name = lframe.frame.f_code.co_varnames[0]
    parent_obj = lframe.frame.f_locals[self_name]
    found, variable_name = ReverseSearchBreadthFirst(parent_obj, obj)
    if found:
      break

  return found, variable_name


def FindLocalVariable(frame, obj):
  found = False
  variable_name = None

  for lframe in inspect.getouterframes(frame.f_back):
    found, variable_name = ReverseSearchBreadthFirst(lframe.frame.f_locals, obj)
    if found:
      break

  return found, variable_name


def GetClassNameAndObjFromFrame(frame):
  class_obj_str = ""
  if frame.f_code.co_argcount == 0:
    return class_obj_str

  likely_obj_name = frame.f_code.co_varnames[0]

  obj = frame.f_locals[likely_obj_name]

  if not hasattr(obj, "__class__") or likely_obj_name != "self":
    return class_obj_str

  name = type(obj).__name__
  variable_name = None
  found = False

  found, variable_name = FindMemberVariable(frame, obj)

  if not found:
    found, variable_name = FindLocalVariable(frame, obj)

  if not found:
    variable_name = GetInstancePlaceHolder(name, obj)

  name = name + "[" + variable_name + "]"

  return name


def CleanNames(names):
  last_name = ""
  output = []
  for name in names:
    if name != last_name:
      output.append(name)
      last_name = name

  # Drop the last scope which is the scope name add op_name lowerings
  return output[:-1]


def GetAllObjectAndClassNames(frame):
  names = []
  while frame is not None:
    name = GetClassNameAndObjFromFrame(frame)
    if len(name) > 0:
      names.append(name)
    frame = frame.f_back

  names.reverse()

  names = CleanNames(names)

  output = "/".join(names)

  if len(output) > 0:
    output += "/"

  return output


class CustomOpNameLowering(TorchDispatchMode):

  def __init__(self):
    super().__init__()

  def __enter__(self):
    self._old_ir_debug = torch_xla._XLAC._get_ir_debug()
    torch_xla._XLAC._set_ir_debug(True)
    return super().__enter__()

  def __exit__(self, exc_type, exc_val, exc_tb):
    torch_xla._XLAC._set_ir_debug(self._old_ir_debug)
    super().__exit__(exc_type, exc_val, exc_tb)

  def __torch_dispatch__(self, func, types, args=(), kwargs={}):
    res = func(*args, **kwargs)
    if 'xla' in str(res.device):
      frame = inspect.currentframe()
      prefix = GetAllObjectAndClassNames(frame)
      torch_xla._XLAC._set_xla_custom_op_name(res, prefix)
    return res

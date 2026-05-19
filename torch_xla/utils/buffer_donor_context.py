from contextlib import contextmanager

import torch_xla


@contextmanager
def alias_with_buffer_donor_config(should_alias: bool = True):
  saved_config = torch_xla._XLAC._xla_get_enable_alias_with_buffer_donor_config(
  )
  torch_xla._XLAC._xla_set_enable_alias_with_buffer_donor_config(should_alias)
  try:
    yield saved_config
  finally:
    torch_xla._XLAC._xla_set_enable_alias_with_buffer_donor_config(saved_config)

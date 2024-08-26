import torch_xla

# Whether to skip checking input is a device data or not in the optim_mod.
# Enabling it will reduce the overhead a bit but will throw a runtime error
# if input is a pending IR.
skip_input_data_check = False
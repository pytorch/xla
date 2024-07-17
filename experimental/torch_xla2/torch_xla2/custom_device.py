import os
import torch
import torch.utils.cpp_extension


import torch_xla2.ops.jaten


# TODO: Do I even need a C++ module at all?
# Load the C++ extension containing your custom kernels.
# foo_module = torch.utils.cpp_extension.load(
#     name="custom_device_extension",
#     sources=[
#         os.path.dirname(__file__) + "/cpp/registration.cpp",
#     ],
#     #extra_include_paths=["cpp_extensions"],
#     extra_cflags=["-g"],
#     verbose=True,
# )

# torch.register_privateuse1_backend('foo')
torch.utils.rename_privateuse1_backend('jax')


# print(foo_module.Tensor)
print('Create a tensor with `jax` device:', torch.tensor([0], device='jax:0'))

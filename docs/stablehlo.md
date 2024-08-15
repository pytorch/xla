Torch Export to StableHLO
--------------------------

This document describes how to use torch export + torch xla to export to 
[StableHLO](https://github.com/openxla/stablehlo) format.

## How to use:
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo
import torch_xla.core.xla_model as xm
import torchvision
import torch

# Initialize XLA device
xla_device = xm.xla_device()

# Load a model (example with ResNet18)
resnet18 = torchvision.models.resnet18()

# Sample input (tuple of tensors)
sample_input = (torch.randn(4, 3, 224, 224), )

# Forward pass to ensure model initialization
output = resnet18(*sample_input)

# Export model to StableHLO format
exported = export(resnet18, sample_input)
stablehlo_program = exported_program_to_stablehlo(exported)

# Display StableHLO text representation
print(stablehlo_program.get_stablehlo_text('forward'))

# Display StableHLO bytecode
print(stablehlo_program.get_stablehlo_bytecode('forward'))

# Run the module on XLA device
sample_input_xla = tuple(s.to(xla_device) for s in sample_input)
output2 = stablehlo_program(*sample_input_xla)

# Compare outputs between original and StableHLO-backed models
print(torch.allclose(output, output2.cpu(), atol=1e-5))

# Saving StableHLO bytecodes to disk
One can save StableHLO to disk with:

stablehlo_program.save('/tmp/stablehlo_dir')


# The path should be a path to an empty directory. If it doesn't exist, it will be created. This directory can be loaded again as another stablehlo_program:

from torch_xla.stablehlo import StableHLOGraphModule
stablehlo_program2 = StableHLOGraphModule.load('/tmp/stablehlo_dir')
output3 = stablehlo_program2(*sample_input_xla)

# Convert saved StableHLO for serving
StableHLO is an open format supported for serving in TensorFlow Serving. However, before using it with TensorFlow Serving, you need to wrap the generated StableHLO bytecode into a tf.saved_model format.

## First, ensure that you have the latest TensorFlow installed in your current Python environment. If not, install it with:

pip install tf-nightly

## Now, you can run a converter (provided in the torch/xla installation):

stablehlo-to-saved-model /tmp/stablehlo_dir /tmp/resnet_tf/1

## After that, you can run your model server on the newly generated tf.saved_model with the TensorFlow Serving binary.

docker pull tensorflow/serving
docker run -p 8500:8500 \
--mount type=bind,source=/tmp/resnet_tf,target=/models/resnet_tf \
-e MODEL_NAME=resnet_tf -t tensorflow/serving &


# Common wrappers
I want to save directly to tf.saved_model format without needing to run a separate command
You can accomplish this by using this helper function:

from torch_xla.tf_saved_model_integration import save_torch_module_as_tf_saved_model

save_torch_module_as_tf_saved_model(
    resnet18,  # Original PyTorch `torch.nn.Module`
    sample_input,  # Sample inputs used for tracing
    '/tmp/resnet_tf'  # Directory for `tf.saved_model`
)

## Other common wrappers
def save_as_stablehlo(exported_model: 'ExportedProgram',
                      stablehlo_dir: os.PathLike,
                      options: Optional[StableHLOExportOptions] = None):
    pass

def save_torch_model_as_stablehlo(
    torchmodel: torch.nn.Module,
    args: Tuple[Any],
    path: os.PathLike,
    options: Optional[StableHLOExportOptions] = None) -> None:
    pass


# Files produced by save_as_stablehlo
Inside /tmp/stablehlo_dir in the example above, you will find three directories: data, constants, and functions. Both data and constants consist of tensors used by the program saved as numpy.ndarray using numpy.save.

The functions directory will contain StableHLO bytecode, here named forward.bytecode, human-readable StableHLO code (MLIR form) forward.mlir, and a JSON file specifying which weights and original user's input become which positional arguments of this StableHLO function, as well as the data types and shapes of every argument.

## Example:
$ find /tmp/stablehlo_dir
./functions
./functions/forward.mlir
./functions/forward.bytecode
./functions/forward.meta
./constants
./constants/3
./constants/1
./constants/0
./constants/2
./data
./data/L__fn___layers_15_feed_forward_w2.weight
./data/L__fn___layers_13_feed_forward_w1.weight
./data/L__fn___layers_3_attention_wo.weight
./data/L__fn___layers_12_ffn_norm_weight
./data/L__fn___layers_25_attention_wo.weight


The JSON file is a serialized form of the torch_xla.stablehlo.StableHLOFunc class. This format is currently also in the prototype stage, and there are no backward compatibility guarantees. The future plan is to standardize a format that the major frameworks (PyTorch, JAX, TensorFlow) can agree on.


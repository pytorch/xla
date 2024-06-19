Torch Export to StableHLO (Prototype feature)
This document describes how to use Torch export with Torch XLA to export models to StableHLO format.

NOTE: This feature is currently in prototype and may undergo changes in future releases.

How to Use:
python
Copy code
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo
import torch_xla.core.xla_model as xm
import torchvision
import torch

Initialize XLA device
xla_device = xm.xla_device()

Load a model (example with ResNet18)
resnet18 = torchvision.models.resnet18()

Sample input (tuple of tensors)
sample_input = (torch.randn(4, 3, 224, 224), )

Forward pass to ensure model initialization
output = resnet18(*sample_input)

Export model to StableHLO format
exported = export(resnet18, sample_input)
stablehlo_program = exported_program_to_stablehlo(exported)

Display StableHLO text representation
print(stablehlo_program.get_stablehlo_text('forward'))

Display StableHLO bytecode
print(stablehlo_program.get_stablehlo_bytecode('forward'))

Run the module on XLA device
sample_input_xla = tuple(s.to(xla_device) for s in sample_input)
output2 = stablehlo_program(*sample_input_xla)

Compare outputs between original and StableHLO-backed models
print(torch.allclose(output, output2.cpu(), atol=1e-5))
Saving StableHLO Bytecodes to Disk
Save StableHLO to disk using:

python
Copy code
stablehlo_program.save('/tmp/stablehlo_dir')
Load the saved StableHLO program from disk:

python
Copy code
from torch_xla.stablehlo import StableHLOGraphModule
stablehlo_program2 = StableHLOGraphModule.load('/tmp/stablehlo_dir')
output3 = stablehlo_program2(*sample_input_xla)
Converting for Serving with TensorFlow Serving
To use StableHLO with TensorFlow Serving, convert it to tf.saved_model format:

bash
Copy code
stablehlo-to-saved-model /tmp/stablehlo_dir /tmp/resnet_tf/1
Run TensorFlow Serving with Docker:

bash
Copy code
docker pull tensorflow/serving
docker run -p 8500:8500
--mount type=bind,source=/tmp/resnet_tf,target=/models/resnet_tf
-e MODEL_NAME=resnet_tf -t tensorflow/serving &
Alternatively, use TensorFlow Serving without Docker:

bash
Copy code
tensorflow_model_server --rest_api_port=8501 --model_name=resnet_tf --model_base_path=/tmp/resnet_tf
For more details, refer to the TensorFlow Serving guide.

Common Wrappers
For saving directly to tf.saved_model format without an additional command:

python
Copy code
from torch_xla.tf_saved_model_integration import save_torch_module_as_tf_saved_model

save_torch_module_as_tf_saved_model(
resnet18, # Original PyTorch torch.nn.Module
sample_input, # Sample inputs used for tracing
'/tmp/resnet_tf' # Directory for tf.saved_model
)
Other common wrappers include functions like save_as_stablehlo for saving StableHLO directly from an ExportedProgram.

Files Produced by save_as_stablehlo
In the saved directory (/tmp/stablehlo_dir), you'll find directories such as data, constants, and functions, containing necessary artifacts for the StableHLO program.
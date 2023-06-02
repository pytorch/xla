import torch_xla
import torch_xla.core.xla_model as xm
import torch
'''
The following MLIR module should be dump in this test
module @IrToHlo.9 attributes {mhlo.cross_program_prefetches = [], mhlo.dynamic_parameter_bindings = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1xi64>, %arg1: tensor<1xi64>) -> tuple<tensor<1xi64>, tensor<1xi64>, tensor<1xi64>> {
    %0 = stablehlo.constant dense<1> : tensor<i64>
    %1 = stablehlo.constant dense<1> : tensor<1xi64>
    %2 = stablehlo.multiply %arg1, %1 : tensor<1xi64>
    %3 = stablehlo.add %arg0, %2 : tensor<1xi64>
    %4 = stablehlo.tuple %arg0, %arg1, %3 {xla_shape = "(s64[1]{0}, s64[1]{0}, s64[1]{0})"} : tuple<tensor<1xi64>, tensor<1xi64>, tensor<1xi64>>
    return %4 : tuple<tensor<1xi64>, tensor<1xi64>, tensor<1xi64>>
  }
}
'''

x = torch.tensor([3], device=xm.xla_device())
y = torch.tensor([3], device=xm.xla_device())
z = x + y

# Example usage of dumping StableHLO given output tensors
stablehlo = xm.xla_get_stablehlo([z])
# print(stablehlo)
# Example usage of dump StableHLO of the entire graph
stablehlo = xm.xla_get_stablehlo()
# print(stablehlo)

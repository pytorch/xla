from dataclasses import dataclass
from typing import List, Union
from torchgen.api.lazy import LazyIrSchema
from torchgen.dest.lazy_ir import aten_symbol, GenLazyIR
from torchgen.gen_lazy_tensor import run_gen_lazy_tensor
from torchgen.model import NativeFunction, NativeFunctionsGroup
from torchgen.api.types import (
    BaseCType,
    OptionalCType,
    VectorCType,
    kernel_signature,
)
import pathlib

# assuming this file is under pytorch/xla
torch_xla_root = pathlib.Path(__file__).parent.absolute().parent
torch_root = torch_xla_root.parent
aten_path = str(torch_root / "aten" / "src" / "ATen")
shape_inference_hdr = str(torch_root / "torch" / "csrc" / "lazy" / "core" /
                          "shape_inference.h")
impl_path = str(torch_xla_root / "torch_xla" / "csrc" / "aten_xla_type.cpp")
source_yaml = str(torch_xla_root / "xla_native_functions.yaml")


@dataclass(frozen=True)
class GenXlaLazyIR(GenLazyIR):

  def lowering_function(self, f: Union[NativeFunctionsGroup,
                                       NativeFunction]) -> str:
    return f"""torch_xla::XlaOpVector Lower(LoweringContext* loctx) const override;"""

  def node_ctor_args_call(self, all_args):
    args_with_type = []
    for i in all_args:
      cpp_type = i.lazy_type.cpp_type()
      if cpp_type == 'torch::lazy::Value':
        cpp_type = 'torch_xla::XlaValue'
      args_with_type.append(f"const {cpp_type}& {i.name}")
    return ", ".join(args_with_type)

  def node_base_ctor_call(self, schema: LazyIrSchema) -> str:
    # backends can customize the way the node base class constructor is called,
    # as long as all of its arguments can be generated from information available from the schema
    base_ctor_value_args_list = []
    for arg in schema.filtered_args(values=True, scalars=False):
      if isinstance(arg.lazy_type, BaseCType) or isinstance(
          arg.lazy_type, VectorCType):
        base_ctor_value_args_list.append(f"{arg.name}")
      elif isinstance(arg.lazy_type, OptionalCType):
        base_ctor_value_args_list.append(f"{arg.name}.value_or(kNullValue)")
      else:
        raise AssertionError(
            f"Unsupported type ({arg.lazy_type}) - add support if necessary")
    base_ctor_value_args = ", ".join(base_ctor_value_args_list)

    scalar_args = schema.filtered_args(values=False, scalars=True)
    scalar_hashes = ", ".join([f"{a.name}" for a in scalar_args])

    return f"""{self.node_base}(torch::lazy::OpKind({aten_symbol(schema)}),
              {{{base_ctor_value_args}}}, std::move(shapes),
              [&]() {{ return {schema.node_name}OutputShape({base_ctor_value_args}); }},
              /* num_outputs */ {len(schema.returns)},
              torch::lazy::MHash({scalar_hashes}))"""


if __name__ == '__main__':
  run_gen_lazy_tensor(
      aten_path=aten_path,
      source_yaml=source_yaml,
      output_dir="torch_xla/csrc/generated",
      dry_run=False,
      impl_path=impl_path,
      node_base="XlaNode",
      node_base_hdr="torch_xla/csrc/generated_file_include.h",
      tensor_class="torch_xla::XLATensor",
      tensor_class_hdr="torch_xla/csrc/tensor.h",
      shape_inference_hdr=shape_inference_hdr,
      lazy_ir_generator=GenXlaLazyIR,
      build_in_tree=False,
      per_operator_headers=True,
      backend_name="XLA",
      gen_forced_fallback_code=False,
      backend_namespace="torch_xla",
      get_tensorlist="GetTensorList",
      get_tensor_or_wrap_number="bridge::GetXlaTensorOrCreateForWrappedNumber",
      try_get_tensor="TryGetXlaTensor",
      metrics_counter='XLA_FN_COUNTER("xla::")',
      create_tensor="XLATensor::Create",
      #create_tensor = "CreateFrom",
      #create_from_first_tensor = True,
      create_aten_from_ltc_tensor="torch_xla::bridge::AtenFromXlaTensor",
      tuple_aten_from_ltc_tensors="torch::lazy::TupleAtenFromLtcTensors",
      lazy_value_class="torch_xla::XlaValue",
      lazy_tensor_ptr="torch_xla::XLATensorPtr",
      get_device_fn="torch_xla::bridge::GetXlaDevice")

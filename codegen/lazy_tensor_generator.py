import sys
import os

from dataclasses import dataclass
from typing import List, Union
from torchgen.api.lazy import LazyIrSchema
from torchgen.dest.lazy_ir import aten_symbol, node_ctor_inputs, GenLazyIR, GenLazyNativeFuncDefinition
from torchgen.gen_lazy_tensor import run_gen_lazy_tensor
from torchgen.model import NativeFunction, NativeFunctionsGroup
from torchgen.api.types import (
    BaseCType,
    OptionalCType,
    VectorCType,
    boolT,
    kernel_signature,
)

xla_root = sys.argv[1]
torch_root = os.path.join(xla_root, "torch")
aten_path = os.path.join(torch_root, "aten", "src", "ATen")
shape_inference_hdr = os.path.join(torch_root, "torch", "csrc", "lazy", "core",
                                   "shape_inference.h")
impl_path = os.path.join(xla_root, "__main__",
                         "torch_xla/csrc/aten_xla_type.cpp")
source_yaml = sys.argv[2]
output_dir = sys.argv[3]


def is_boolean_dtype(lazy_type):
  return lazy_type == BaseCType(boolT)


@dataclass(frozen=True)
class GenXlaLazyIR(GenLazyIR):

  def lowering_function(self, f: Union[NativeFunctionsGroup,
                                       NativeFunction]) -> str:
    return f"""torch_xla::XlaOpVector Lower(LoweringContext* loctx) const override;"""

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

    shape_fn_inputs_list = [
        f"{a.name}" for a in (schema.positional_args + schema.keyword_args)
        if (a.is_lazy_value or isinstance(a.lazy_type, VectorCType) or
            is_boolean_dtype(a.lazy_type) or a.name == 'reduction' or
            a.name == 'dim')
    ]
    shape_fn_inputs = ", ".join(shape_fn_inputs_list)

    scalar_args = schema.filtered_args(values=False, scalars=True)
    scalar_hashes = ", ".join([f"{a.name}" for a in scalar_args])

    return f"""{self.node_base}(torch::lazy::OpKind({aten_symbol(schema)}),
              {{{base_ctor_value_args}}},
              [&]() {{ return {schema.node_name}OutputShape({shape_fn_inputs}); }},
              /* num_outputs */ {len(schema.returns)},
              torch::lazy::MHash({scalar_hashes}))"""


# Upstream class lives at torchgen/dest/lazy_ir.py.
# We override this class to remove torch::lazy::Shape related logic.
# Resulting NativeFuncDefinition is generated at xla/torch_xla/csrc/XlaNativeFunctions.cpp.
@dataclass(frozen=True)
class GenXlaLazyNativeFuncDefinition(GenLazyNativeFuncDefinition):

  # This function is responsible for shape inference for `torch::lazy::Shape`.
  # We don't need `torch::lazy::Shape` for our codegen, so returning an empty string.
  def shape_inference(self, func: NativeFunction, schema: LazyIrSchema) -> str:
    return ""

  def build_ir_node(self, func: NativeFunction, schema: LazyIrSchema) -> str:
    node_ctor_input_str = node_ctor_inputs(schema)
    return f"""torch::lazy::NodePtr node = torch::lazy::ReuseNode<{schema.node_name}>({node_ctor_input_str});
      if (!node) {{
          {self.shape_inference(func, schema)}
          node = torch::lazy::MakeNode<{schema.node_name}>({node_ctor_input_str});
          CacheNode(node);
      }}
      """


if __name__ == '__main__':
  run_gen_lazy_tensor(
      aten_path=aten_path,
      source_yaml=source_yaml,
      output_dir=output_dir,
      dry_run=False,
      impl_path=impl_path,
      node_base="XlaNode",
      node_base_hdr="torch_xla/csrc/generated_file_include.h",
      tensor_class="torch_xla::XLATensor",
      tensor_class_hdr="torch_xla/csrc/tensor.h",
      shape_inference_hdr=shape_inference_hdr,
      lazy_ir_generator=GenXlaLazyIR,
      native_func_definition_generator=GenXlaLazyNativeFuncDefinition,
      build_in_tree=False,
      per_operator_headers=True,
      backend_name="XLA",
      gen_forced_fallback_code=False,
      use_lazy_shape=False,
      backend_namespace="torch_xla",
      get_tensorlist="GetTensorList",
      get_tensor_or_wrap_number="bridge::GetXlaTensorOrCreateForWrappedNumber",
      try_get_tensor="bridge::TryGetXlaTensor",
      metrics_counter='TORCH_LAZY_FN_COUNTER("xla::")',
      create_tensor="XLATensor::Create",
      create_aten_from_ltc_tensor="torch_xla::bridge::AtenFromXlaTensor",
      tuple_aten_from_ltc_tensors="torch_xla::bridge::TupleAtenFromXlaTensors",
      lazy_tensor_ptr="torch_xla::XLATensorPtr",
      get_device_fn="torch_xla::bridge::GetXlaDevice")

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
import pathlib

# assuming this file is under pytorch/xla
torch_xla_root = pathlib.Path(__file__).parent.absolute().parent
torch_root = torch_xla_root.parent
aten_path = str(torch_root / "aten" / "src" / "ATen")
shape_inference_hdr = str(torch_root / "torch" / "csrc" / "lazy" / "core" /
                          "shape_inference.h")
impl_path = str(torch_xla_root / "torch_xla" / "csrc" / "aten_xla_type.cpp")
source_yaml = str(torch_xla_root / "xla_native_functions.yaml")


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

  def gen(self, schema: LazyIrSchema) -> List[str]:
    opkind = schema.opkind or aten_symbol(schema)

    # for now, we just want one IR class decl and soon after also the method defs
    # and we use the functional version not out/inplace.
    all_args = schema.filtered_args()
    value_args = schema.filtered_args(values=True, scalars=False)
    scalar_args = schema.filtered_args(values=False, scalars=True)

    ctor_args = [f"const {i.lazy_type.cpp_type()}& {i.name}" for i in all_args]
    reuse_ctor_args = ", ".join(ctor_args)
    # if schema.properties.ShapePrecompute:
    #     ctor_args.append("std::vector<wonjoo>&& shapes")
    node_ctor_args = ", ".join(ctor_args)

    scalar_initializers = ",\n        ".join([
        # This code is just special casing the mapping from string_view -> strings
        f"{a.name}({a.name}.has_value() ? c10::make_optional(std::string(*{a.name})) : c10::nullopt)"
        if a.lazy_type.cpp_type() == "c10::optional<c10::string_view>" else
        f"{a.name}({a.name})" for a in scalar_args
    ])
    if len(scalar_initializers):
      scalar_initializers = f",\n        {scalar_initializers}"
    scalar_decls = "\n  ".join([
        f"std::string {a.name};" if a.lazy_type.cpp_type() == "c10::string_view"
        else f"c10::optional<std::string> {a.name};"
        if a.lazy_type.cpp_type() == "c10::optional<c10::string_view>" else
        f"{a.lazy_type.cpp_type()} {a.name};" for a in scalar_args
    ])
    optional_values = [
        arg.name
        for arg in schema.filtered_args(values=True, scalars=False)
        if isinstance(arg.lazy_type, OptionalCType)
    ]
    has_optional_decls = "\n  ".join(
        [f"bool has_{value}: 1;" for value in optional_values])
    has_optional_defs = "\n    ".join(
        [f"has_{value} = !!{value};" for value in optional_values])
    members_to_string = []
    for arg in scalar_args:
      if isinstance(arg.lazy_type, OptionalCType):
        members_to_string.append(f"""if ({arg.name}.has_value()) {{
    ss << ", {arg.name}=" << {arg.name}.value();
  }} else {{
    ss << ", {arg.name}=null";
  }}""")
      else:
        members_to_string.append(f'ss << ", {arg.name}=" << {arg.name};')
    members_to_string_str = "\n    ".join(members_to_string)

    return [
        f"""\
class {schema.node_name} : public {self.node_base} {{
public:
static torch::lazy::OpKind ClassOpKind() {{
  return torch::lazy::OpKind({opkind});
}}

{schema.node_name}({node_ctor_args})
    : {self.node_base_ctor_call(schema)}{scalar_initializers}
{{
  {has_optional_defs}
}}

std::string ToString() const override {{
  std::stringstream ss;
  ss << {self.node_base}::ToString();
  {members_to_string_str}
  return ss.str();
}}

{self.create_function(schema, reuse_ctor_args)}

{self.can_be_reused_function(schema, reuse_ctor_args)}

{self.lowering_function(schema)}

{scalar_decls}
{has_optional_decls}

}};

""",
    ]


@dataclass(frozen=True)
class GenXlaLazyNativeFuncDefinition(GenLazyNativeFuncDefinition):

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
      output_dir="torch_xla/csrc/generated",
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
      backend_namespace="torch_xla",
      get_tensorlist="GetTensorList",
      get_tensor_or_wrap_number="bridge::GetXlaTensorOrCreateForWrappedNumber",
      try_get_tensor="bridge::TryGetXlaTensor",
      metrics_counter='XLA_FN_COUNTER("xla::")',
      create_tensor="XLATensor::Create",
      create_aten_from_ltc_tensor="torch_xla::bridge::AtenFromXlaTensor",
      tuple_aten_from_ltc_tensors="torch_xla::bridge::TupleAtenFromXlaTensors",
      lazy_tensor_ptr="torch_xla::XLATensorPtr",
      get_device_fn="torch_xla::bridge::GetXlaDevice")

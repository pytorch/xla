#include "torch_xla/csrc/ops/custom_call.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/shape_helper.h"

namespace torch_xla {

CustomCall::CustomCall(
    torch::lazy::OpList inputs, const std::string& call_target,
    xla::Shape output_shape, bool has_side_effect,
    const std::string& backend_config, const int api_version,
    const std::unordered_map<std::string, std::string>& frontend_attributes)
    : XlaNode(xla_custom_call, inputs, std::move(output_shape),
              /*num_outputs=*/output_shape.tuple_shapes_size(),
              torch::lazy::MHash(call_target)),
      call_target_(call_target),
      has_side_effect_(has_side_effect),
      backend_config_(backend_config),
      api_version_(api_version),
      frontend_attributes_(frontend_attributes) {}

CustomCall::CustomCall(torch::lazy::OpList inputs,
                       const std::string& call_target, xla::Shape output_shape,
                       bool has_side_effect, const std::string& backend_config,
                       const int api_version)
    : CustomCall(inputs, call_target, output_shape, has_side_effect,
                 backend_config, api_version,
                 std::unordered_map<std::string, std::string>()) {}

torch::lazy::NodePtr CustomCall::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<CustomCall>(operands, call_target_,
                                         this->xla_shape(), has_side_effect_,
                                         backend_config_, api_version_);
}

XlaOpVector CustomCall::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  inputs.reserve(this->operands().size());
  for (auto& operand : operands()) {
    inputs.push_back(loctx->GetOutputOp(operand));
  }
  xla::Shape output_shape = this->xla_shape();
  const int n_outputs = output_shape.tuple_shapes_size();
  if (n_outputs == 1) {
    output_shape = output_shape.tuple_shapes(0);
  }
  XLA_CHECK(api_version_ >= 0 && api_version_ < 5);

  xla::FrontendAttributes feattr;
  feattr.mutable_map()->insert(frontend_attributes_.begin(),
                               frontend_attributes_.end());
  xla::XlaScopedFrontendAttributesAssignment feattr_assign(inputs[0].builder(),
                                                           feattr);
  xla::XlaOp output = xla::CustomCall(
      inputs[0].builder(), call_target_, inputs, output_shape,
      /*opaque=*/backend_config_,
      /*has_side_effect=*/has_side_effect_,
      /*output_operand_aliasing=*/{},
      /*literal=*/nullptr,
      /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/static_cast<xla::CustomCallApiVersion>(api_version_));
  std::vector<xla::XlaOp> result;
  if (n_outputs == 1) {
    result = {output};
  } else {
    result.reserve(n_outputs);
    for (int i = 0; i < n_outputs; ++i) {
      result.push_back(xla::GetTupleElement(output, i));
    }
  }
  return ReturnOps(result, loctx);
}

std::string CustomCall::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", call_target=" << call_target_
     << ", has_side_effect=" << has_side_effect_
     << ", backend_config=" << backend_config_
     << ", api_version=" << api_version_;
  return ss.str();
}

}  // namespace torch_xla

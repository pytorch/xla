#include "torch_xla/csrc/ops/triton_op.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(std::vector<const torch::lazy::Value> inputs, 
                           std::vector<float> fparams, std::vector<int> iparams) {
  std::cout << "TritonOp::NodeOutputShape" << std::endl;
  return xla::ShapeUtil::MakeShape(xla::F32, {1});
  // auto lower_for_shape_fn =
  //     [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
  //   return BuildTritonOp(operands[0], dim, keepdim);
  // };
  // return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

// TODO? What is the op kind?
// TODO? We don't know the number of inputs and parameters....
// or even the type of parameters
// how to pass the input shape...
TritonOp::TritonOp(std::vector<const torch::lazy::Value> inputs, 
                   std::vector<float> fparams, std::vector<int> iparams);
    : XlaNode(xla_custom_op, {input},
              [&]() { return NodeOutputShape(inputs, fparams, iparams); },
              /*num_outputs=*/1, torch::lazy::MHash(name)),
      fparams_(fparams),
      iparams_(iparams),
      num_inputs_(inputs.size()) {
        // TODO: compile the triton kernel - how to ensure this happens only once?
        // TODO: how do i get the name of the function call?
        name_ = ptx_kernel_name;
        XLA_REGISTER_CUSTOM_CALL_TARGET(ptx_kernel_name, "CUDA");
    }

torch::lazy::NodePtr TritonOp::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<TritonOp>(operands.at(0), dim_, keepdim_);
}

XlaOpVector TritonOp::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> operands[num_inputs_];
  for (int i = 0; i < num_inputs_; i++) {
    operands[i] = loctx->GetOutputOp(operand(i));
  }
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {1});
  xla::XlaOp out = xla::CustomCall(input[0].builder(), name_, operands, shape);
  return ReturnOp(out, loctx);
  // return ReturnOp(BuildTritonOp(input, dim_, keepdim_), loctx);
}

std::string TritonOp::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", name=" << name_;
  return ss.str();
}

}  // namespace torch_xla

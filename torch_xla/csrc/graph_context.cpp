#include "graph_context.h"
#include "absl/strings/str_cat.h"

namespace torch_xla {

XlaGraphContext::XlaGraphContext(bool collate_parameters)
    : collate_parameters_(collate_parameters), builder_("XlaGraphContext") {}

xla::XlaOp XlaGraphContext::GetParameter(
    const std::shared_ptr<xla::ComputationClient::Data>& data) {
  if (collate_parameters_) {
    auto it = parameters_map_.find(data.get());
    if (it == parameters_map_.end()) {
      xla::XlaOp param =
          xla::Parameter(builder(), parameters_.size(), data->shape(),
                         absl::StrCat("param_", parameters_.size()));
      parameters_.push_back(data);
      it = parameters_map_.emplace(data.get(), param).first;
    }
    return it->second;
  }
  xla::XlaOp param =
      xla::Parameter(builder(), parameters_.size(), data->shape(),
                     absl::StrCat("param_", parameters_.size()));
  parameters_.push_back(data);
  return param;
}

std::vector<xla::ComputationClient::Data*> XlaGraphContext::GetParametersData()
    const {
  std::vector<xla::ComputationClient::Data*> parameters;
  for (auto& param : parameters_) {
    parameters.push_back(param.get());
  }
  return parameters;
}

xla::int64 XlaGraphContext::AddResult(xla::XlaOp op) {
  root_tuple_.push_back(std::move(op));
  return root_tuple_.size() - 1;
}

xla::StatusOr<xla::XlaComputation> XlaGraphContext::Build() {
  if (!root_tuple_.empty()) {
    auto root = xla::Tuple(builder(), root_tuple_);
    return builder()->Build(root);
  }
  return builder()->Build();
}

xla::StatusOr<xla::XlaComputation> XlaGraphContext::Build(
    const xla::XlaOp& root) {
  CHECK(root_tuple_.empty());
  return builder()->Build(root);
}

XlaGraphNode::XlaGraphNode(
    Generator generator, xla::Shape shape,
    tensorflow::gtl::ArraySlice<const std::shared_ptr<XlaGraphNode>> inputs)
    : generator_(std::move(generator)),
      shape_(std::move(shape)),
      inputs_(inputs.begin(), inputs.end()) {
  for (auto& input : inputs_) {
    graph_size_ += input->graph_size();
  }
}

}  // namespace torch_xla

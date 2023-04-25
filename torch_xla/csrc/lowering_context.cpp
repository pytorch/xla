#include "torch_xla/csrc/lowering_context.h"

#include <torch/csrc/lazy/core/ir_metadata.h>

#include <sstream>
#include <stdexcept>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/sys_util.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

class HloMetadataSetter {
 public:
  HloMetadataSetter(LoweringContext* loctx, const torch::lazy::Node* node) {
    if (ShouldPopulateXlaOpMetadata()) {
      PopulateXlaOpMetadata(loctx, node);
      loctx_ = loctx;
    }
  }

  ~HloMetadataSetter() {
    if (loctx_ != nullptr) {
      loctx_->builder()->ClearOpMetadata();
    }
  }

 private:
  static bool ShouldPopulateXlaOpMetadata() {
    static bool op_metadata = xla::sys_util::GetEnvBool("XLA_HLO_DEBUG", false);
    return op_metadata;
  }

  static void PopulateXlaOpMetadata(LoweringContext* loctx,
                                    const torch::lazy::Node* node) {
    xla::OpMetadata metadata;
    // NOTE: we apply some string manipulation as xprof backend utility
    // for nesting/grouping traces depends on certain op name/type
    // patterns for classification.
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/utils/tf_op_utils.cc#L55
    std::string op_type =
        absl::StrReplaceAll(node->op().ToString(), {{":", "_"}});
    metadata.set_op_type(op_type);
    const torch::lazy::MetaData& nmeta = node->metadata();
    std::string op_name_prefix;
    if (!nmeta.scope.empty()) {
      op_name_prefix =
          absl::StrCat(absl::StrReplaceAll(nmeta.scope, {{":", "_"}}), "/");
    }
    metadata.set_op_name(absl::StrCat(op_name_prefix, op_type));

    if (!nmeta.frame_info.empty()) {
      const torch::lazy::SourceLocation& frame = nmeta.frame_info.front();
      std::string::size_type pos = frame.file.find_last_of('/');
      if (pos == std::string::npos) {
        pos = 0;
      } else {
        ++pos;
      }
      metadata.set_source_file(frame.function + "@" + frame.file.substr(pos));
      metadata.set_source_line(frame.line);
    }
    loctx->builder()->SetOpMetadata(std::move(metadata));
  }

  LoweringContext* loctx_ = nullptr;
};

}  // namespace

LoweringContext::LoweringContext(const std::string& name,
                                 torch::lazy::BackendDevice device)
    : torch::lazy::LoweringContext(name, device), builder_(name) {}

LoweringContext::LoweringContext(
    const std::string& name, torch::lazy::BackendDevice device,
    c10::ArrayRef<const torch::lazy::Node*> post_order,
    torch::lazy::Util::EmissionMap emit_status)
    : torch::lazy::LoweringContext(name, device, {}, emit_status),
      builder_(name) {
  for (auto node : post_order) {
    LowerNode(node);
  }
}

xla::XlaOp LoweringContext::GetParameter(
    const std::shared_ptr<torch::lazy::BackendData>& data) {
  torch::lazy::BackendData::Handle handle = data->GetHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    xla::XlaOp param = xla::Parameter(builder(), parameters_.size(),
                                      UnwrapXlaData(data)->shape(),
                                      absl::StrCat("p", parameters_.size()));
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(data);
  }
  parameter_sequence_.push_back(it->second.index);
  return it->second.param;
}

const std::vector<torch::lazy::BackendDataPtr>&
LoweringContext::GetParametersData() const {
  return parameters_;
}

const std::vector<size_t>& LoweringContext::GetParameterSequence() const {
  return parameter_sequence_;
}

xla::XlaOp LoweringContext::GetResult(size_t index) const {
  return root_tuple_.at(index);
}

void LoweringContext::SetResult(size_t index, xla::XlaOp op) {
  root_tuple_.at(index) = std::move(op);
}

xla::StatusOr<xla::XlaComputation> LoweringContext::BuildXla() {
  if (!root_tuple_.empty()) {
    xla::XlaOp root = xla::Tuple(builder(), root_tuple_);
    return builder()->Build(root);
  }
  return builder()->Build();
}

xla::StatusOr<xla::XlaComputation> LoweringContext::BuildXla(xla::XlaOp root) {
  XLA_CHECK(root_tuple_.empty());
  return builder()->Build(root);
}

void LoweringContext::AssignOutputOp(const torch::lazy::Output& output,
                                     xla::XlaOp op) {
  emitted_outputs_[output] = std::move(op);
}

xla::XlaOp LoweringContext::GetOutputOp(const torch::lazy::Output& output) {
  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    auto post_order =
        torch::lazy::Util::ComputePostOrder(output.node, &emit_status_);
    for (auto node : post_order) {
      LowerNode(node);
    }
    // At this point the outpout better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    XLA_CHECK(it != emitted_outputs_.end())
        << "No XLA operation emitted for output: " << output;
  }
  return it->second;
}

XlaOpVector LoweringContext::LowerNode(const torch::lazy::Node* node) {
  XlaOpVector result_ops;
  try {
    HloMetadataSetter meta_setter(this, node);

    const XlaNode* casted = dynamic_cast<const XlaNode*>(node);
    result_ops = casted->Lower(this);
  } catch (const std::exception& ex) {
    ReportBuilderError(node, ex.what());
  }
  if (!builder()->first_error().ok()) {
    ReportBuilderError(node, /*error_msg=*/nullptr);
  }
  return result_ops;
}

void LoweringContext::ReportBuilderError(const torch::lazy::Node* node,
                                         const char* error_msg) {
  std::stringstream ss;
  ss << "Error while lowering: " << node->ToString() << "\n";
  if (!builder()->first_error().ok()) {
    ss << "XLA builder error: " << builder()->GetCurrentStatus() << "\n";
  }
  if (error_msg != nullptr) {
    ss << "Error: " << error_msg << "\n";
  }
  const torch::lazy::MetaData& nmeta = node->metadata();
  if (!nmeta.scope.empty()) {
    ss << "Scope: " << nmeta.scope << "\n";
  }
  ss << nmeta.frame_info;
  throw std::runtime_error(ss.str());
}

void LoweringContext::SetUpAlias(const std::vector<int64_t>& output_index,
                                 int64_t param_number,
                                 const std::vector<int64_t>& param_index,
                                 bool must_alias) {
  XLA_CHECK_EQ(output_index.size(), 1);
  XLA_CHECK_EQ(param_index.size(), 1);
  builder_.SetUpAlias({output_index[0]}, param_number, {param_index[0]});
}

bool LoweringContext::CheckResultShape(
    const torch::lazy::BackendDataPtr& parameter_data, size_t result_idx) {
  xla::XlaOp root = GetResult(result_idx);
  const xla::Shape& root_shape = XlaHelpers::ShapeOfXlaOp(root);
  return UnwrapXlaData(parameter_data)->shape() == root_shape;
}

size_t LoweringContext::AddResult(const torch::lazy::Output& output) {
  root_tuple_.push_back(GetOutputOp(output));
  return root_tuple_.size() - 1;
}

size_t LoweringContext::AddResult(xla::XlaOp op) {
  root_tuple_.push_back(op);
  return root_tuple_.size() - 1;
}

void LoweringContext::AddParameter(const torch::lazy::Output& output,
                                   size_t index,
                                   const torch::lazy::Shape& shape,
                                   const std::string& name) {
  XLA_ERROR() << "not implemented";
  return;
}

torch::lazy::ComputationPtr LoweringContext::Build() {
  xla::XlaComputation xla_computation = ConsumeValue(BuildXla());
  return std::make_shared<torch_xla::Computation>(
      builder_.name(), std::move(xla_computation), device_);
}

}  // namespace torch_xla

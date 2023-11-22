#include "torch_xla/csrc/lowering_context.h"

#include <torch/csrc/lazy/core/ir_metadata.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/unwrap_data.h"

namespace torch_xla {

// Invalid stack frame id
static int kInvalidIndex = 0;

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
    static bool op_metadata =
        runtime::sys_util::GetEnvBool("XLA_HLO_DEBUG", false);
    return FLAGS_torch_lazy_ir_debug || op_metadata;
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

    const CustomOpNameMetaData* custom_opname_meta =
        dynamic_cast<const CustomOpNameMetaData*>(node->user_metadata());

    // const XlaNode* xla_node_cast = dynamic_cast<const XlaNode*>(node);
    std::string op_name_prefix;
    size_t max_stack_depth = nmeta.frame_info.size();

    if (custom_opname_meta != nullptr) {
      op_name_prefix = custom_opname_meta->op_name_prefix;
      max_stack_depth = custom_opname_meta->max_stack_depth;
    } else {
      TF_LOG(WARNING) << "*** No metadata! op_type=" << op_type;
    }

    /*
    if (xla_node_cast != nullptr && !xla_node_cast->custom_op_name().empty()) {
      op_name_prefix = xla_node_cast->custom_op_name();

      if (xla_node_cast->max_call_stack_depth() != 0) {
        max_stack_depth = xla_node_cast->max_call_stack_depth();
      }
    }
    */

    if (!nmeta.scope.empty()) {
      op_name_prefix =
          absl::StrCat(absl::StrReplaceAll(nmeta.scope, {{":", "_"}}), "/");
    }
    metadata.set_op_name(absl::StrCat(op_name_prefix, op_type));

    if (!nmeta.frame_info.empty()) {
      auto frame_it = nmeta.frame_info.rbegin();
      int parent_frame_id = kInvalidIndex;
      int depth = 0;
      for (; frame_it != nmeta.frame_info.rend() && depth < max_stack_depth;
           ++frame_it) {
        // Where we aren't working with an XLANode there is no way to pass down
        // stack depth -
        if (custom_opname_meta == 0) {
          std::string_view func_search(frame_it->function);
          if (func_search.find("__torch_dispatch__") != func_search.npos) {
            break;
          }
        }

        parent_frame_id =
            loctx->AddStackFrameLocation(*frame_it, parent_frame_id);
        ++depth;
      }

      // Point to first entry / deepest call / top frame in call stack
      --frame_it;

      metadata.set_source_file(frame_it->file);
      metadata.set_source_line(frame_it->line);
      metadata.set_stack_frame_id(parent_frame_id);
    }

    loctx->builder()->SetOpMetadata(std::move(metadata));
  }

  LoweringContext* loctx_ = nullptr;
};

}  // namespace

int FindId(std::string_view key, std::map<std::string_view, int>& index) {
  auto entry_iterator = index.find(key);
  if (entry_iterator == index.end()) {
    return 0;
  } else {
    return entry_iterator->second;
  }
}

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

// TODO(lsy323): Get reserved number for unbounded dim after it's added in XLA.
static constexpr int64_t kUnboundedSize = std::numeric_limits<int64_t>::min();

xla::XlaOp LoweringContext::GetParameter(
    const std::shared_ptr<torch::lazy::BackendData>& data,
    const std::unordered_set<uint32_t>& unbounded_dynamic_dims) {
  torch::lazy::BackendData::Handle handle = data->GetHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    xla::Shape shape =
        std::dynamic_pointer_cast<runtime::ComputationClient::Data>(data)
            ->shape();
    for (const int dim : unbounded_dynamic_dims) {
      shape.set_dynamic_dimension(dim, true);
      shape.set_dimensions(dim, kUnboundedSize);
    }
    xla::XlaOp param = xla::Parameter(builder(), parameters_.size(), shape,
                                      absl::StrCat("p", parameters_.size()));
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(data);
  } else {
    XLA_CHECK(unbounded_dynamic_dims.empty())
        << "The unbounded dynamic dims can only be set when Parameter is "
           "created.";
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
  xla::StatusOr<xla::XlaComputation> xla;
  if (!root_tuple_.empty()) {
    xla::XlaOp root = xla::Tuple(builder(), root_tuple_);
    xla = builder()->Build(root);
  } else {
    xla = builder()->Build();
  }

  if (xla.ok()) {
    (*xla->mutable_proto()->mutable_stack_frame_index()) = indexes_;
  }

  return xla;
}

xla::StatusOr<xla::XlaComputation> LoweringContext::BuildXla(xla::XlaOp root) {
  XLA_CHECK(root_tuple_.empty());
  auto xla = builder()->Build(root);

  if (xla.ok()) {
    (*xla->mutable_proto()->mutable_stack_frame_index()) = indexes_;
  }

  return xla;
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
    if (!casted->dynamic_dims().empty()) {
      xla::internal::XlaBuilderFriend builder_friend;
      auto* inst = builder_friend.GetInstruction(result_ops[0]);
      auto* mutable_dynamic =
          inst->mutable_shape()->mutable_is_dynamic_dimension();
      if (mutable_dynamic->empty()) {
        for (int i = 0; i < inst->dimensions_size(); i++) {
          mutable_dynamic->Add(false);
        }
      }
      auto* mutable_dims = inst->mutable_shape()->mutable_dimensions();
      for (const auto dim : casted->dynamic_dims()) {
        mutable_dynamic->Set(dim, true);
        mutable_dims->Set(dim, kUnboundedSize);
      }
    }
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
  const xla::Shape& root_shape = ShapeHelper::ShapeOfXlaOp(root);
  return std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
             parameter_data)
             ->shape() == root_shape;
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

int64_t LoweringContext::AddStackFrameLocation(
    const torch::lazy::SourceLocation& frame, int64_t parent_frame_id) {
  int line = frame.line;
  int column = 0;  // Not provided in torch stack
  std::string filename = frame.file;
  std::string function_name = frame.function;

  int filename_id = FindId(filename, file_name_to_id_);
  if (filename_id == 0) {
    indexes_.add_file_names(std::move(filename));
    filename_id = indexes_.file_names_size();
    file_name_to_id_[indexes_.file_names(filename_id - 1)] = filename_id;
  }

  int function_name_id = FindId(function_name, function_name_to_id_);
  if (function_name_id == 0) {
    indexes_.add_function_names(std::move(function_name));
    function_name_id = indexes_.function_names_size();
    function_name_to_id_[indexes_.function_names(function_name_id - 1)] =
        function_name_id;
  }

  auto location_tuple =
      std::make_tuple(filename_id, function_name_id, line, column);
  auto file_location_iterator = file_location_to_id_.find(location_tuple);
  int file_location_id = 0;
  if (file_location_iterator == file_location_to_id_.end()) {
    auto file_location = indexes_.add_file_locations();
    file_location->set_file_name_id(filename_id);
    file_location->set_function_name_id(function_name_id);
    file_location->set_line(line);
    file_location->set_column(column);

    file_location_id = indexes_.file_locations_size();
    file_location_to_id_[location_tuple] = file_location_id;
  } else {
    file_location_id = file_location_iterator->second;
  }

  auto frame_tuple = std::make_tuple(file_location_id, parent_frame_id);
  auto stack_frame_iterator = frame_to_id_.find(frame_tuple);
  int stack_frame_id = 0;
  if (stack_frame_iterator == frame_to_id_.end()) {
    auto frame = indexes_.add_stack_frames();
    frame->set_file_location_id(file_location_id);
    frame->set_parent_frame_id(parent_frame_id);

    stack_frame_id = indexes_.stack_frames_size();
    frame_to_id_[frame_tuple] = stack_frame_id;
  } else {
    stack_frame_id = stack_frame_iterator->second;
  }

  return stack_frame_id;
}

torch::lazy::ComputationPtr LoweringContext::Build() {
  xla::XlaComputation xla_computation = ConsumeValue(BuildXla());

  return std::make_shared<runtime::ComputationClient::Computation>(
      builder_.name(), std::move(xla_computation), device_);
}

}  // namespace torch_xla

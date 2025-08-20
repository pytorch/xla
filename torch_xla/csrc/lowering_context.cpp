#include "torch_xla/csrc/lowering_context.h"

#include <torch/csrc/lazy/core/ir_metadata.h>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/stack_frame_index_builder.h"
#include "torch_xla/csrc/status.h"
#include "torch_xla/csrc/torch_xla_op_sharding.h"

namespace torch_xla {

namespace {

class HloMetadataSetter {
 public:
  HloMetadataSetter(LoweringContext& lowering_context,
                    const torch::lazy::Node& node)
      : lowering_context_(lowering_context) {
    if (ShouldPopulateXlaOpMetadata()) {
      PopulateXlaOpMetadata(lowering_context, node);
    }
  }

  // This class is neither copyable nor movable.
  HloMetadataSetter(const HloMetadataSetter&) = delete;
  HloMetadataSetter& operator=(const HloMetadataSetter&) = delete;
  HloMetadataSetter(HloMetadataSetter&&) = delete;
  HloMetadataSetter& operator=(HloMetadataSetter&&) = delete;

  ~HloMetadataSetter() {
    if (ShouldPopulateXlaOpMetadata()) {
      lowering_context_.builder()->ClearOpMetadata();
    }
  }

 private:
  // Returns true iff this class should populate XLA op metadata in its
  // constructor.
  static bool ShouldPopulateXlaOpMetadata() {
    static const bool op_metadata =
        runtime::sys_util::GetEnvBool("XLA_HLO_DEBUG", false);
    return FLAGS_torch_lazy_ir_debug || op_metadata;
  }

  static void PopulateXlaOpMetadata(LoweringContext& lowering_context,
                                    const torch::lazy::Node& node) {
    xla::OpMetadata metadata;
    // NOTE: we apply some string manipulation as xprof backend utility
    // for nesting/grouping traces depends on certain op name/type
    // patterns for classification.
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/utils/tf_op_utils.cc#L55
    const std::string op_type =
        absl::StrReplaceAll(node.op().ToString(), {{":", "_"}});
    metadata.set_op_type(op_type);

    const torch::lazy::MetaData& nmeta = node.metadata();
    auto* const custom_opname_meta =
        dynamic_cast<const CustomOpNameMetaData*>(node.user_metadata());

    std::string op_name_prefix;
    size_t max_stack_depth = nmeta.frame_info.size();

    if (custom_opname_meta != nullptr) {
      op_name_prefix = custom_opname_meta->op_name_prefix;
      max_stack_depth = custom_opname_meta->max_stack_depth;
    }

    if (!nmeta.scope.empty()) {
      op_name_prefix =
          absl::StrCat(absl::StrReplaceAll(nmeta.scope, {{":", "_"}}), "/");
    }
    metadata.set_op_name(absl::StrCat(op_name_prefix, op_type));

    // Sets file, line and stack_frame_id in metadata
    lowering_context.stack_frame_index_builder()->AddStackFrameLocations(
        nmeta.frame_info, static_cast<int>(max_stack_depth), metadata);

    lowering_context.builder()->SetOpMetadata(std::move(metadata));
  }

  LoweringContext& lowering_context_;
};

}  // namespace

LoweringContext::LoweringContext(const std::string& name,
                                 torch::lazy::BackendDevice device)
    : torch::lazy::LoweringContext(name, std::move(device)),
      builder_(name),
      stack_frame_index_builder_(std::make_shared<StackFrameIndexBuilder>()) {}

LoweringContext::LoweringContext(
    const std::string& name, torch::lazy::BackendDevice device,
    const c10::ArrayRef<const torch::lazy::Node*> post_order,
    torch::lazy::Util::EmissionMap emit_status)
    : torch::lazy::LoweringContext(name, std::move(device), {},
                                   std::move(emit_status)),
      builder_(name),
      stack_frame_index_builder_(std::make_shared<StackFrameIndexBuilder>()) {
  for (const auto* node : post_order) {
    LowerNode(*node);
  }
}

xla::XlaOp LoweringContext::GetParameter(
    const std::shared_ptr<torch::lazy::BackendData>& backend_data,
    const std::unordered_set<uint32_t>& unbounded_dynamic_dims) {
  const torch::lazy::BackendData::Handle handle = backend_data->GetHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    auto* const data =
        dynamic_cast<runtime::ComputationClient::Data*>(backend_data.get());
    ABSL_CHECK(data != nullptr);
    xla::Shape shape = data->shape();
    for (const int dim : unbounded_dynamic_dims) {
      shape.set_dynamic_dimension(dim, true);
      shape.set_dimensions(dim, xla::Shape::kUnboundedSize);
    }
    const size_t param_index = parameters_.size();
    const std::string param_name = absl::StrCat("p", param_index);
    xla::XlaOp param;
    if (data->HasSharding()) {
      const torch_xla::OpSharding sharding = data->GetSharding();
      const xla::XlaScopedShardingAssignment scoped_sharding(
          builder(), sharding.GetXlaOpSharding());
      param = xla::Parameter(builder(), param_index, shape, param_name);
    } else {
      param = xla::Parameter(builder(), param_index, shape, param_name);
    }
    it = parameters_map_.emplace(handle, Parameter{param, param_index}).first;
    parameters_.push_back(backend_data);
  } else {
    ABSL_CHECK(unbounded_dynamic_dims.empty())
        << "The unbounded dynamic dims can only be set when Parameter is "
           "created.";
  }
  parameter_sequence_.push_back(it->second.index);
  return it->second.param;
}

std::optional<size_t> LoweringContext::GetParameterId(
    const std::shared_ptr<torch::lazy::BackendData>& backend_data) const {
  const torch::lazy::BackendData::Handle handle = backend_data->GetHandle();
  const auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    return std::nullopt;
  }
  return it->second.index;
}

const std::vector<torch::lazy::BackendDataPtr>&
LoweringContext::GetParametersData() const {
  return parameters_;
}

const std::vector<size_t>& LoweringContext::GetParameterSequence() const {
  return parameter_sequence_;
}

xla::XlaOp LoweringContext::GetResult(const size_t index) const {
  return root_tuple_.at(index);
}

void LoweringContext::SetResult(const size_t index, const xla::XlaOp op) {
  root_tuple_.at(index) = op;
}

absl::StatusOr<xla::XlaComputation> LoweringContext::BuildXla() {
  absl::StatusOr<xla::XlaComputation> xla;

  // check whether build for cond/body computation or not, and skip Tuple step
  // if yes
  if (!root_tuple_.empty() & (root_tuple_.size() == 1) &
      ((get_name_string() == "condctx") or (get_name_string() == "bodyctx"))) {
    xla = builder()->Build(root_tuple_.at(0));
  } else if (!root_tuple_.empty()) {
    const xla::XlaOp root = xla::Tuple(builder(), root_tuple_);
    xla = builder()->Build(root);
  } else {
    xla = builder()->Build();
  }

  if (xla.ok()) {
    (*xla->mutable_proto()->mutable_stack_frame_index()) =
        stack_frame_index_builder()->stack_frame_index();
  }

  return xla;
}

absl::StatusOr<xla::XlaComputation> LoweringContext::BuildXla(
    const xla::XlaOp root) {
  ABSL_CHECK(root_tuple_.empty());
  auto xla = builder()->Build(root);

  if (xla.ok()) {
    (*xla->mutable_proto()->mutable_stack_frame_index()) =
        stack_frame_index_builder()->stack_frame_index();
  }

  return xla;
}

void LoweringContext::AssignOutputOp(const torch::lazy::Output& output,
                                     const xla::XlaOp op) {
  emitted_outputs_[output] = op;
}

xla::XlaOp LoweringContext::GetOutputOp(const torch::lazy::Output& output) {
  auto it = emitted_outputs_.find(output);

  if (it == emitted_outputs_.end()) {
    const auto post_order =
        torch::lazy::Util::ComputePostOrder(output.node, &emit_status_);
    for (const auto* const node : post_order) {
      LowerNode(*node);
    }
    // At this point the output better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    ABSL_CHECK(it != emitted_outputs_.end())
        << "No XLA operation emitted for output: " << output;
  }
  return it->second;
}

void LoweringContext::ExtractShardingAndSetDenormalizedTileAssignments(
    std::vector<std::shared_ptr<torch_xla::OpSharding>> shardings) {
  for (auto sharding : shardings) {
    std::vector<int64_t> denormalized_tile_assignment =
        sharding->GetDenormalizedTileAssignment();
    if (!denormalized_tile_assignment.empty()) {
      denormalized_tile_assignments_.push_back(denormalized_tile_assignment);
    }
  }
}

XlaOpVector LoweringContext::LowerNode(const torch::lazy::Node& node) {
  XlaOpVector result_ops;
  try {
    const HloMetadataSetter meta_setter(*this, node);
    const XlaNode* const casted = dynamic_cast<const XlaNode*>(&node);

    result_ops = casted->Lower(this);
    // save the denormalized_tile_assignment from all nodes and then use it
    // during Compile
    auto shardings = casted->GetShardings();
    if (!shardings.empty()) {
      ExtractShardingAndSetDenormalizedTileAssignments(shardings);
    }
    if (!casted->dynamic_dims().empty()) {
      const xla::internal::XlaBuilderFriend builder_friend;
      auto* const inst = builder_friend.GetInstruction(result_ops[0]);
      auto* const mutable_dynamic =
          inst->mutable_shape()->mutable_is_dynamic_dimension();
      if (mutable_dynamic->empty()) {
        for (int i = 0; i < inst->dimensions_size(); i++) {
          mutable_dynamic->Add(false);
        }
      }
      auto* const mutable_dims = inst->mutable_shape()->mutable_dimensions();
      for (const auto dim : casted->dynamic_dims()) {
        mutable_dynamic->Set(dim, true);
        mutable_dims->Set(dim, xla::Shape::kUnboundedSize);
      }
    }
  } catch (const std::exception& ex) {
    ReportBuilderError(node, ex.what());
  }
  if (!builder()->first_error().ok()) {
    ReportBuilderError(node, /*error_msg=*/"");
  }
  return result_ops;
}

void LoweringContext::ReportBuilderError(const torch::lazy::Node& node,
                                         const absl::string_view error_msg) {
  std::stringstream ss;
  ss << "Error while lowering: " << node.ToString() << "\n";
  if (!builder()->first_error().ok()) {
    ss << "XLA builder error: " << builder()->GetCurrentStatus() << "\n";
  }
  if (!error_msg.empty()) {
    ss << "Error: " << error_msg << "\n";
  }
  const torch::lazy::MetaData& nmeta = node.metadata();
  if (!nmeta.scope.empty()) {
    ss << "Scope: " << nmeta.scope << "\n";
  }
  ss << nmeta.frame_info;
  throw std::runtime_error(ss.str());
}

void LoweringContext::SetUpAlias(const std::vector<int64_t>& output_index,
                                 const int64_t param_number,
                                 const std::vector<int64_t>& param_index,
                                 const bool must_alias) {
  ABSL_CHECK_EQ(output_index.size(), 1);
  ABSL_CHECK_EQ(param_index.size(), 1);
  builder_.SetUpAlias({output_index[0]}, param_number, {param_index[0]});
}

bool LoweringContext::CheckResultShape(
    const torch::lazy::BackendDataPtr& parameter_data,
    const size_t result_idx) {
  const xla::XlaOp root = GetResult(result_idx);
  const xla::Shape& root_shape = ShapeHelper::ShapeOfXlaOp(root);
  return std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
             parameter_data)
             ->shape() == root_shape;
}

size_t LoweringContext::AddResult(const torch::lazy::Output& output) {
  root_tuple_.push_back(GetOutputOp(output));
  return root_tuple_.size() - 1;
}

size_t LoweringContext::AddResult(const xla::XlaOp op) {
  root_tuple_.push_back(op);
  return root_tuple_.size() - 1;
}

void LoweringContext::AddParameter(const torch::lazy::Output& output,
                                   const size_t index,
                                   const torch::lazy::Shape& shape,
                                   const std::string& name) {
  ABSL_LOG(FATAL) << "not implemented";
  return;
}

torch::lazy::ComputationPtr LoweringContext::Build() {
  xla::XlaComputation xla_computation = GetValueOrThrow(BuildXla());
  return std::make_shared<runtime::ComputationClient::Computation>(
      builder_.name(), std::move(xla_computation), device_);
}

}  // namespace torch_xla

#include "lazy_xla/csrc/compiler/xla_lowering_context.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "lazy_xla/csrc/compiler/nnc_computation_client.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace xla_backend {

lazy_tensors::StatusOr<lazy_tensors::ProgramShape>
GenericComputationXla::GetProgramShape() const {
  xla::ProgramShape program_shape =
      ConsumeValue(computation_.GetProgramShape());
  std::vector<lazy_tensors::Shape> parameter_shapes;
  parameter_shapes.reserve(program_shape.parameters_size());
  for (const xla::Shape& xla_parameter_shape : program_shape.parameters()) {
    parameter_shapes.push_back(
        compiler::XlaHelpers::LazyTensorsShape(xla_parameter_shape));
  }
  lazy_tensors::Shape result_shape =
      compiler::XlaHelpers::LazyTensorsShape(program_shape.result());
  return lazy_tensors::ProgramShape(
      parameter_shapes, program_shape.parameter_names(), result_shape);
}

namespace {

class HloMetadataSetter {
 public:
  HloMetadataSetter(XlaLoweringContext* loctx, const ir::Node* node) {
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
        lazy_tensors::sys_util::GetEnvBool("XLA_HLO_DEBUG", false);
    return op_metadata;
  }

  static void PopulateXlaOpMetadata(XlaLoweringContext* loctx,
                                    const ir::Node* node) {
    xla::OpMetadata metadata;
    // NOTE: we apply some string manipulation as xprof backend utility
    // for nesting/grouping traces depends on certain op name/type
    // patterns for classification.
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/utils/tf_op_utils.cc#L55
    std::string op_type =
        absl::StrReplaceAll(node->op().ToString(), {{":", "_"}});
    metadata.set_op_type(op_type);
    const ir::MetaData& nmeta = node->metadata();
    std::string op_name_prefix;
    if (!nmeta.scope.empty()) {
      op_name_prefix =
          absl::StrCat(absl::StrReplaceAll(nmeta.scope, {{":", "_"}}), "/");
    }
    metadata.set_op_name(absl::StrCat(op_name_prefix, op_type));

    if (!nmeta.frame_info.empty()) {
      const SourceLocation& frame = nmeta.frame_info.front();
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

  XlaLoweringContext* loctx_ = nullptr;
};

xla::ShapeIndex XlaShapeIndex(const lazy_tensors::ShapeIndex& shape_index) {
  LTC_CHECK_EQ(shape_index.size(), 1);
  return {shape_index[0]};
}

}  // namespace

lazy_tensors::Shape XlaLoweringContext::GetResultShape(size_t index) const {
  xla::XlaOp root = GetResult(index);
  return compiler::XlaHelpers::LazyTensorsShape(
      compiler::XlaHelpers::ShapeOfXlaOp(root));
}

size_t XlaLoweringContext::AddResult(const ir::Output& output) {
  return AddResult(GetOutputOp(output));
}

void XlaLoweringContext::LowerNodeToResult(const ir::Node* node) {
  for (auto& xla_op : LowerNode(node)) {
    AddResult(xla_op);
  }
}

void XlaLoweringContext::AddParameter(const ir::Output& output, size_t index,
                                      const lazy_tensors::Shape& shape,
                                      const std::string& name) {
  AssignOutputOp(output,
                 xla::Parameter(builder(), index,
                                compiler::XlaHelpers::XlaShape(shape), name));
}

lazy_tensors::StatusOr<std::shared_ptr<lazy_tensors::GenericComputation>>
XlaLoweringContext::Build() {
  if (!root_tuple_.empty()) {
    xla::XlaOp root = xla::Tuple(builder(), root_tuple_);
    return std::shared_ptr<lazy_tensors::GenericComputation>(
        new GenericComputationXla(ConsumeValue(builder()->Build(root))));
  }
  return std::shared_ptr<lazy_tensors::GenericComputation>(
      new GenericComputationXla(ConsumeValue(builder()->Build())));
}

void XlaLoweringContext::SetUpAlias(
    const lazy_tensors::ShapeIndex& output_index,
    lazy_tensors::int64 param_number,
    const lazy_tensors::ShapeIndex& param_index) {
  builder()->SetUpAlias(XlaShapeIndex(output_index), param_number,
                        XlaShapeIndex(param_index));
}

xla::XlaOp XlaLoweringContext::GetParameter(
    const std::shared_ptr<lazy_tensors::client::Data>& data) {
  lazy_tensors::client::Data::OpaqueHandle handle = data->GetOpaqueHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    xla::XlaOp param =
        xla::Parameter(builder(), parameters_.size(),
                       compiler::XlaHelpers::XlaShape(data->shape()),
                       absl::StrCat("p", parameters_.size()));
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(data);
  }
  parameter_sequence_.push_back(it->second.index);
  return it->second.param;
}

size_t XlaLoweringContext::AddResult(xla::XlaOp op) {
  root_tuple_.push_back(std::move(op));
  return root_tuple_.size() - 1;
}

xla::XlaOp XlaLoweringContext::GetResult(size_t index) const {
  return root_tuple_.at(index);
}

void XlaLoweringContext::AssignOutputOp(const ir::Output& output,
                                        xla::XlaOp op) {
  emitted_outputs_[output] = std::move(op);
}

xla::XlaOp XlaLoweringContext::GetOutputOp(const ir::Output& output) {
  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    auto post_order = ir::Util::ComputePostOrder(output.node, &emit_status_);
    for (auto node : post_order) {
      LowerNode(node);
    }
    // At this point the outpout better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    LTC_CHECK(it != emitted_outputs_.end())
        << "No XLA operation emitted for output: " << output;
  }
  return it->second;
}

XlaOpVector XlaLoweringContext::LowerNode(const ir::Node* node) {
  XlaOpVector result_ops;
  try {
    HloMetadataSetter meta_setter(this, node);

    result_ops = LowerNodeToXla(node, this);
  } catch (const std::exception& ex) {
    ReportBuilderError(node, ex.what());
  }
  if (!builder()->first_error().ok()) {
    ReportBuilderError(node, /*error_msg=*/nullptr);
  }
  return result_ops;
}

void XlaLoweringContext::ReportBuilderError(const ir::Node* node,
                                            const char* error_msg) {
  std::stringstream ss;
  ss << "Error while lowering: " << node->ToString() << "\n";
  if (!builder()->first_error().ok()) {
    ss << "Builder error: " << builder()->GetCurrentStatus() << "\n";
  }
  if (error_msg != nullptr) {
    ss << "Error: " << error_msg << "\n";
  }
  const ir::MetaData& nmeta = node->metadata();
  if (!nmeta.scope.empty()) {
    ss << "Scope: " << nmeta.scope << "\n";
  }
  ss << nmeta.frame_info;
  throw std::runtime_error(ss.str());
}

}  // namespace xla_backend
}  // namespace compiler

namespace ir {

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name, Device device,
    lazy_tensors::Span<const Node* const> post_order,
    Util::EmissionMap emit_status) {
  return std::make_unique<compiler::xla_backend::XlaLoweringContext>(
      name, device, post_order, emit_status);
}

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name, Device device) {
  return std::make_unique<compiler::xla_backend::XlaLoweringContext>(name,
                                                                     device);
}

}  // namespace ir
}  // namespace torch_lazy_tensors

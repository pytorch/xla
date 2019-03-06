#include "torch_xla/csrc/view.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/ops/generic_slice.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/update_slice.h"
#include "torch_xla/csrc/ops/view.h"

namespace torch_xla {
namespace {

bool IsNarrow(const ViewInfo& view_info) {
  return xla::util::Multiply<xla::int64>(view_info.sizes) !=
         xla::util::Multiply<xla::int64>(view_info.shape.dimensions());
}

ir::Value ApplyViewInfo(ir::Value ir_value, const ViewInfo& view_info) {
  if (IsNarrow(view_info)) {
    return ir::MakeNode<ir::ops::GenericSlice>(ir_value, view_info.indices,
                                               view_info.shape.dimensions());
  } else if (!view_info.permutation.empty()) {
    return ir::MakeNode<ir::ops::Permute>(ir_value, view_info.permutation);
  } else {
    return ir::MakeNode<ir::ops::View>(ir_value, view_info.shape.dimensions());
  }
}

ir::Value ApplyUpdate(ir::Value ir_value,
                      const Alias::UpdateData& update_data) {
  // We first bring the source IR value forward, by reshaping and slicing.
  std::vector<ir::Value> tmp_values({ir_value});
  for (size_t i = 0; i < update_data.view_infos.size(); ++i) {
    const ViewInfo& view_info = update_data.view_infos[i];
    tmp_values.push_back(ApplyViewInfo(tmp_values.back(), view_info));
  }
  // We then move backward given the source update value, by reshaping and
  // slice-updating.
  ir::Value result = update_data.ir_value;
  for (size_t i = update_data.view_infos.size(); i > 0; --i) {
    const ViewInfo& view_info = update_data.view_infos[i - 1];
    if (IsNarrow(view_info)) {
      result = ir::MakeNode<ir::ops::UpdateSlice>(tmp_values[i - 1], result,
                                                  view_info.indices);
    } else if (!view_info.permutation.empty()) {
      result = ir::MakeNode<ir::ops::Permute>(
          result, xla::InversePermutation(view_info.permutation));
    } else {
      result = ir::MakeNode<ir::ops::View>(result, view_info.sizes);
    }
  }
  return result;
}

}  // namespace

ViewInfo::ViewInfo(xla::Shape shape, std::vector<xla::int64> sizes)
    : shape(std::move(shape)),
      indices(sizes.size(), 0),
      sizes(std::move(sizes)) {}

ViewInfo::ViewInfo(std::vector<xla::int64> sizes,
                   std::vector<xla::int64> permutation, xla::PrimitiveType type)
    : shape(xla::ShapeUtil::MakeShape(type, xla::Permute(permutation, sizes))),
      indices(sizes.size(), 0),
      sizes(std::move(sizes)),
      permutation(std::move(permutation)) {}

void Alias::Update(ir::Value ir_value, std::vector<ViewInfo> view_infos) {
  updates_.push_back({std::move(ir_value), std::move(view_infos)});
  ++generation_;
}

ir::Value Alias::SyncUpdateOperations() {
  for (auto& update_data : updates_) {
    ir_value_ = ApplyUpdate(ir_value_, update_data);
  }
  updates_.clear();
  return ir_value_;
}

View::View(xla::Shape shape, std::shared_ptr<Alias> alias, ViewInfo view_info)
    : shape_(std::move(shape)), alias_(std::move(alias)) {
  view_infos_.push_back(std::move(view_info));
}

View::View(xla::Shape shape, std::shared_ptr<Alias> alias,
           std::vector<ViewInfo> view_infos)
    : view_infos_(std::move(view_infos)),
      shape_(std::move(shape)),
      alias_(std::move(alias)) {}

void View::Update(ir::Value ir_value) {
  alias_->Update(std::move(ir_value), view_infos_);
}

std::shared_ptr<View> View::CreateSubView(xla::Shape shape,
                                          ViewInfo view_info) {
  std::vector<ViewInfo> view_infos(view_infos_);
  view_infos.push_back(std::move(view_info));
  return std::make_shared<View>(std::move(shape), alias_,
                                std::move(view_infos));
}

View::IrNode View::GetViewIrNode() {
  if (IsUpToDate()) {
    return {ir_value_, false};
  }
  ir::Value update = alias_->SyncUpdateOperations();
  for (auto& view_info : view_infos_) {
    update = ApplyViewInfo(update, view_info);
  }
  ir_value_ = update;
  generation_ = alias_->generation();
  return {ir_value_, true};
}

}  // namespace torch_xla

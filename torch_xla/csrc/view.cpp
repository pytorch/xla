#include "torch_xla/csrc/view.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/generic_slice.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/resize.h"
#include "torch_xla/csrc/ops/select.h"
#include "torch_xla/csrc/ops/unselect.h"
#include "torch_xla/csrc/ops/update_slice.h"
#include "torch_xla/csrc/ops/view.h"

namespace torch_xla {
namespace {

ir::Value ApplyViewInfo(ir::Value ir_value, const ViewInfo& view_info) {
  switch (view_info.view_type) {
    case ViewInfo::Type::kSelect:
      return ir::MakeNode<ir::ops::Select>(
          ir_value, view_info.select->dim, view_info.select->start,
          view_info.select->end, view_info.select->stride);
    case ViewInfo::Type::kNarrow:
      return ir::MakeNode<ir::ops::GenericSlice>(ir_value, view_info.indices,
                                                 view_info.shape.dimensions());
    case ViewInfo::Type::kNoOp:
      return ir_value;
    case ViewInfo::Type::kPermute:
      return ir::MakeNode<ir::ops::Permute>(ir_value, view_info.permutation);
    case ViewInfo::Type::kReshape:
      return ir::MakeNode<ir::ops::View>(ir_value,
                                         view_info.shape.dimensions());
    case ViewInfo::Type::kResize:
      return ir::MakeNode<ir::ops::Resize>(ir_value,
                                           view_info.shape.dimensions());
    default:
      XLA_ERROR() << "Invalid view type: "
                  << xla::util::GetEnumValue(view_info.view_type);
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
    switch (view_info.view_type) {
      case ViewInfo::Type::kSelect:
        result = ir::MakeNode<ir::ops::Unselect>(
            tmp_values[i - 1], result, view_info.select->dim,
            view_info.select->start, view_info.select->end,
            view_info.select->stride);
        break;
      case ViewInfo::Type::kNarrow:
        result = ir::MakeNode<ir::ops::UpdateSlice>(tmp_values[i - 1], result,
                                                    view_info.indices);
        break;
      case ViewInfo::Type::kNoOp:
        break;
      case ViewInfo::Type::kPermute:
        result = ir::MakeNode<ir::ops::Permute>(
            result, xla::InversePermutation(view_info.permutation));
        break;
      case ViewInfo::Type::kReshape:
        result = ir::MakeNode<ir::ops::View>(result, view_info.sizes);
        break;
      case ViewInfo::Type::kResize:
        result = ir::MakeNode<ir::ops::Resize>(result, view_info.sizes);
        break;
      default:
        XLA_ERROR() << "Invalid view type: "
                    << xla::util::GetEnumValue(view_info.view_type);
    }
  }
  return result;
}

}  // namespace

ViewInfo::ViewInfo(Type view_type, xla::Shape shape,
                   std::vector<xla::int64> sizes)
    : view_type(view_type),
      shape(std::move(shape)),
      indices(sizes.size(), 0),
      sizes(std::move(sizes)) {}

ViewInfo::ViewInfo(Type view_type, std::vector<xla::int64> sizes,
                   std::vector<xla::int64> permutation, xla::PrimitiveType type)
    : view_type(view_type),
      shape(xla::ShapeUtil::MakeShape(type,
                                      XlaHelpers::Permute(permutation, sizes))),
      sizes(std::move(sizes)),
      permutation(std::move(permutation)) {
  XLA_CHECK(view_type == Type::kPermute);
}

ViewInfo::ViewInfo(Type view_type, const xla::Shape& source_shape,
                   SelectInfo select)
    : view_type(view_type),
      shape(ir::ops::Select::MakeSelectShape(
          source_shape, select.dim, select.start, select.end, select.stride)),
      sizes(source_shape.dimensions()),
      select(std::move(select)) {
  XLA_CHECK(view_type == Type::kSelect);
}

void Alias::Update(ir::Value ir_value, std::vector<ViewInfo> view_infos) {
  if (!updates_.empty() && updates_.back().view_infos == view_infos) {
    updates_.back().ir_value = std::move(ir_value);
  } else {
    updates_.push_back({std::move(ir_value), std::move(view_infos)});
  }
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

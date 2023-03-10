#include "torch_xla/csrc/view.h"

#include <torch/csrc/lazy/core/util.h>

#include <algorithm>
#include <functional>
#include <numeric>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/as_strided_view_update.h"
#include "torch_xla/csrc/ops/diagonal.h"
#include "torch_xla/csrc/ops/diagonal_view_update.h"
#include "torch_xla/csrc/ops/generic_slice.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/resize.h"
#include "torch_xla/csrc/ops/select.h"
#include "torch_xla/csrc/ops/unselect.h"
#include "torch_xla/csrc/ops/update_slice.h"
#include "torch_xla/csrc/ops/view.h"

namespace torch_xla {
namespace {

torch::lazy::Value ApplyViewInfo(torch::lazy::Value ir_value,
                                 const ViewInfo& view_info) {
  switch (view_info.view_type) {
    case ViewInfo::Type::kSelect:
      return torch::lazy::MakeNode<Select>(
          ir_value, view_info.select->dim, view_info.select->start,
          view_info.select->end, view_info.select->stride);
    case ViewInfo::Type::kNarrow:
      return torch::lazy::MakeNode<GenericSlice>(ir_value, view_info.indices,
                                                 view_info.shape.dimensions());
    case ViewInfo::Type::kNoOp:
      return ir_value;
    case ViewInfo::Type::kPermute:
      return torch::lazy::MakeNode<Permute>(ir_value, view_info.permutation);
    case ViewInfo::Type::kReshape:
      return torch::lazy::MakeNode<ViewOp>(
          ir_value,
          torch::lazy::ToVector<int64_t>(view_info.shape.dimensions()));
    case ViewInfo::Type::kResize:
      return torch::lazy::MakeNode<Resize>(
          ir_value,
          torch::lazy::ToVector<int64_t>(view_info.shape.dimensions()));
    case ViewInfo::Type::kAsStrided:
      return torch::lazy::MakeNode<AsStrided>(
          ir_value,
          torch::lazy::ToVector<int64_t>(view_info.shape.dimensions()),
          view_info.as_strided->stride, view_info.as_strided->offset);
    case ViewInfo::Type::kDiagonal:
      return torch::lazy::MakeNode<Diagonal>(
          ir_value, view_info.diagonal->offset, view_info.diagonal->dim1,
          view_info.diagonal->dim2);
    default:
      XLA_ERROR() << "Invalid view type: "
                  << torch::lazy::GetEnumValue(view_info.view_type);
  }
}

torch::lazy::Value ApplyUpdate(torch::lazy::Value ir_value,
                               const Alias::UpdateData& update_data) {
  // We first bring the source IR value forward, by reshaping and slicing.
  std::vector<torch::lazy::Value> tmp_values({ir_value});
  for (size_t i = 0; i < update_data.view_infos.size(); ++i) {
    const ViewInfo& view_info = update_data.view_infos[i];
    tmp_values.push_back(ApplyViewInfo(tmp_values.back(), view_info));
  }
  // We then move backward given the source update value, by reshaping and
  // slice-updating.
  torch::lazy::Value result = update_data.ir_value;
  for (size_t i = update_data.view_infos.size(); i > 0; --i) {
    const ViewInfo& view_info = update_data.view_infos[i - 1];
    switch (view_info.view_type) {
      case ViewInfo::Type::kSelect:
        result = torch::lazy::MakeNode<Unselect>(
            tmp_values[i - 1], result, view_info.select->dim,
            view_info.select->start, view_info.select->end,
            view_info.select->stride);
        break;
      case ViewInfo::Type::kNarrow:
        result = torch::lazy::MakeNode<UpdateSlice>(tmp_values[i - 1], result,
                                                    view_info.indices);
        break;
      case ViewInfo::Type::kNoOp:
        break;
      case ViewInfo::Type::kPermute:
        result = torch::lazy::MakeNode<Permute>(
            result, xla::InversePermutation(view_info.permutation));
        break;
      case ViewInfo::Type::kReshape:
        result = torch::lazy::MakeNode<ViewOp>(
            result, torch::lazy::ToVector<int64_t>(
                        view_info.source_shape.dimensions()));
        break;
      case ViewInfo::Type::kResize:
        result = torch::lazy::MakeNode<Resize>(
            result, torch::lazy::ToVector<int64_t>(
                        view_info.source_shape.dimensions()));
        break;
      case ViewInfo::Type::kAsStrided:
        result = torch::lazy::MakeNode<AsStridedViewUpdate>(
            tmp_values[i - 1], result,
            torch::lazy::ToVector<int64_t>(view_info.source_shape.dimensions()),
            view_info.as_strided->stride, view_info.as_strided->offset);
        break;
      case ViewInfo::Type::kDiagonal:
        result = torch::lazy::MakeNode<DiagonalViewUpdate>(
            tmp_values[i - 1], result, view_info.diagonal->offset,
            view_info.diagonal->dim1, view_info.diagonal->dim2);
        break;
      default:
        XLA_ERROR() << "Invalid view type: "
                    << torch::lazy::GetEnumValue(view_info.view_type);
    }
  }
  return result;
}

}  // namespace

ViewInfo::ViewInfo(Type view_type, xla::Shape shape, xla::Shape source_shape)
    : view_type(view_type),
      shape(std::move(shape)),
      indices(source_shape.rank(), 0),
      source_shape(std::move(source_shape)) {}

ViewInfo::ViewInfo(Type view_type, xla::Shape source_shape,
                   std::vector<int64_t> permutation)
    : view_type(view_type),
      shape(Permute::MakePermuteShape(source_shape, permutation)),
      source_shape(std::move(source_shape)),
      permutation(std::move(permutation)) {
  XLA_CHECK(view_type == Type::kPermute);
}

ViewInfo::ViewInfo(Type view_type, const xla::Shape& source_shape,
                   SelectInfo select)
    : view_type(view_type),
      shape(Select::MakeSelectShape(source_shape, select.dim, select.start,
                                    select.end, select.stride)),
      source_shape(source_shape),
      select(std::move(select)) {
  XLA_CHECK(view_type == Type::kSelect);
}

ViewInfo::ViewInfo(Type view_type, xla::Shape shape, xla::Shape source_shape,
                   AsStridedInfo as_strided)
    : view_type(view_type),
      shape(std::move(shape)),
      source_shape(std::move(source_shape)),
      as_strided(std::move(as_strided)) {
  XLA_CHECK(view_type == Type::kAsStrided);
}

ViewInfo::ViewInfo(Type view_type, const xla::Shape& source_shape,
                   DiagonalInfo diagonal)
    : view_type(view_type),
      shape(Diagonal::MakeDiagonalShape(source_shape, diagonal.offset,
                                        diagonal.dim1, diagonal.dim2)),
      source_shape(source_shape),
      diagonal(std::move(diagonal)) {
  XLA_CHECK(view_type == Type::kDiagonal);
}

void Alias::Update(torch::lazy::Value ir_value,
                   std::vector<ViewInfo> view_infos) {
  if (!updates_.empty() && updates_.back().view_infos == view_infos) {
    updates_.back().ir_value = std::move(ir_value);
  } else {
    updates_.push_back({std::move(ir_value), std::move(view_infos)});
  }
  ++generation_;
}

torch::lazy::Value Alias::SyncUpdateOperations() {
  for (auto& update_data : updates_) {
    ir_value_ = ApplyUpdate(ir_value_, update_data);
  }
  updates_.clear();
  return ir_value_;
}

View::View(xla::Shape shape, std::shared_ptr<Alias> alias, ViewInfo view_info)
    : shape_(std::move(shape)), alias_(std::move(alias)) {
  view_infos_.push_back(std::move(view_info));
  if (view_info.view_type == ViewInfo::Type::kNoOp) {
    ir_value_ = alias_->ir_value();
  }
}

View::View(xla::Shape shape, std::shared_ptr<Alias> alias,
           std::vector<ViewInfo> view_infos)
    : view_infos_(std::move(view_infos)),
      shape_(std::move(shape)),
      alias_(std::move(alias)) {
  bool all_view_info_no_op = true;
  for (const ViewInfo& view_info : view_infos_) {
    if (view_info.view_type != ViewInfo::Type::kNoOp) {
      all_view_info_no_op = false;
      break;
    }
  }
  if (all_view_info_no_op) {
    ir_value_ = alias_->ir_value();
  }
}

void View::Update(torch::lazy::Value ir_value) {
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
  torch::lazy::Value update = alias_->SyncUpdateOperations();
  for (auto& view_info : view_infos_) {
    update = ApplyViewInfo(update, view_info);
  }
  ir_value_ = update;
  generation_ = alias_->generation();
  return {ir_value_, true};
}

}  // namespace torch_xla

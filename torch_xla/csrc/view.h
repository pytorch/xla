#pragma once

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

struct SelectInfo {
  bool operator==(const SelectInfo& ref) const {
    return dim == ref.dim && start == ref.start && end == ref.end &&
           stride == ref.stride;
  }

  xla::int64 dim = 0;
  xla::int64 start = 0;
  xla::int64 end = 0;
  xla::int64 stride = 0;
};

struct ViewInfo {
  ViewInfo() = default;
  ViewInfo(xla::Shape shape, std::vector<xla::int64> sizes);
  ViewInfo(std::vector<xla::int64> sizes, std::vector<xla::int64> permutation,
           xla::PrimitiveType type);
  ViewInfo(const xla::Shape& source_shape, SelectInfo select);

  bool operator==(const ViewInfo& ref) const {
    return shape == ref.shape && indices == ref.indices && sizes == ref.sizes &&
           permutation == ref.permutation && select == ref.select;
  }

  // The shape of the result of a view. In case of narrowing, this represents
  // the size of the narrow slice.
  xla::Shape shape;
  // In case of narrowing, the starting indices from where the narrow slice is
  // cut.
  std::vector<xla::int64> indices;
  // The dimension sizes of the source of this view.
  std::vector<xla::int64> sizes;
  // The permutation to be used. If empty, this is not a permute operation.
  std::vector<xla::int64> permutation;
  // Information used for sliced views.
  absl::optional<SelectInfo> select;
};

// When a "view" (capture by reference) is taken on a node, an Alias object is
// created on the captured node itself, with its current IR Node value.
class Alias {
 public:
  struct UpdateData {
    ir::Value ir_value;
    std::vector<ViewInfo> view_infos;
  };

  explicit Alias(ir::Value ir_value) : ir_value_(std::move(ir_value)) {}

  const ir::Value& ir_value() const { return ir_value_; }

  const std::vector<UpdateData>& updates() const { return updates_; }

  size_t generation() const { return generation_; }

  // Appends an update to the IR value stored within the alias. The ir_value is
  // the value to be written, and view_infos represents the forward path from
  // the alias's ir_value to the update ir_value.
  void Update(ir::Value ir_value, std::vector<ViewInfo> view_infos);

  ir::Value SyncUpdateOperations();

 private:
  // The IR value which is the root at which the view was created.
  ir::Value ir_value_;
  // The stacked updates on the view. Orders matter, as most recent updates
  // might overwrite older ones.
  std::vector<UpdateData> updates_;
  // Incremented every time an update happens. Used by view to track alias
  // changes and regenerate the most current value.
  size_t generation_ = 0;
};

class View {
 public:
  struct IrNode {
    ir::Value ir_value;
    bool updated;
  };

  View(xla::Shape shape, std::shared_ptr<Alias> alias, ViewInfo view_info);
  View(xla::Shape shape, std::shared_ptr<Alias> alias,
       std::vector<ViewInfo> view_infos);

  void Update(ir::Value ir_value);

  const xla::Shape& shape() const { return shape_; }

  const std::shared_ptr<Alias>& alias() const { return alias_; }

  std::shared_ptr<View> CreateSubView(xla::Shape shape, ViewInfo view_info);

  // Extracts the current IrNode out of a view, into a IrNode structure
  // where the updated fields tells whether a new IR value has been created, or
  // the cached one returned.
  IrNode GetViewIrNode();

  bool IsUpToDate() const {
    return ir_value_ && generation_ == alias_->generation();
  }

 private:
  std::vector<ViewInfo> view_infos_;
  xla::Shape shape_;
  std::shared_ptr<Alias> alias_;
  ir::Value ir_value_;
  size_t generation_ = 0;
};

}  // namespace torch_xla

#pragma once

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AsStridedViewUpdate : public Node {
 public:
  AsStridedViewUpdate(const Value& target, const Value& input,
                      std::vector<xla::int64> size, xla::int64 storage_offset);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<xla::int64>& size() const { return size_; }

  xla::int64 storage_offset() const { return storage_offset_; }

 private:
  std::vector<xla::int64> size_;
  xla::int64 storage_offset_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla

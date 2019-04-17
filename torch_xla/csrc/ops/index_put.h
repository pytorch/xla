#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class IndexPut : public Node {
 public:
  IndexPut(const ir::Value& base, const ir::Value& indices,
           xla::int64 start_dim, const ir::Value& values, bool accumulate);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::int64 start_dim() const { return start_dim_; }

  bool accumulate() const { return accumulate_; }

 private:
  // The dimension number at which indexing starts.
  xla::int64 start_dim_;
  // Whether to accumulate instead of set.
  bool accumulate_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla

#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class TopK : public Node {
 public:
  TopK(const Value& input, xla::int64 k, xla::int64 dim, bool largest,
       bool sorted);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::int64 k() const { return k_; };

  xla::int64 dim() const { return dim_; };

  bool largest() const { return largest_; }

  bool sorted() const { return sorted_; }

 private:
  xla::int64 k_;
  xla::int64 dim_;
  bool largest_;
  bool sorted_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla

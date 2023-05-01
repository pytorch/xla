#ifndef XLA_TORCH_XLA_CSRC_OPS_DYNAMIC_IR_H_
#define XLA_TORCH_XLA_CSRC_OPS_DYNAMIC_IR_H_

#include <torch/csrc/lazy/core/dynamic_ir.h>

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "third_party/xla_client/debug_macros.h"
#include "torch/csrc/lazy/core/dynamic_ir.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {

/**
 * The goal of "dynamic" Nodes is to patch a hole in our tracing.
 * Previously, if a user called `sizes` on a Tensor, it would leak out
 * of our tracing system, as `sizes` returns a torch.Size or an int. To
 * prevent this from happening, we introduce torch::lazy::DimensionNode, a new
 * type of XlaNode that abstracts the operation of getting the dimensions of a
 * Tensor.
 *
 * Consider the following example:
 * ```
 * numel = x.shape()[0] * x.shape()[1]
 * ```
 *
 * Here, `x.shape()[i]` will be a SizeNode (subclass of DimensionNode),
 * and the multiplication of the two SizeNodes will be represented by
 * a SizeMul (also a subclass of DimensionNode). Through this, we can
 * prevent `numel` from being represented as a Python int and thus
 * burned into the Graph.
 */

// Represents the result of calling `size` on a Tensor
class SizeNode : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeNode(torch::lazy::Value input, size_t dim);
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override { return upper_bound_; }
  bool isSymbolic() const override { return true; }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  size_t dim_ = 0;
  int64_t upper_bound_;
  mutable bool dynamic_value_computed_ = false;
  // represent the runtime size of the current size node.
  mutable int64_t runtime_size_;
};

class SizeEq : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeEq(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override {
    TORCH_CHECK(false, "Comparison operators should be using getDynamicValue");
  }
  bool isSymbolic() const override { return true; }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override {
    // TODO: not sure we will ever need it?
    TORCH_CHECK(false, "Lowering comparison nodes isn't supported yet!");
  }
};

class SizeNe : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeNe(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override {
    TORCH_CHECK(false, "Comparison operators should be using getDynamicValue");
  }
  bool isSymbolic() const override { return true; }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override {
    // TODO: not sure we will ever need it?
    TORCH_CHECK(false, "Lowering comparison nodes isn't supported yet!");
  }
};

class SizeGe : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeGe(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override {
    TORCH_CHECK(false, "Comparison operators should be using getDynamicValue");
  }
  bool isSymbolic() const override { return true; }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override {
    // TODO: not sure we will ever need it?
    TORCH_CHECK(false, "Lowering comparison nodes isn't supported yet!");
  }
};

class SizeLt : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeLt(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override {
    TORCH_CHECK(false, "Comparison operators should be using getDynamicValue");
  }
  bool isSymbolic() const override { return true; }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override {
    // TODO: not sure we will ever need it?
    TORCH_CHECK(false, "Lowering comparison nodes isn't supported yet!");
  }
};

class SizeAdd : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeAdd(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override { return upper_bound_; }
  bool isSymbolic() const override { return true; }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t upper_bound_;
};

class SizeSub : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeSub(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override { return upper_bound_; }
  bool isSymbolic() const override { return true; }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t upper_bound_;
};

class SizeMul : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeMul(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override { return upper_bound_; }
  bool isSymbolic() const override { return true; }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t upper_bound_;
};

class SizeDiv : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeDiv(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override { return upper_bound_; }
  bool isSymbolic() const override { return true; }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t upper_bound_;
};

class SizeMod : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeMod(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override { return upper_bound_; }
  bool isSymbolic() const override { return true; }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t upper_bound_;
};

class SizeError : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeError();
  int64_t getDynamicValue() const override;
  int64_t getStaticValue() const override {
    XLA_CHECK(false) << "SizeError shouldn't be called.";
    return -1;
  }
  bool isSymbolic() const override {
    XLA_CHECK(false) << "SizeError shouldn't be called.";
    return true;
  }
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;
};

const torch::lazy::DimensionNode* DimCast(torch::lazy::Output output);
const torch::lazy::DimensionNode* DimCast(const torch::lazy::Node* node);
const std::shared_ptr<torch::lazy::DimensionNode> DimCast(
    const torch::lazy::NodePtr& node);

class SizeConstant : public torch_xla::Scalar,
                     public torch::lazy::DimensionNode {
 public:
  SizeConstant(int64_t val);
  int64_t getStaticValue() const override { return value().to<int64_t>(); };
  int64_t getDynamicValue() const override { return getStaticValue(); };
  bool isSymbolic() const override { return false; };
  std::string ToString() const override {
    return this->torch_xla::Scalar::ToString();
  };
  virtual XlaOpVector Lower(LoweringContext* loctx) const override {
    return torch_xla::Scalar::Lower(loctx);
  };
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_DYNAMIC_IR_H_

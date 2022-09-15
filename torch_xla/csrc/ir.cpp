#include "torch_xla/csrc/ir.h"

#include <functional>
#include <sstream>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "torch/csrc/lazy/core/config.h"
#include "torch/csrc/lazy/core/hash.h"
#include "torch/csrc/lazy/core/ir_metadata.h"
#include "torch/csrc/lazy/python/python_util.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace {

using ShapeCache =
    xla::util::Cache<torch::lazy::hash_t, xla::Shape, torch::lazy::HashReducer>;

ShapeCache* GetShapeCache() {
  static int64_t shape_cache_size =
      xla::sys_util::GetEnvInt("XLA_IR_SHAPE_CACHE_SIZE", 4096);
  static ShapeCache* cache = new ShapeCache(shape_cache_size);
  return cache;
}

torch::lazy::hash_t GetOperandHashes(const torch::lazy::OpList& operands,
                                     const torch::lazy::hash_t& node_hash) {
  torch::lazy::hash_t hash = node_hash;
  for (auto& operand : operands) {
    if (!operand) {
      hash = torch::lazy::HashCombine(
          hash, static_cast<uint64_t>(torch::lazy::kNullOpt));
      continue;
    }
    hash = torch::lazy::HashCombine(hash, operand.hash());
  }
  return hash;
}

}  // namespace

XlaNode::XlaNode(torch::lazy::OpKind op, torch::lazy::OpList operands,
                 std::vector<torch::lazy::Shape>&& shapes, xla::Shape xla_shape,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : torch::lazy::Node(op, operands, std::move(shapes), num_outputs),
      xla_shape_(std::move(xla_shape)),
      node_hash_(torch::lazy::HashCombine(op.hash(), hash_seed)),
      dag_hash_(GetOperandHashes(operands, node_hash_)) {
  std::cout << "WONJOO: XlaNode::XlaNode1" << std::endl;
}

XlaNode::XlaNode(torch::lazy::OpKind op, torch::lazy::OpList operands,
                 std::vector<torch::lazy::Shape>&& shapes,
                 const std::function<xla::Shape()>& xla_shape_fn,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : torch::lazy::Node(op, operands, std::move(shapes), num_outputs),
      node_hash_(torch::lazy::HashCombine(op.hash(), hash_seed)),
      dag_hash_(GetOperandHashes(operands, node_hash_)) {
  xla_shape_ = GetOpShape(xla_shape_fn);
  std::cout << "WONJOO: XlaNode::XlaNode2" << std::endl;
}

XlaNode::XlaNode(torch::lazy::OpKind op, torch::lazy::OpList operands,
                 torch::lazy::Shape shape, xla::Shape xla_shape,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : torch::lazy::Node(op, operands, std::vector<torch::lazy::Shape>{shape},
                        num_outputs),
      xla_shape_(std::move(xla_shape)),
      node_hash_(torch::lazy::HashCombine(op.hash(), hash_seed)),
      dag_hash_(GetOperandHashes(operands, node_hash_)) {
  std::cout << "WONJOO: XlaNode::XlaNode3" << std::endl;
}

XlaNode::XlaNode(torch::lazy::OpKind op, torch::lazy::OpList operands,
                 xla::Shape xla_shape, size_t num_outputs,
                 torch::lazy::hash_t hash_seed)
    : XlaNode(op, operands, std::vector<torch::lazy::Shape>{}, xla_shape,
              num_outputs, hash_seed) {
  std::cout << "WONJOO: XlaNode::XlaNode4" << std::endl;
}

XlaNode::XlaNode(torch::lazy::OpKind op, torch::lazy::OpList operands,
                 const std::function<torch::lazy::Shape()>& shape_fn,
                 const std::function<xla::Shape()>& xla_shape_fn,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : XlaNode(std::move(op), operands, xla::Shape(), num_outputs, hash_seed) {
  // Forward the constructor to the one above (with empty shape), so we have the
  // full hash information, then fetch/compute the real shape.
  addComputedShape(shape_fn);
  xla_shape_ = GetOpShape(xla_shape_fn);
  std::cout << "WONJOO: XlaNode::XlaNode5" << std::endl;
}

XlaNode::XlaNode(torch::lazy::OpKind op, torch::lazy::OpList operands,
                 const std::function<xla::Shape()>& xla_shape_fn,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : XlaNode(std::move(op), operands, xla::Shape(), num_outputs, hash_seed) {
  // Forward the constructor to the one above (with empty shape), so we have the
  // full hash information, then fetch/compute the real shape.
  xla_shape_ = GetOpShape(xla_shape_fn);
  std::cout << "WONJOO: XlaNode::XlaNode6" << std::endl;
}

XlaNode::XlaNode(torch::lazy::OpKind op, torch::lazy::Shape shape,
                 xla::Shape xla_shape, size_t num_outputs,
                 torch::lazy::hash_t hash_seed)
    : torch::lazy::Node(op, shape, num_outputs),
      xla_shape_(std::move(xla_shape)),
      node_hash_(GetOpHash(op, xla_shape_, hash_seed)),
      dag_hash_(node_hash_) {
  std::cout << "WONJOO: XlaNode::XlaNode7" << std::endl;
  std::cout << "WONJOO: XlaNode::XlaNode7-2, xla_shape_.ToString(true)="
            << xla_shape_.ToString(true) << std::endl;
  std::cout << "WONJOO: XlaNode::XlaNode7-3, xla_shape_.has_layout()="
            << xla_shape_.has_layout() << std::endl;
}

XlaNode::XlaNode(torch::lazy::OpKind op, xla::Shape xla_shape,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : XlaNode(op, torch::lazy::Shape(), xla_shape, num_outputs, hash_seed) {
  std::cout << "WONJOO: XlaNode::XlaNode8" << std::endl;
  std::cout << "WONJOO: XlaNode::XlaNode8-2, xla_shape_.ToString(true)="
            << xla_shape_.ToString(true) << std::endl;
  std::cout << "WONJOO: XlaNode::XlaNode8-3, xla_shape_.has_layout()="
            << xla_shape_.has_layout() << std::endl;
}

XlaNode::~XlaNode() {}

const xla::Shape& XlaNode::xla_shape(size_t output_index) const {
  if (xla_shape_.IsTuple()) {
    return xla_shape_.tuple_shapes(output_index);
  }
  XLA_CHECK_EQ(output_index, 0);
  return xla_shape_;
}

XlaOpVector XlaNode::ReturnOp(xla::XlaOp op, LoweringContext* loctx) const {
  XLA_CHECK_EQ(num_outputs(), 1);
  loctx->AssignOutputOp(torch::lazy::Output(this), op);
  return XlaOpVector({std::move(op)});
}

XlaOpVector XlaNode::ReturnOps(absl::Span<const xla::XlaOp> ops,
                               LoweringContext* loctx) const {
  XLA_CHECK_EQ(num_outputs(), ops.size());
  XlaOpVector result;
  for (size_t i = 0; i < ops.size(); ++i) {
    loctx->AssignOutputOp(torch::lazy::Output(this, i), ops[i]);
    result.push_back(ops[i]);
  }
  return result;
}

torch::lazy::NodePtr XlaNode::Clone(torch::lazy::OpList operands) const {
  XLA_ERROR() << "Cloning not implemented for node: " << *this;
}

XlaOpVector XlaNode::Lower(LoweringContext* loctx) const {
  XLA_ERROR() << "Lowering not implemented for node: " << *this;
}

torch::lazy::hash_t XlaNode::GetOpHash(torch::lazy::OpKind op,
                                       const xla::Shape& shape,
                                       torch::lazy::hash_t hash_seed) {
  torch::lazy::hash_t h =
      torch::lazy::HashCombine(op.hash(), torch::lazy::Hash(shape.ToString()));
  return torch::lazy::HashCombine(h, hash_seed);
}

xla::Shape XlaNode::GetOpShape(
    const std::function<xla::Shape()>& shape_fn) const {
  ShapeCache* shape_cache = GetShapeCache();
  auto shape = shape_cache->Get(hash());
  if (shape == nullptr) {
    shape = shape_cache->Add(hash(), std::make_shared<xla::Shape>(shape_fn()));
  }
  std::cout << "WONJOO: XlaNode::GetOpShape, shape->ToString(true)="
            << shape->ToString(true) << std::endl;
  std::cout << "WONJOO: XlaNode::GetOpShape, shape->has_layout()="
            << shape->has_layout() << std::endl;
  return *shape;
}

const xla::Shape& GetXlaShape(const torch::lazy::Value& value) {
  XlaNode* casted = dynamic_cast<XlaNode*>(value.node.get());
  const xla::Shape& ret = casted->xla_shape(value.index);
  std::cout << "WONJOO: at ir.cpp::GetXlaShape, shape.ToString(true)="
            << ret.ToString(true) << std::endl;
  std::cout << "WONJOO: at ir.cpp::GetXlaShape, shape.has_layout()="
            << ret.has_layout() << std::endl;
  return ret;
}

}  // namespace torch_xla

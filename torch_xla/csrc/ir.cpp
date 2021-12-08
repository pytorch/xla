#include "torch_xla/csrc/ir.h"

#include <functional>
#include <sstream>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "torch/csrc/lazy/core/hash.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace {

using ShapeCache =
    xla::util::Cache<torch::lazy::hash_t, xla::Shape, torch::lazy::HashReducer>;

struct ScopeEntry {
  std::string name;
  size_t saved_next_id = 1;
};

struct ScopeContext {
  std::vector<ScopeEntry> scopes;
  size_t next_id = 1;
};

thread_local ScopeContext g_scope_context;

void PushScope(const std::string& name) {
  size_t id = g_scope_context.next_id;
  g_scope_context.scopes.push_back(
      {absl::StrCat(name, ".", id), g_scope_context.next_id + 1});
  g_scope_context.next_id = 1;
}

void PopScope() {
  XLA_CHECK(!g_scope_context.scopes.empty());
  g_scope_context.next_id = g_scope_context.scopes.back().saved_next_id;
  g_scope_context.scopes.pop_back();
}

void ResetScopeContext() {
  XLA_CHECK_EQ(g_scope_context.scopes.size(), 0);
  g_scope_context.next_id = 1;
}

std::string GetCurrentScope() {
  std::string scope;
  for (auto& scope_entry : g_scope_context.scopes) {
    if (scope.empty()) {
      absl::StrAppend(&scope, scope_entry.name);
    } else {
      absl::StrAppend(&scope, "/", scope_entry.name);
    }
  }
  return scope;
}

ShapeCache* GetShapeCache() {
  static xla::int64_t shape_cache_size =
      xla::sys_util::GetEnvInt("XLA_IR_SHAPE_CACHE_SIZE", 4096);
  static ShapeCache* cache = new ShapeCache(shape_cache_size);
  return cache;
}
}  // namespace

bool Use::operator<(const Use& rhs) const {
  if (node->op() != rhs.node->op()) {
    return node->op() < rhs.node->op();
  }
  if (operand_index != rhs.operand_index) {
    return operand_index < rhs.operand_index;
  }
  return index < rhs.index;
}

std::string Use::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ", operand_index=" << operand_index
     << ", index=" << index;
  return ss.str();
}

size_t Output::Hasher::operator()(const Output& output) const {
  return torch::lazy::StdHashCombine(
      reinterpret_cast<std::ptrdiff_t>(output.node), output.index);
}

const xla::Shape& Output::shape() const { return node->shape(index); }

const xla::Shape& Output::node_shape() const { return node->shape(); }

torch::lazy::hash_t Output::hash() const {
  return torch::lazy::HashCombine(node->hash(), index);
}

std::string Output::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ", index=" << index;
  return ss.str();
}

const xla::Shape& Value::shape() const { return node->shape(index); }

const xla::Shape& Value::node_shape() const { return node->shape(); }

torch::lazy::hash_t Value::hash() const {
  return torch::lazy::HashCombine(node->hash(), index);
}

OpKind OpKind::Get(const std::string& name) {
  return OpKind(c10::Symbol::fromQualString(name));
}

torch::lazy::hash_t OpKind::hash() const {
  return torch::lazy::StringHash(op.toQualString());
}

Node::Node(OpKind op, OpList operands, xla::Shape shape, size_t num_outputs,
           torch::lazy::hash_t hash_seed)
    : op_(std::move(op)),
      num_outputs_(num_outputs),
      shape_(std::move(shape)),
      node_hash_(torch::lazy::HashCombine(op_.hash(), hash_seed)),
      hash_(node_hash_) {
  metadata_.scope = GetCurrentScope();
  metadata_.frame_info = GetFrameInfo();
  for (auto& operand : operands) {
    AddOperand(operand.node, operand.index);
    hash_ = torch::lazy::HashCombine(hash_, operand.hash());
  }
}

Node::Node(OpKind op, OpList operands,
           const std::function<xla::Shape()>& shape_fn, size_t num_outputs,
           torch::lazy::hash_t hash_seed)
    : Node(std::move(op), operands, xla::Shape(), num_outputs, hash_seed) {
  // Forward the constructor to the one above (with empty shape), so we have the
  // full hash information, then fetch/compute the real shape.
  shape_ = GetOpShape(shape_fn);
}

Node::Node(OpKind op, xla::Shape shape, size_t num_outputs,
           torch::lazy::hash_t hash_seed)
    : op_(std::move(op)),
      num_outputs_(num_outputs),
      shape_(std::move(shape)),
      node_hash_(GetOpHash(op_, shape_, hash_seed)),
      hash_(node_hash_) {
  metadata_.scope = GetCurrentScope();
  metadata_.frame_info = GetFrameInfo();
}

Node::~Node() {
  for (size_t i = 0; i < operands_as_outputs_.size(); ++i) {
    operands_[i]->RemoveUse(Use(this, i, operands_as_outputs_[i].index));
  }
}

const xla::Shape& Node::shape(size_t output_index) const {
  if (shape_.IsTuple()) {
    return shape_.tuple_shapes(output_index);
  }
  XLA_CHECK_EQ(output_index, 0);
  return shape_;
}

void Node::AddOperand(NodePtr node, size_t index) {
  XLA_CHECK_LT(index, node->num_outputs());
  operands_.push_back(std::move(node));
  operands_as_outputs_.push_back(Output(operands_.back().get(), index));
  operands_.back()->AddUse(Use(this, operands_.size() - 1, index));
}

void Node::ReplaceOperand(size_t operand_no, NodePtr node, size_t index) {
  XLA_CHECK_LT(index, node->num_outputs());
  Output* output = &operands_as_outputs_.at(operand_no);
  operands_[operand_no]->RemoveUse(Use(this, operand_no, output->index));
  node->AddUse(Use(this, operand_no, index));
  *output = Output(node.get(), index);
  operands_[operand_no] = std::move(node);
}

void Node::ReplaceAllUsesWith(NodePtr node, size_t index) {
  // A call to ReplaceOperand() will end up calling RemoveUse() into the
  // current node, so snapshot the current uses and iterate over them.
  std::vector<Use> current_uses(uses_.begin(), uses_.end());
  for (auto& use : current_uses) {
    use.node->ReplaceOperand(use.operand_index, node, index);
  }
}

XlaOpVector Node::ReturnOp(xla::XlaOp op, LoweringContext* loctx) const {
  XLA_CHECK_EQ(num_outputs(), 1);
  loctx->AssignOutputOp(Output(this), op);
  return XlaOpVector({std::move(op)});
}

XlaOpVector Node::ReturnOps(absl::Span<const xla::XlaOp> ops,
                            LoweringContext* loctx) const {
  XLA_CHECK_EQ(num_outputs(), ops.size());
  XlaOpVector result;
  for (size_t i = 0; i < ops.size(); ++i) {
    loctx->AssignOutputOp(Output(this, i), ops[i]);
    result.push_back(ops[i]);
  }
  return result;
}

std::string Node::ToString() const {
  std::stringstream ss;
  ss << shape() << " " << op();
  if (num_outputs() > 1) {
    ss << ", num_outputs=" << num_outputs();
  }
  if (!metadata_.scope.empty()) {
    ss << ", scope=" << metadata_.scope;
  }
  torch::lazy::EmitShortFrameInfo(ss, metadata_.frame_info);
  return ss.str();
}

NodePtr Node::Clone(OpList operands) const {
  XLA_ERROR() << "Cloning not implemented for node: " << *this;
}

XlaOpVector Node::Lower(LoweringContext* loctx) const {
  XLA_ERROR() << "Lowering not implemented for node: " << *this;
}

torch::lazy::hash_t Node::GetOpHash(OpKind op, const xla::Shape& shape,
                                    torch::lazy::hash_t hash_seed) {
  torch::lazy::hash_t h =
      torch::lazy::HashCombine(op.hash(), torch::lazy::Hash(shape.ToString()));
  return torch::lazy::HashCombine(h, hash_seed);
}

xla::Shape Node::GetOpShape(const std::function<xla::Shape()>& shape_fn) const {
  ShapeCache* shape_cache = GetShapeCache();
  auto shape = shape_cache->Get(hash());
  if (shape == nullptr) {
    shape = shape_cache->Add(hash(), std::make_shared<xla::Shape>(shape_fn()));
  }
  return *shape;
}

std::vector<torch::lazy::SourceLocation> Node::GetFrameInfo() {
  // At the time of writing, retrieving Python frames costs from 1us up to 20us.
  // This per IR Node. Since it is not unreasonable to have a many hundreds of
  // IR Node, this can be a multi-millisecond cost, which is not negligible.
  static bool wants_frames = xla::sys_util::GetEnvBool("XLA_IR_DEBUG", false);
  return wants_frames ? GetPythonFrames()
                      : std::vector<torch::lazy::SourceLocation>();
}

ScopePusher::ScopePusher(const std::string& name) { PushScope(name); }

ScopePusher::~ScopePusher() { PopScope(); }

void ScopePusher::ResetScopes() { ResetScopeContext(); }

}  // namespace ir
}  // namespace torch_xla

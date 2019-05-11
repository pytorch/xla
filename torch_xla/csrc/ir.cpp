#include "torch_xla/csrc/ir.h"

#include <functional>
#include <sstream>

#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace {

using ShapeCache = xla::util::Cache<size_t, xla::Shape>;

ShapeCache* GetShapeCache() {
  static const size_t kMaxShapeCacheSize = 1024;
  static ShapeCache* cache = new ShapeCache(kMaxShapeCacheSize);
  return cache;
}

void EmitShortFrameInfo(std::ostream& stream,
                        const std::vector<SourceLocation>& frames) {
  if (!frames.empty()) {
    const SourceLocation& frame = frames.front();
    std::string::size_type pos = frame.file.find_last_of('/');
    if (pos == std::string::npos) {
      pos = 0;
    } else {
      ++pos;
    }
    stream << ", location=" << frame.function << "@" << frame.file.substr(pos)
           << ":" << frame.line;
  }
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
  return xla::util::HashCombine(reinterpret_cast<std::ptrdiff_t>(output.node),
                                output.index);
}

const xla::Shape& Output::shape() const { return node->shape(index); }

const xla::Shape& Output::node_shape() const { return node->shape(); }

size_t Output::hash() const {
  return xla::util::HashCombine(node->hash(), index);
}

std::string Output::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ", index=" << index;
  return ss.str();
}

const xla::Shape& Value::shape() const { return node->shape(index); }

const xla::Shape& Value::node_shape() const { return node->shape(); }

size_t Value::hash() const {
  return xla::util::HashCombine(node->hash(), index);
}

OpKind OpKind::Get(const std::string& name) {
  return OpKind(c10::Symbol::fromQualString(name));
}

size_t OpKind::hash() const { return xla::util::StringHash(op.toQualString()); }

Node::Node(OpKind op, OpList operands, xla::Shape shape, size_t num_outputs,
           size_t hash_seed)
    : op_(std::move(op)),
      num_outputs_(num_outputs),
      shape_(std::move(shape)),
      node_hash_(xla::util::HashCombine(op_.hash(), hash_seed)),
      hash_(node_hash_) {
  metadata_.frame_info = GetFrameInfo();
  for (auto& operand : operands) {
    AddOperand(operand.node, operand.index);
    hash_ = xla::util::HashCombine(hash_, operand.hash());
  }
}

Node::Node(OpKind op, OpList operands,
           const std::function<xla::Shape()>& shape_fn, size_t num_outputs,
           size_t hash_seed)
    : Node(std::move(op), operands, xla::Shape(), num_outputs, hash_seed) {
  // Forward the constructor to the one above (with empty shape), so we have the
  // full hash information, then fetch/compute the real shape.
  shape_ = GetOpShape(shape_fn);
}

Node::Node(OpKind op, xla::Shape shape, size_t num_outputs, size_t hash_seed)
    : op_(std::move(op)),
      num_outputs_(num_outputs),
      shape_(std::move(shape)),
      node_hash_(GetOpHash(op_, shape_, hash_seed)),
      hash_(node_hash_) {
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

XlaOpVector Node::ReturnOps(tensorflow::gtl::ArraySlice<const xla::XlaOp> ops,
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
  EmitShortFrameInfo(ss, metadata_.frame_info);
  return ss.str();
}

XlaOpVector Node::Lower(LoweringContext* loctx) const {
  XLA_ERROR() << "Lowering not implemented for node: " << *this;
}

size_t Node::GetOpHash(OpKind op, const xla::Shape& shape, size_t hash_seed) {
  size_t h =
      xla::util::HashCombine(op.hash(), xla::util::Hash(shape.ToString()));
  return xla::util::HashCombine(h, hash_seed);
}

xla::Shape Node::GetOpShape(const std::function<xla::Shape()>& shape_fn) const {
  ShapeCache* shape_cache = GetShapeCache();
  auto shape = shape_cache->Get(hash());
  if (shape == nullptr) {
    shape = shape_cache->Add(hash(), std::make_shared<xla::Shape>(shape_fn()));
  }
  return *shape;
}

std::vector<SourceLocation> Node::GetFrameInfo() {
  // At the time of writing, retrieving Python frames costs from 1us up to 20us.
  // This per IR Node. Since it is not unreasonable to have a many hundreds of
  // IR Node, this can be a multi-millisecond cost, which is not negligible.
  static bool wants_frames = xla::sys_util::GetEnvBool("XLA_IR_DEBUG", false);
  return wants_frames ? GetPythonFrames() : std::vector<SourceLocation>();
}

}  // namespace ir
}  // namespace torch_xla

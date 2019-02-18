#include "torch_xla/csrc/ir_dump_util.h"

#include <regex>
#include <sstream>
#include <unordered_map>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/ir_util.h"

namespace torch_xla {
namespace ir {
namespace {

using NodeIdMap = std::unordered_map<const Node*, size_t>;

struct AttrTag {
  std::string name;
  std::string value;
  std::string::size_type pos;
};

absl::optional<AttrTag> ParseAttrTag(const std::string& node_string,
                                     std::string::size_type pos) {
  const std::regex tag_regex("^([a-zA-Z0-9_]+)=");
  std::smatch match;
  if (!std::regex_search(node_string.begin() + pos, node_string.end(), match,
                         tag_regex)) {
    return absl::nullopt;
  }

  std::string::size_type vpos = match[1].second - node_string.begin() + 1;
  int nested_open = -1;
  int nested_close = -1;
  size_t nest_count = 1;
  AttrTag tag;
  tag.name = match[1].str();
  for (pos = vpos; pos < node_string.size(); ++pos) {
    if (nested_open < 0) {
      if (node_string.compare(pos, 2, ", ") == 0) {
        break;
      }
      switch (node_string[pos]) {
        case '(':
          nested_open = node_string[pos];
          nested_close = ')';
          break;
        case '[':
          nested_open = node_string[pos];
          nested_close = ']';
          break;
        case '{':
          nested_open = node_string[pos];
          nested_close = '}';
          break;
      }
    } else if (node_string[pos] == nested_close) {
      --nest_count;
      if (nest_count == 0) {
        nest_count = 1;
        nested_open = nested_close = -1;
      }
    } else if (node_string[pos] == nested_open) {
      ++nest_count;
    }
  }
  tag.value = node_string.substr(vpos, pos - vpos);
  tag.pos = pos;
  return tag;
}

NodeIdMap GenerateIdMap(
    tensorflow::gtl::ArraySlice<const Node* const> post_order) {
  NodeIdMap id_map;
  for (auto node : post_order) {
    XLA_CHECK(id_map.emplace(node, id_map.size()).second) << node->ToString();
  }
  return id_map;
}

std::string GenerateDotNodeLabel(const Node* node) {
  static const size_t kMaxValueSize = 64;
  std::stringstream ss;
  ss << node->op() << "\\n" << node->shape();

  std::string node_string = node->ToString();
  std::string op_string = node->op().ToString();
  std::string::size_type pos = node_string.find(op_string);
  XLA_CHECK_NE(pos, std::string::npos) << node_string << " : " << op_string;
  pos += op_string.size();
  for (;;) {
    if (node_string.compare(pos, 2, ", ") == 0) {
      pos += 2;
    }
    auto tag = ParseAttrTag(node_string, pos);
    if (!tag) {
      break;
    }
    ss << "\\n" << tag->name << "=";
    if (tag->value.size() < kMaxValueSize) {
      ss << tag->value;
    } else {
      ss << tag->value.substr(0, kMaxValueSize) << "...";
    }
    pos = tag->pos;
  }
  return ss.str();
}

std::string GenerateDotNodeSpec(const Node* node) {
  std::stringstream ss;
  ss << "label=\"" << GenerateDotNodeLabel(node) << "\"";
  return ss.str();
}

}  // namespace

std::string DumpUtil::ToDot(
    tensorflow::gtl::ArraySlice<const Node* const> nodes) {
  auto post_order = Util::ComputePostOrder(nodes);
  NodeIdMap id_map = GenerateIdMap(post_order);
  std::stringstream ss;
  ss << "digraph G {\n";
  for (auto node : post_order) {
    ss << "  node" << id_map.at(node) << " [" << GenerateDotNodeSpec(node)
       << "]\n";
  }
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    const Node* node = *it;
    size_t id = id_map.at(node);
    for (auto& output : node->operands()) {
      ss << "  node" << id_map.at(output.node) << " -> node" << id;
      if (output.node->num_outputs() > 1) {
        ss << " [label=\"" << output.index << "\"]";
      }
      ss << "\n";
    }
  }
  ss << "}\n";
  return ss.str();
}

}  // namespace ir
}  // namespace torch_xla

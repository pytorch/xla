#include "tensorflow/compiler/xla/xla_client/color_output.h"

namespace xla {
namespace torch_xla {

namespace {

std::string join(const std::vector<std::string> &pieces,
                 const std::string &delimiter) {
  std::stringstream ss;
  std::size_t i = 0;
  for (const auto &s : pieces) {
    if (i++) ss << delimiter;
    ss << s;
  }
  return ss.str();
}

}  // end of anonymous namespace

static const char *prev_char(const char *original, const char *start, char c) {
  while (start > original && *start != c) {
    --start;
  }
  return start;
}

std::string short_fn_name(const std::string &fn_name) {
  std::string result = fn_name;
  const char *start = fn_name.c_str();
  const char *s = strchr(start, '(');
  if (s && *s && s > start) {
    if (const char *s0 = prev_char(start, s - 1, ' ')) {
      if (*s0 == ' ') {
        ++s0;
      }
      const size_t sz = s - s0 + 1;
      result = std::string(s0, sz);
      result.append(")");
    }
  }
  return result;
}

#ifdef WSE_DEBUG_LOGGING
__thread int EnterLeave::depth_ = 0;
const std::string EnterLeave::library_ = "ptxla";
const ::xla::torch_xla::Color EnterLeave::library_color_ =
    ::xla::torch_xla::Color::FG_BLUE;
std::mutex EnterLeave::mtx_;
#endif

}  // namespace torch_xla
}  // namespace xla
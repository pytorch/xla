#pragma once

#include <sys/syscall.h>

#include <memory>
#include <sstream>
#include <stdexcept>

#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "tensorflow/compiler/xla/xla_client/computation_client_manager.h"

namespace xla {
namespace torch_xla {

enum class Color {
  BG_INVALID = -1,
  FG_RESET = 0,
  FG_RED = 31,
  FG_GREEN = 32,
  FG_YELLOW = 33,
  FG_BLUE = 34,
  FG_MAGENTA = 35,
  FG_CYAN = 36,
  FG_WHITE = 37,
  FG_DEFAULT = 39,
  BG_RED = 41,
  BG_GREEN = 42,
  BG_BLUE = 44,
  BG_MAGENTA = 45,
  BG_CYAN = 46,
  BG_WHITE = 47,
  BG_DEFAULT = 49,
};

class ColorModifier {
  const Color code;
  const bool bright;

 public:
  ColorModifier(Color pCode, const bool is_bright = true)
      : code(pCode), bright(is_bright) {}

  friend std::ostream &operator<<(std::ostream &os, const ColorModifier &mod) {
    return os << "\033[" << (mod.bright ? "1;" : "0;") << (int)mod.code << "m";
  }
};

class ColorScope {
  std::ostream &os_;

 public:
  inline ColorScope(std::ostream &os, Color pCode, bool bright = true)
      : os_(os) {
    ColorModifier mod(pCode, bright);
    os << mod;
  }
  inline ColorScope(std::ostream &os, std::vector<Color> codes,
                    bool bright = false)
      : os_(os) {
    for (auto c : codes) {
      ColorModifier mod(c, bright);
      os << mod;
    }
  }
  ColorScope(Color pCode, bool bright = true) : os_(std::cout) {
    os_ << ColorModifier(pCode, bright) << std::flush;
  }
  ~ColorScope() {
    os_ << ColorModifier(Color::FG_DEFAULT) << ColorModifier(Color::FG_DEFAULT)
        << ColorModifier(Color::BG_DEFAULT);
  }
};

template <typename T>
inline std::string to_string(const T &obj) {
  std::stringstream ss;
  ss << obj;
  return std::move(ss.str());
}

#define WSE_DEBUG_LOGGING

#ifdef WSE_DEBUG_LOGGING

class EnterLeave {
  static __thread int depth_;
  static const std::string library_;
  static const Color library_color_;
  const std::string label_;
  const pid_t thread_id_;
  const bool both_;
  const Color use_color_;
  static std::mutex mtx_;

 public:
  static std::string concat(const char *s0, const char *s1, const char *s2) {
    std::string s;
    if (s0 && *s0) {
      s = s0;
      s += "::";
    }
    if (s1) {
      s += s1;
    }
    if (s2 && *s2) {
      s += " (";
      s += s2;
      s += ")";
    }
    return s;
  }
  inline EnterLeave(const std::string &label, bool both = true,
                    const Color use_color = Color::BG_INVALID)
      : label_(label),
        thread_id_(syscall(SYS_gettid)),
        both_(both),
        use_color_(use_color == Color::BG_INVALID ? library_color_
                                                  : use_color) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (int x = 0; x < depth_; ++x) {
      printf("  ");
    }
    ColorScope color_scope(use_color_);
    printf("%s[tid=%d (%s)]: %s\n", both_ ? "ENTER" : "HERE", thread_id_,
           library_.c_str(), label.c_str());
    fflush(stdout);
    ++depth_;
  }
  inline ~EnterLeave() {
    std::lock_guard<std::mutex> lk(mtx_);
    --depth_;
    if (both_) {
      ColorScope color_scope(use_color_);
      for (int x = 0; x < depth_; ++x) {
        printf("  ");
      }
      printf("LEAVE[tid=%d (%s)]: %s\n", thread_id_, library_.c_str(),
             label_.c_str());
      fflush(stdout);
    }
  }
};
#else

class EnterLeave {
 public:
  inline EnterLeave(const std::string &label, bool both = true) {}
};

#endif  // WSE_DEBUG_LOGGING

#ifdef WSE_DEBUG_LOGGING

std::string short_fn_name(const std::string &fn_name);

#define HEREC(__color$)                                                 \
  EnterLeave __here(                                                    \
      EnterLeave::concat(                                               \
          nullptr,                                                      \
          ::xla::torch_xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), \
          ""                                                            \
          ""),                                                          \
      true, __color$)
#define HERE()                                                               \
  EnterLeave __here(EnterLeave::concat(                                      \
      nullptr, ::xla::torch_xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), \
      ""))
#define HEREX()                                                              \
  EnterLeave __here(                                                         \
      EnterLeave::concat(                                                    \
          nullptr,                                                           \
          ::xla::torch_xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), ""), \
      false)
#define HEREXC(__color$)                                                     \
  EnterLeave __here(                                                         \
      EnterLeave::concat(                                                    \
          nullptr,                                                           \
          ::xla::torch_xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), ""), \
      false, __color$)
#define HEREXCT(__color$)                                                    \
  EnterLeave __here(                                                         \
      EnterLeave::concat(                                                    \
          std::to_string(this).c_str(),                                      \
          ::xla::torch_xla::short_fn_name(__PRETTY_FUNCTION__).c_str(), ""), \
      false, __color$)
#else
#define HERE() ((void)0)
#define HEREX() ((void)0)
#endif

#define ENDL std::endl << std::flush

}  // namespace torch_xla
}  // namespace xla

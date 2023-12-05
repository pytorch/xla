#ifndef XLA_CLIENT_ENV_VARS_H_
#define XLA_CLIENT_ENV_VARS_H_

#include <optional>

namespace torch_xla {
namespace runtime {

struct DeviceCapabilities {
  bool supports_float64;
  bool supports_bool;
  // TODO figure out https://github.com/pytorch/xla/blob/2c6e4a773cc70cdea3c606d410b3aef8f7dfb6f7/torch_xla/csrc/resize_ops.cpp#L268
  std::optional<int32_t> dense_gather_factor; // TODO better name
  std::optional<int32_t> dense_scatter_factor; // TODO better name
  // TODO figure out https://github.com/pytorch/xla/blob/2c6e4a773cc70cdea3c606d410b3aef8f7dfb6f7/torch_xla/csrc/aten_xla_type.cpp#L3156
  std::optional<std::string> default_rng_bit_generator_name;
};

}
}

#endif

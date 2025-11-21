#include "xla_generator.h"

#include <cstring>
#include <deque>
#include <vector>

#include <ATen/Functions.h>
#include <ATen/core/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/CallOnce.h>
#include <c10/util/intrusive_ptr.h>

#include "absl/status/status.h"

#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/status.h"

namespace at {

namespace detail {

namespace {

// Total number of XLA devices in the system.
static int64_t num_xla_devices;

// Ensures default_gens_xla is initialized once.
static std::deque<c10::once_flag> xla_gens_init_flag;

// Default, global XLA generators, one per XLA device.
static std::vector<at::Generator> default_gens_xla;

/*
 * Populates the global variables related to XLA generators
 * Warning: this function must only be called once!
 */
static absl::Status InitGlobalVars() {
  static const absl::Status* init_status = new absl::Status([]() {
    XLA_ASSIGN_OR_RETURN(auto c_client,
                         torch_xla::runtime::GetComputationClient());
    num_xla_devices = static_cast<int64_t>(c_client->GetNumDevices());
    xla_gens_init_flag.resize(num_xla_devices);
    default_gens_xla.resize(num_xla_devices);
    return absl::OkStatus();
  }());
  return *init_status;
}

// Validates and normalizes an XLA device index.
// If requested_index == -1, the current device index is used.
// Returns InvalidArgument if the resolved index is out of range.
static absl::StatusOr<c10::DeviceIndex> NormalizeXLADeviceIndex(
    c10::DeviceIndex requested_index) {
  c10::DeviceIndex idx = requested_index;
  if (idx == -1) {
    idx = torch_xla::bridge::GetCurrentAtenDevice().index();
  }
  if (idx < 0 || idx >= num_xla_devices) {
    return absl::InvalidArgumentError(
        "Invalid device index for XLA generator. Provided index: " +
        std::to_string(idx));
  }
  return idx;
}

}  // anonymous namespace

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * GetDefaultXLAGenerator gets the default generator for a particular
 * XLA device.
 */
absl::StatusOr<const at::Generator&> GetDefaultXLAGenerator(
    c10::DeviceIndex device_index) {
  XLA_RETURN_IF_ERROR(InitGlobalVars(), "Failed to initialize XLA generators");
  // Normalize and validate the target device index; default to current device
  // when unspecified
  XLA_ASSIGN_OR_RETURN(c10::DeviceIndex idx,
                       NormalizeXLADeviceIndex(device_index),
                       "Invalid XLA device index");
  c10::call_once(xla_gens_init_flag[idx], [&] {
    default_gens_xla[idx] = at::make_generator<XLAGeneratorImpl>(idx);
    default_gens_xla[idx].seed();
  });
  return default_gens_xla[idx];
}

/**
 * Utility to create a XLAGeneratorImpl. Returns a shared_ptr
 */
absl::StatusOr<at::Generator> CreateXLAGenerator(
    c10::DeviceIndex device_index) {
  XLA_RETURN_IF_ERROR(InitGlobalVars(), "Failed to initialize XLA generators");
  // Normalize and validate the target device index; default to current device
  // when unspecified
  XLA_ASSIGN_OR_RETURN(c10::DeviceIndex idx,
                       NormalizeXLADeviceIndex(device_index),
                       "Invalid XLA device index");
  auto gen = at::make_generator<XLAGeneratorImpl>(idx);
  auto xla_gen = at::check_generator<XLAGeneratorImpl>(gen);
  xla_gen->set_current_seed(c10::default_rng_seed_val);
  return gen;
}

}  // namespace detail
}  // namespace at

namespace at {

XLAGeneratorImpl::XLAGeneratorImpl(DeviceIndex device_index)
    : c10::GeneratorImpl{Device(DeviceType::XLA, device_index),
                         DispatchKeySet(c10::DispatchKey::XLA)} {
  state_ = c10::make_intrusive<XLAGeneratorState>();
}

XLAGeneratorImpl::XLAGeneratorImpl(DeviceIndex device_index,
                                   c10::intrusive_ptr<XLAGeneratorState> state)
    : c10::GeneratorImpl{Device(DeviceType::XLA, device_index),
                         DispatchKeySet(c10::DispatchKey::XLA)},
      state_(std::move(state)) {}

DeviceType XLAGeneratorImpl::device_type() { return DeviceType::XLA; }

std::shared_ptr<XLAGeneratorImpl> XLAGeneratorImpl::clone() const {
  return std::shared_ptr<XLAGeneratorImpl>(clone_impl());
}

XLAGeneratorImpl* XLAGeneratorImpl::clone_impl() const {
  return new XLAGeneratorImpl(device_.index(), state_->clone());
}

void XLAGeneratorImpl::set_current_seed(uint64_t seed) { state_->seed_ = seed; }

uint64_t XLAGeneratorImpl::current_seed() const { return state_->seed_; }

uint64_t XLAGeneratorImpl::seed() {
  uint64_t random = c10::detail::getNonDeterministicRandom(true);
  set_current_seed(random);
  return random;
}

void XLAGeneratorImpl::set_offset(uint64_t offset) { state_->offset_ = offset; }

uint64_t XLAGeneratorImpl::get_offset() const { return state_->offset_; }

/* Serialize the generator state into a CPU tensor. */
c10::intrusive_ptr<c10::TensorImpl> XLAGeneratorImpl::get_state() const {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(uint64_t);
  static const size_t total_size = seed_size + offset_size;

  auto state_tensor =
      at::empty({(int64_t)total_size},
                at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  uint8_t* data_ptr = state_tensor.data_ptr<uint8_t>();
  memcpy(data_ptr, &state_->seed_, seed_size);
  memcpy(data_ptr + seed_size, &state_->offset_, offset_size);
  return state_tensor.getIntrusivePtr();
}

void XLAGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(uint64_t);
  static const size_t total_size = seed_size + offset_size;

  TORCH_CHECK(new_state.numel() == total_size,
              "The given state must be a byte tensor of size ", total_size,
              ", but was size ", new_state.numel());
  TORCH_CHECK(new_state.dtype() == at::kByte,
              "The given state must be a byte tensor, but was ",
              new_state.dtype());
  TORCH_CHECK(new_state.is_cpu(), "The given state must be a CPU tensor");

  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&state_->seed_, new_rng_state, seed_size);
  memcpy(&state_->offset_, new_rng_state + seed_size, offset_size);
}

}  // namespace at

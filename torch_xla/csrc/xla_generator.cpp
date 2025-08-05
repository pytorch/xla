#include "xla_generator.h"
#include <ATen/core/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Device.h>
#include <c10/core/TensorImpl.h>
#include <cstring>

namespace at {

XLAGeneratorImpl::XLAGeneratorImpl(DeviceIndex device_index)
    : c10::GeneratorImpl{Device(DeviceType::XLA, device_index), DispatchKeySet(c10::DispatchKey::XLA)} {
  state_ = c10::make_intrusive<XLAGeneratorState>();
}

XLAGeneratorImpl::XLAGeneratorImpl(DeviceIndex device_index, c10::intrusive_ptr<XLAGeneratorState> state)
    : c10::GeneratorImpl{Device(DeviceType::XLA, device_index), DispatchKeySet(c10::DispatchKey::XLA)}, state_(std::move(state)) {}

DeviceType XLAGeneratorImpl::device_type() {
  return DeviceType::XLA;
}

std::shared_ptr<XLAGeneratorImpl> XLAGeneratorImpl::clone() const {
  return std::shared_ptr<XLAGeneratorImpl>(clone_impl());
}

XLAGeneratorImpl* XLAGeneratorImpl::clone_impl() const {
  return new XLAGeneratorImpl(device_.index(), state_->clone());
}

void XLAGeneratorImpl::set_current_seed(uint64_t seed) {
  state_->seed_ = seed;
}

uint64_t XLAGeneratorImpl::current_seed() const {
  return state_->seed_;
}

uint64_t XLAGeneratorImpl::seed() {
  uint64_t random = c10::detail::getNonDeterministicRandom(true);
  set_current_seed(random);
  return random;
}

void XLAGeneratorImpl::set_offset(uint64_t offset) {
  state_->offset_ = offset;
}

uint64_t XLAGeneratorImpl::get_offset() const {
  return state_->offset_;
}

/* Serialize the generator state into a CPU tensor. */
c10::intrusive_ptr<c10::TensorImpl> XLAGeneratorImpl::get_state() const {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(uint64_t);
  static const size_t total_size = seed_size + offset_size;

  auto state_tensor = at::empty({(int64_t)total_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  uint8_t* data_ptr = state_tensor.data_ptr<uint8_t>();
  memcpy(data_ptr, &state_->seed_, seed_size);
  memcpy(data_ptr + seed_size, &state_->offset_, offset_size);
  return state_tensor.getIntrusivePtr();
}

void XLAGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(uint64_t);
  static const size_t total_size = seed_size + offset_size;

  TORCH_CHECK(new_state.numel() == total_size, "The given state must be a byte tensor of size ", total_size, ", but was size ", new_state.numel());
  TORCH_CHECK(new_state.dtype() == at::kByte, "The given state must be a byte tensor, but was ", new_state.dtype());
  TORCH_CHECK(new_state.is_cpu(), "The given state must be a CPU tensor");

  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&state_->seed_, new_rng_state, seed_size);
  memcpy(&state_->offset_, new_rng_state + seed_size, offset_size);
}

} // namespace at

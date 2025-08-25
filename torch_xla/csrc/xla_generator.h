#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>

namespace at {

// Holds the actual state variables for the XLA generator.
struct XLAGeneratorState : c10::intrusive_ptr_target {
  uint64_t seed_ = 0;
  uint64_t offset_ = 0;

  // Constructor
  XLAGeneratorState(uint64_t seed = 0, uint64_t offset = 0)
      : seed_(seed), offset_(offset) {}

  // Cloning method
  c10::intrusive_ptr<XLAGeneratorState> clone() {
    return c10::make_intrusive<XLAGeneratorState>(seed_, offset_);
  }
};

struct TORCH_API XLAGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  XLAGeneratorImpl(DeviceIndex device_index = -1);
  XLAGeneratorImpl(DeviceIndex device_index,
                   c10::intrusive_ptr<XLAGeneratorState> state);
  ~XLAGeneratorImpl() override = default;

  // Cloning support
  std::shared_ptr<XLAGeneratorImpl> clone() const;

  // --- Core Virtual Methods to Override ---
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  void set_state(const c10::TensorImpl& new_state) override;

  // --- Additional Methods ---
  static c10::DeviceType device_type();

 private:
  // Private clone implementation
  XLAGeneratorImpl* clone_impl() const override;

  // The actual state is held in a separate, cloneable object.
  c10::intrusive_ptr<XLAGeneratorState> state_;
};

}  // namespace at
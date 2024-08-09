#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <torch/csrc/Device.h>
#include <torch/extension.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/EmptyTensor.h>

#include <iostream>

// This file contains the heavy lifting to add a new C++ backend
// and integrate it directly into the PyTorch backend. It mainly involves:
//
// (1) Writing a custom allocator and registering it to pytorch
//     (see DummyCustomAllocator)
// (2) Writing a custom device guard, registering it to pytorch,
//     and using the device guard in kernels
//     (see DummyDeviceGuard)
// (3) Writing a custom aten::empty.memory_format function


// basic dummy add function
// at::Tensor custom_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
//   const at::OptionalDeviceGuard device_guard(at::device_of(self));
//   std::cout << "Custom aten::add.Tensor() called!" << std::endl;
//   // Since this custom device is just for testing, not bothering to implement kernels.
//   return at::empty(self.sizes(), self.options());
// }

// =====================================
// ========= Custom Allocators =========
// =====================================

// PyTorch provides an API for registering custom allocators for your device.
// You can create one by inheriting from the at::Allocator class,
// and registering your allocator for the particular device type
// (PrivateUse1 for open registration devices)

// A dummy allocator for our custom device, that secretly uses the CPU
// struct DummyCustomAllocator final : at::Allocator {
//   DummyCustomAllocator() = default;
//   at::DataPtr allocate(size_t nbytes) const override {
//     std::cout << "Custom allocator's allocate() called!" << std::endl;
//     void* data = c10::alloc_cpu(nbytes);
//     return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, 0)};
//   }

//   static void ReportAndDelete(void* ptr) {
//     if (!ptr) {
//       return;
//     }
//     std::cout << "Custom allocator's delete() called!" << std::endl;
//     c10::free_cpu(ptr);
//   }

//   at::DeleterFnPtr raw_deleter() const override {
//     return &ReportAndDelete;
//   }
// };

// // Register our dummy allocator
// static DummyCustomAllocator global_custom_alloc;
// REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

// =====================================
// ============= Device Guards =========
// =====================================

// PyTorch has an API for registering device guards.
// Device guards can be used to set the current "active" device,
// and e.g. error if the user provides an invalid device index.
//
// If your device doesn't support indices (e.g. foo:0 vs. foo:1),
// then the guards probably aren't needed.
//
// You can use it by creating a DeviceGuard class, registering it
// in PyTorch, and invoking the device guard before any kernels are called.
// For a more full-featured example of a device guard,
// check out the code at c10/cuda/CUDAGuard.h

// Represents the current "active" device.
// The dummy device guard registered below is meant to show how a backend
// can integrate custom device guard with pytorch.
// For something like cuda this represents the current active cuda device,
// which is directly set using the cuda API calls cudaGetDevice/cudaSetDevice.
// static uint16_t CURR_DEVICE = -1;

// Create and register a dummy device guard.
// struct DummyDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
//   static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;
//   DummyDeviceGuardImpl() {}
//   explicit DummyDeviceGuardImpl(c10::DeviceType t) {
//     TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
//   }
//   at::DeviceType type() const override {
//     return at::DeviceType::PrivateUse1;
//   }
//   at::Device exchangeDevice(at::Device d) const override {
//     TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
//     TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
//     at::Device old_device = getDevice();
//     if (old_device.index() != d.index()) {
//       // "set the active device"
//       CURR_DEVICE = d.index();
//     }
//     return old_device;
//   }
//   at::Device getDevice() const override {
//     return at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE);
//   }
//   void setDevice(at::Device d) const override {
//     TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1);
//     TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");
//     at::Device current_device = getDevice();
//     if (current_device != d) {
//       CURR_DEVICE = d.index();
//     }
//   }
//   void uncheckedSetDevice(at::Device d) const noexcept override {
//     auto current_device = getDevice();
//     if (current_device != d) {
//       CURR_DEVICE = d.index();
//     }
//   }
//   at::Stream getStream(at::Device d) const noexcept override {
//     // no-op
//     return at::Stream(at::Stream::DEFAULT, d);
//   }
//   // NB: These do NOT set the current device
//   at::Stream exchangeStream(at::Stream) const noexcept override {
//     // no-op
//     return at::Stream(at::Stream::DEFAULT, at::Device(at::DeviceType::PrivateUse1, CURR_DEVICE));
//   }
//   at::DeviceIndex deviceCount() const noexcept override {
//     // Hardcoding the number of "valid" devices here at 2.
//     return 2;
//   }

//   // Event-related functions
//   void record(
//       void** /*event*/,
//       const at::Stream& /*stream*/,
//       const at::DeviceIndex /*device_index*/,
//       const c10::EventFlag /*flag*/) const override {
//     TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.");
//   }
//   void block(void* /*event*/, const at::Stream& /*stream*/) const override {
//     TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
//   }
//   bool queryEvent(void* /*event*/) const override {
//     TORCH_CHECK(false, at::DeviceType::PrivateUse1, " backend doesn't support events.")
//   }
//   void destroyEvent(void* /*event*/, const at::DeviceIndex /*device_index*/)
//       const noexcept override {}

//   // Stream-related functions
//   bool queryStream(const at::Stream& /*stream*/) const override {
//     return true;
//   }
//   void synchronizeStream(const at::Stream& /*stream*/) const override {
//     // Don't wait for anything.
//   }
// };

// struct DummyGuard {
//   explicit DummyGuard() = delete;
//   explicit DummyGuard(at::DeviceIndex device_index) : guard_(device_index) {}
//   explicit DummyGuard(at::Device device) : guard_(device) {}
//   DummyGuard(const DummyGuard&) = delete;
//   DummyGuard& operator=(const DummyGuard&) = delete;
//   DummyGuard(DummyGuard&& other) = delete;
//   DummyGuard& operator=(DummyGuard&& other) = delete;

//   void set_device(at::Device device) {
//     guard_.set_device(device);
//   }

//   void reset_device(at::Device device) {
//     guard_.reset_device(device);
//   }

//   void set_index(at::DeviceIndex device_index) {
//     guard_.set_index(device_index);
//   }

//   at::Device original_device() const {
//     return guard_.original_device();
//   }

//   at::Device current_device() const {
//     return guard_.current_device();
//   }

//  private:
//   c10::impl::InlineDeviceGuard<DummyDeviceGuardImpl> guard_;
// };

// C10_REGISTER_GUARD_IMPL(PrivateUse1, DummyDeviceGuardImpl);


// =====================================
// ============= KERNELS ===============
// =====================================

// basic dummy empty function, so we can directly construct tensors on the custom device
// This dummy test device will just use the CPU allocator, and ignores pinned memory.
//
// Note: this kernel is very simple because our "custom device" just uses the normal TensorImpl object
// to store data under the hood.
// In PyTorch core today, both cpu and cuda are implemented with an ordinary TensorImpl class.
// Sometimes, backends prefer to subclass TensorImpl in order to store extra information.
// If this is the case, then this kernel is where you'll be responsible for creating and returning
// a fresh at::Tensor object, that properly stores a TensorImpl of your subclass.
// at::Tensor custom_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
//   const at::OptionalDeviceGuard device_guard(device);
//   std::cout << "Custom aten::empty.memory_format() called!" << std::endl;
//   constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
//   return at::detail::empty_generic(size, &global_custom_alloc, private_use_ks, c10::dtype_or_default(dtype), memory_format);
// }

// at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
//   const at::OptionalDeviceGuard device_guard(at::device_of(self));
//   // Not bothering to implement.
//   // Should fill the tensor's data with "value".
//   return self;
// }

// // basic dummy copy_() function, so we can copy from the custom device to/from CPU
// at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
//   const at::OptionalDeviceGuard device_guard(at::device_of(self));
//   std::cout << "Custom aten::_copy_from() called!" << std::endl;
//   TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
//   TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");

//   // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
//   TORCH_CHECK(self.sizes() == dst.sizes());
//   TORCH_CHECK(self.scalar_type() == dst.scalar_type());
//   TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

//   std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
//   return dst;
// }


// This macro does the heavy lifting.
// With TORCH_LIBRARY_IMPL, you can register custom kernels for your backend.
// For open registration, we're registering all of our kernels to the PrivateUse1 dispatch key.
// Later in this file, we map a custom device to the PrivateUse1 device type,
// which allows user code that puts a tensor on your custom_device to eventually get plumbed
// into the kernels registered here.
//
// This macro registers your kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
// TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
//   m.impl("add.Tensor", &custom_add_Tensor);
//   m.impl("empty.memory_format", &custom_empty_memory_format);
//   m.impl("fill_.Scalar", &custom_fill__scalar);
//   m.impl("_copy_from", &custom__copy_from);
// }

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device(int idx) {
  return c10::Device(c10::DeviceType::PrivateUse1, idx);
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_device", &get_custom_device, "get custom device object");
}

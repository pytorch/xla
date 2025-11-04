#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdlib>

#include "test/cpp/torch_xla_test.h"
#include "torch_xla/csrc/xla_generator.h"

namespace torch_xla {
namespace cpp_test {

// Ensure PJRT is configured to a CPU backend for tests that touch the PJRT
// runtime. Optionally allow overriding the environment values by passing
// `pjrt_device` and/or `cpu_num_devices`.
static void EnsurePjrtCpuBackend(const char* pjrt_device = nullptr,
                                 const char* cpu_num_devices = nullptr) {
  // PJRT_DEVICE: override if provided, otherwise set default if not present
  if (pjrt_device != nullptr && pjrt_device[0] != '\0') {
    // Force override of any existing value
    setenv("PJRT_DEVICE", pjrt_device, 1);
  } else {
    const char* pjrt = std::getenv("PJRT_DEVICE");
    if (pjrt == nullptr || pjrt[0] == '\0') {
      // Use CPU backend with a single device by default.
      setenv("PJRT_DEVICE", "CPU", 1);
    }
  }

  // CPU_NUM_DEVICES: override if provided, otherwise set default if not present
  if (cpu_num_devices != nullptr && cpu_num_devices[0] != '\0') {
    // Force override of any existing value
    setenv("CPU_NUM_DEVICES", cpu_num_devices, 1);
  } else {
    const char* cpu_devices = std::getenv("CPU_NUM_DEVICES");
    if (cpu_devices == nullptr || cpu_devices[0] == '\0') {
      // Default to a single CPU device. Preserve existing behavior of not
      // overwriting if already present (use overwrite=0 to match previous
      // semantics).
      setenv("CPU_NUM_DEVICES", "1", 0);
    }
  }
}

// Test fixture for XLAGenerator tests
class XLAGeneratorTest : public ::torch_xla::cpp_test::TorchXlaTest {
 protected:
  // Runs once before the test suite / test case to ensure PJRT is configured
  // before any XLA runtime initialization happens in per-test SetUp().
  static void SetUpTestCase() { EnsurePjrtCpuBackend("CPU", "2"); }

  void SetUp() {
    // Create a generator for XLA device 0
    gen_ = at::make_generator<at::XLAGeneratorImpl>(0);
  }

  at::Generator gen_;
};

TEST_F(XLAGeneratorTest, Constructor) {
  // Check that the generator was created for the correct device
  ASSERT_EQ(gen_.device().type(), at::DeviceType::XLA);
  ASSERT_EQ(gen_.device().index(), 0);

  // Check that the initial seed is 0
  ASSERT_EQ(gen_.current_seed(), 0);
}

TEST_F(XLAGeneratorTest, Seed) {
  // Test setting and getting the current seed
  uint64_t seed_val = 12345;
  gen_.set_current_seed(seed_val);
  ASSERT_EQ(gen_.current_seed(), seed_val);

  // Test the seed() method, which should set a non-deterministic seed
  uint64_t old_seed = gen_.current_seed();
  uint64_t new_seed = gen_.seed();
  // The new seed should be different from the old one and set as the current
  // seed
  ASSERT_NE(new_seed, old_seed);
  ASSERT_EQ(gen_.current_seed(), new_seed);
}

TEST_F(XLAGeneratorTest, GetAndSetState) {
  uint64_t seed_val = 98765;
  uint64_t offset_val = 0;

  // Set seed and offset on the original generator
  gen_.set_current_seed(seed_val);
  gen_.set_offset(offset_val);

  // Get the state from the original generator
  at::Tensor state_tensor = gen_.get_state();

  // Create a new generator
  auto new_gen = at::make_generator<at::XLAGeneratorImpl>(1);
  ASSERT_NE(new_gen.current_seed(), seed_val);

  // Set the state of the new generator
  new_gen.set_state(state_tensor);

  // Verify the state of the new generator
  ASSERT_EQ(new_gen.current_seed(), seed_val);
  ASSERT_EQ(new_gen.get_offset(), offset_val);
}

TEST_F(XLAGeneratorTest, SetStateValidation) {
  // Test that set_state throws with incorrect tensor properties
  auto new_gen = at::make_generator<at::XLAGeneratorImpl>(0);

  // Incorrect size
  auto wrong_size_tensor = at::empty({10}, at::kByte);
  EXPECT_THROW(new_gen.set_state(wrong_size_tensor), c10::Error);

  // Incorrect dtype
  auto wrong_dtype_tensor = at::empty({16}, at::kInt);
  EXPECT_THROW(new_gen.set_state(wrong_dtype_tensor), c10::Error);
}

TEST_F(XLAGeneratorTest, Clone) {
  uint64_t seed_val = 1;
  uint64_t offset_val = 0;

  // Set state on the original generator
  gen_.set_current_seed(seed_val);
  gen_.set_offset(offset_val);

  // Clone the generator
  auto cloned_gen = gen_.clone();

  // Verify that the cloned generator has the same state but is a different
  // object
  ASSERT_NE(std::addressof(cloned_gen), std::addressof(gen_));
  ASSERT_EQ(cloned_gen.device(), gen_.device());
  ASSERT_EQ(cloned_gen.current_seed(), gen_.current_seed());
  ASSERT_EQ(cloned_gen.get_offset(), offset_val);

  // Modify the original generator's seed and check that the clone is unaffected
  gen_.set_current_seed(9999);
  ASSERT_EQ(cloned_gen.current_seed(), seed_val);
  ASSERT_NE(cloned_gen.current_seed(), gen_.current_seed());
}

TEST_F(XLAGeneratorTest, GetDefaultXLAGenerator) {
  EnsurePjrtCpuBackend();
  // Test getting default generator for device 0
  auto result = at::detail::GetDefaultXLAGenerator(0);
  ASSERT_TRUE(result.ok()) << "Failed to get default generator: "
                           << result.status();

  const at::Generator& default_gen = result.value();
  ASSERT_EQ(default_gen.device().type(), at::DeviceType::XLA);
  ASSERT_EQ(default_gen.device().index(), 0);

  // Test getting default generator with -1 (should default to device 0)
  auto result_default = at::detail::GetDefaultXLAGenerator(-1);
  ASSERT_TRUE(result_default.ok())
      << "Failed to get default generator with -1: " << result_default.status();

  const at::Generator& default_gen_neg1 = result_default.value();
  ASSERT_EQ(default_gen_neg1.device().type(), at::DeviceType::XLA);
  ASSERT_EQ(default_gen_neg1.device().index(), 0);

  // Test that subsequent calls return the same generator instance
  auto result2 = at::detail::GetDefaultXLAGenerator(0);
  ASSERT_TRUE(result2.ok());
  const at::Generator& default_gen2 = result2.value();
  ASSERT_EQ(default_gen, default_gen2);

  // Test getting non-defuault device generator
  auto result_device1 = at::detail::GetDefaultXLAGenerator(1);
  ASSERT_TRUE(result_device1.ok())
      << "Failed to get default generator for device 1: "
      << result_device1.status();

  const at::Generator& default_gen_device1 = result_device1.value();
  ASSERT_EQ(default_gen_device1.device().type(), at::DeviceType::XLA);
  ASSERT_EQ(default_gen_device1.device().index(), 1);
  ASSERT_NE(default_gen_device1, default_gen);
}

TEST_F(XLAGeneratorTest, GetDefaultXLAGeneratorInvalidDevice) {
  EnsurePjrtCpuBackend();
  // Test with invalid device indices
  auto result_neg2 = at::detail::GetDefaultXLAGenerator(-2);
  ASSERT_FALSE(result_neg2.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(result_neg2.status()));
  ASSERT_THAT(result_neg2.status().message(),
              testing::HasSubstr("Invalid XLA device index"));

  // Test with very large device index (assuming there aren't 1000 XLA devices)
  auto result_large = at::detail::GetDefaultXLAGenerator(100);
  ASSERT_FALSE(result_large.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(result_large.status()));
  ASSERT_THAT(result_large.status().message(),
              testing::HasSubstr("Invalid XLA device index"));
}

TEST_F(XLAGeneratorTest, CreateXLAGenerator) {
  EnsurePjrtCpuBackend("CPU", "2");
  // Test creating generator for device 0
  auto result = at::detail::CreateXLAGenerator(1);
  ASSERT_TRUE(result.ok()) << "Failed to create generator: " << result.status();

  at::Generator created_gen = result.value();
  ASSERT_EQ(created_gen.device().type(), at::DeviceType::XLA);
  ASSERT_EQ(created_gen.device().index(), 1);

  // Test that the generator is initialized with default seed
  ASSERT_EQ(created_gen.current_seed(), c10::default_rng_seed_val);

  // Test creating generator with -1 (should use current device)
  auto result_current = at::detail::CreateXLAGenerator(-1);
  ASSERT_TRUE(result_current.ok())
      << "Failed to create generator with -1: " << result_current.status();

  at::Generator created_gen_neg1 = result_current.value();
  ASSERT_EQ(created_gen_neg1.device().type(), at::DeviceType::XLA);
  // Device index should be >= 0 (actual device depends on current XLA device)
  ASSERT_GE(created_gen_neg1.device().index(), 0);
}

TEST_F(XLAGeneratorTest, CreateXLAGeneratorUniqueness) {
  EnsurePjrtCpuBackend();
  // Test that each call creates a new generator instance
  auto result1 = at::detail::CreateXLAGenerator(0);
  auto result2 = at::detail::CreateXLAGenerator(0);

  ASSERT_TRUE(result1.ok());
  ASSERT_TRUE(result2.ok());

  at::Generator gen1 = result1.value();
  at::Generator gen2 = result2.value();

  // Should be different instances (compare generators, not their stack
  // addresses)
  ASSERT_NE(gen1, gen2);

  // But should have same device and initial seed
  ASSERT_EQ(gen1.device(), gen2.device());
  ASSERT_EQ(gen1.current_seed(), gen2.current_seed());

  // Modifying one should not affect the other
  gen1.set_current_seed(12345);
  ASSERT_NE(gen1.current_seed(), gen2.current_seed());
}

TEST_F(XLAGeneratorTest, CreateXLAGeneratorInvalidDevice) {
  EnsurePjrtCpuBackend();
  // Test with invalid device indices
  auto result_neg2 = at::detail::CreateXLAGenerator(-2);
  ASSERT_FALSE(result_neg2.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(result_neg2.status()));
  ASSERT_THAT(result_neg2.status().message(),
              testing::HasSubstr("Invalid XLA device index"));

  // Test with very large device index (assuming there aren't 100 XLA devices)
  auto result_large = at::detail::CreateXLAGenerator(100);
  ASSERT_FALSE(result_large.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(result_large.status()));
  ASSERT_THAT(result_large.status().message(),
              testing::HasSubstr("Invalid XLA device index"));
}

}  // namespace cpp_test
}  // namespace torch_xla

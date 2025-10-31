#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdlib>

#include "test/cpp/torch_xla_test.h"
#include "torch_xla/csrc/xla_generator.h"

namespace torch_xla {
namespace cpp_test {

// Test fixture for XLAGenerator tests
class XLAGeneratorTest : public ::torch_xla::cpp_test::TorchXlaTest {
 protected:
  void SetUp() {
    // Create a generator for XLA device 0
    gen_ = at::make_generator<at::XLAGeneratorImpl>(0);
  }

  at::Generator gen_;
};

// Ensure PJRT is configured to a CPU backend for tests that touch the PJRT
// runtime.
static void EnsurePjrtCpuBackend() {
  const char* pjrt = std::getenv("PJRT_DEVICE");
  if (pjrt == nullptr || pjrt[0] == '\0') {
    // Use CPU backend with a single device by default.
    setenv("PJRT_DEVICE", "CPU", 1);
  }
  const char* cpu_devices = std::getenv("CPU_NUM_DEVICES");
  if (cpu_devices == nullptr || cpu_devices[0] == '\0') {
    setenv("CPU_NUM_DEVICES", "1", 0);
  }
}

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
  ASSERT_EQ(std::addressof(default_gen), std::addressof(default_gen2));
}

TEST_F(XLAGeneratorTest, GetDefaultXLAGeneratorInvalidDevice) {
  EnsurePjrtCpuBackend();
  // Test with invalid device indices
  auto result_neg2 = at::detail::GetDefaultXLAGenerator(-2);
  ASSERT_FALSE(result_neg2.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(result_neg2.status()));

  // Test with very large device index (assuming there aren't 1000 XLA devices)
  auto result_large = at::detail::GetDefaultXLAGenerator(1000);
  ASSERT_FALSE(result_large.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(result_large.status()));
}

TEST_F(XLAGeneratorTest, CreateXLAGenerator) {
  EnsurePjrtCpuBackend();
  // Test creating generator for device 0
  auto result = at::detail::CreateXLAGenerator(0);
  ASSERT_TRUE(result.ok()) << "Failed to create generator: " << result.status();

  at::Generator created_gen = result.value();
  ASSERT_EQ(created_gen.device().type(), at::DeviceType::XLA);
  ASSERT_EQ(created_gen.device().index(), 0);

  // Test that the generator is initialized with default seed
  ASSERT_EQ(created_gen.current_seed(), c10::default_rng_seed_val);

  // Test creating generator with -1 (should use current device)
  auto result_default = at::detail::CreateXLAGenerator(-1);
  ASSERT_TRUE(result_default.ok())
      << "Failed to create generator with -1: " << result_default.status();

  at::Generator created_gen_neg1 = result_default.value();
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

  // Should be different instances
  ASSERT_NE(std::addressof(gen1), std::addressof(gen2));

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

  // Test with very large device index (assuming there aren't 1000 XLA devices)
  auto result_large = at::detail::CreateXLAGenerator(1000);
  ASSERT_FALSE(result_large.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(result_large.status()));
}

}  // namespace cpp_test
}  // namespace torch_xla

#include <gtest/gtest.h>
#include <torch/torch.h>

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

}  // namespace cpp_test
}  // namespace torch_xla
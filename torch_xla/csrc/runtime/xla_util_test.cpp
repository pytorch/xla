#include "torch_xla/csrc/runtime/xla_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status_matchers.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla_util.h"

namespace torch_xla {
namespace runtime {
namespace util {

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(XlaUtilTest, ShapeHash) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2, 2});
  EXPECT_EQ(ShapeHash(shape), ShapeHash(shape));
}

template <typename MessageType>
absl::StatusOr<MessageType> ParseTextProto(const std::string& text_proto) {
  tsl::protobuf::TextFormat::Parser parser;
  MessageType parsed_proto;
  tsl::protobuf::io::ArrayInputStream input_stream(text_proto.data(),
                                                   text_proto.size());
  if (!parser.Parse(&input_stream, &parsed_proto)) {
    return tsl::errors::InvalidArgument("Could not parse text proto: ",
                                        text_proto);
  }
  return parsed_proto;
}

TEST(XlaUtilTest, CreateModule) {
  TF_ASSERT_OK_AND_ASSIGN(
      xla::HloModuleProto hlo_module_proto,
      ParseTextProto<xla::HloModuleProto>(
          R"pb(
            name: "myname"
            id: 7
            entry_computation_name: "mycomp"
            entry_computation_id: 0
            computations {
              id: 0
              name: "c1"
              instructions: {
                name: "i1"
                id: 1
                opcode: "constant"
                shape: {
                  element_type: S32
                  layout {}
                }
                literal: {
                  shape: {
                    element_type: S32
                    layout {}
                  }
                  s32s: 0
                }
              }
              instructions: {
                name: "constant.3"
                id: 0
                opcode: "constant"
                shape: {
                  element_type: S32
                  layout {}
                }
                literal: {
                  shape: {
                    element_type: S32
                    layout {}
                  }
                  s32s: 0
                }
              }
              root_id: 1
            }
            host_program_shape: { result: { element_type: 4 } }
          )pb"));

  xla::HloModule m("cool_module", {});
  auto got = CreateModuleFromProto(hlo_module_proto);
  EXPECT_THAT(got, IsOk());
  EXPECT_EQ((*got)->name(), "myname");
  EXPECT_EQ((*got)->computation_count(), 1);
}

TEST(XlaUtilTest, XlaToHlo) {
  xla::Shape input_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2, 2});
  xla::XlaBuilder builder("AddComputation");
  xla::XlaOp x = xla::Parameter(&builder, 0, input_shape, "x");
  xla::XlaOp y = xla::Parameter(&builder, 1, input_shape, "y");
  xla::XlaOp sum = xla::Add(x, y);
  ASSERT_THAT(GetComputationHloText(*builder.Build()),
              IsOkAndHolds(AllOf(
                  HasSubstr("HloModule AddComputation.4"),
                  HasSubstr("%AddComputation.4 (x.1: f32[2,2], y.2: f32[2,2])"),
                  HasSubstr("ROOT %add.3"))));
}

TEST(XlaUtilTest, TestDeterministicModuleProtoSerializationEmptyProto) {
  xla::HloModuleProto empty_proto;
  auto result =
      ::ConsumeValue(GetDeterministicSerializedModuleProto(empty_proto));
  // Verify that the result is an empty string
  EXPECT_TRUE(result.empty());
}

TEST(XlaUtilTest, TestDeterministicModuleProtoSerialization) {
  // Create a test HLO module with a known structure
  TF_ASSERT_OK_AND_ASSIGN(
      xla::HloModuleProto hlo_module_proto,
      ParseTextProto<xla::HloModuleProto>(
          R"pb(
            name: "myname"
            id: 9
            entry_computation_name: "MyCustomName.9"
            entry_computation_id: 9
            computations {
              id: 9
              name: "MyCustomName.9"
              instructions: {
                name: "p0.1"
                id: 1
                opcode: "parameter"
                shape: {
                  element_type: S64
                  layout { tail_padding_alignment_in_elements: 1 }
                }
                metadata {
                  op_type: "xla__device_data"
                  op_name: "xla__device_data"
                  source_file: "/ansible/pytorch/xla/small_test.py"
                  source_line: 14
                  stack_frame_id: 1
                }
              }
              instructions: {
                name: "p1.2"
                id: 2
                opcode: "parameter"
                parameter_number: 1
                shape: {
                  element_type: S64
                  layout { tail_padding_alignment_in_elements: 1 }
                }
                metadata {
                  op_type: "xla__device_data"
                  op_name: "xla__device_data"
                  source_file: "/ansible/pytorch/xla/small_test.py"
                  source_line: 13
                  stack_frame_id: 2
                }
              }
              instructions: {
                name: "call.7"
                id: 7
                opcode: "call"
                shape: {
                  element_type: S64
                  layout { tail_padding_alignment_in_elements: 1 }
                }
                metadata {
                  op_type: "xla___op_some_op"
                  op_name: "xla___op_some_op"
                  source_file: "/ansible/pytorch/xla/torch_xla/core/xla_op_registry.py"
                  source_line: 44
                  stack_frame_id: 4
                }
                called_computation_ids: 3
                operand_ids: 2
                operand_ids: 1
              }
              instructions: {
                name: "tuple.8"
                id: 8
                opcode: "tuple"
                shape: {
                  element_type: TUPLE
                  tuple_shapes {
                    element_type: S64
                    layout { tail_padding_alignment_in_elements: 1 }
                  }
                }
                operand_ids: 7
              }
              root_id: 8
            }
            host_program_shape: {
              parameters {
                element_type: S64
                layout { tail_padding_alignment_in_elements: 1 }
              }
              parameters {
                element_type: S64
                layout { tail_padding_alignment_in_elements: 1 }
              }
              result {
                element_type: TUPLE
                tuple_shapes {
                  element_type: S64
                  layout { tail_padding_alignment_in_elements: 1 }
                }
              }
              parameter_names: "p0"
              parameter_names: "p1"
            }
          )pb"));

  // Define a set of dummy fixed key-value pairs for frontend attributes.
  std::vector<std::pair<std::string, std::string>> attr_pairs = {
      {"key1", "value1"},
      {"key2", "value2"},
      {"key3", "value3"},
      {"key4", "value4"}};

  auto shuffle_and_hash = [&attr_pairs](xla::HloModuleProto hlo_module_proto) {
    // Create a random number generator for shuffling.
    std::random_device random_device;
    std::mt19937 random_generator(random_device());

    for (auto& computation : *hlo_module_proto.mutable_computations()) {
      for (auto& instruction : *computation.mutable_instructions()) {
        std::shuffle(attr_pairs.begin(), attr_pairs.end(), random_generator);
        auto* frontend_attrs = instruction.mutable_frontend_attributes();
        // Add the dummy shuffled pairs to the frontend attributes.
        for (const auto& pair : attr_pairs) {
          (*frontend_attrs->mutable_map())[pair.first] = pair.second;
        }
      }
    }
    std::string serialized_proto =
        ::ConsumeValue(GetDeterministicSerializedModuleProto(hlo_module_proto));
    return torch::lazy::Hash(serialized_proto);
  };

  // Compute hashes with different random orderings of attributes
  torch::lazy::hash_t hash1 = shuffle_and_hash(hlo_module_proto);
  torch::lazy::hash_t hash2 = shuffle_and_hash(hlo_module_proto);
  // Verify that different orderings produce the same hash
  ASSERT_EQ(hash1, hash2)
      << "Hashes should match regardless of the frontend attribute ordering";
}

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla

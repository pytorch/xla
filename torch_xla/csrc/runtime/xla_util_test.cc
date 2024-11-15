#include "torch_xla/csrc/runtime/xla_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <set>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status_matchers.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
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

TEST(XlaUtilrest, CreateModule) {
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

TEST(XlaUtilrest, XlaToHlo) {
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

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_CLIENT_VARIABLE_OPS_H_
#define XLA_CLIENT_VARIABLE_OPS_H_

#include "tsl/framework/allocator.h"
#include "third_party/xla_client/openxla_op_kernel.h"
#include "third_party/xla_client/openxla_register_types.h"
#include "third_party/xla_client/openxla_resource_mgr.h"
#include "third_party/xla_client/openxla_resource_var.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/types.h"

namespace xla {

class VariableOp : public OpKernel {
 public:
  explicit VariableOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* ctx) override;

 private:
  DataType dtype_;
  TensorShape shape_;
  ContainerInfo cinfo_;

  TF_DISALLOW_COPY_AND_ASSIGN(VariableOp);
};

}  // namespace xla

#endif  // XLA_CLIENT_VARIABLE_OPS_H_
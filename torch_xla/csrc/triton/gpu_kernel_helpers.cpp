/* Copyright 2019 The JAX Authors.

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

#include "torch_xla/csrc/triton/gpu_kernel_helpers.h"

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

namespace torch_xla {
namespace XLA_GPU_NAMESPACE {

namespace {
std::string ErrorString(gpuError_t error) { return gpuGetErrorString(error); }

std::string ErrorString(CUresult error) {
  const char* str;

  CUresult result = cuGetErrorName(error, &str);
  if (result == CUDA_SUCCESS) {
    return str;
  }
  return absl::StrFormat(
      "Unknown CUDA error %d; cuGetErrorName failed. This probably means that "
      "JAX was unable to load the CUDA libraries.",
      error);
}

std::string ErrorString(gpusparseStatus_t status) {
  return cusparseGetErrorString(status);
}

std::string ErrorString(gpusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:
      return "cuSolver success.";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "cuSolver has not been initialized";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "cuSolver allocation failed";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "cuSolver invalid value error";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "cuSolver architecture mismatch error";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return "cuSolver mapping error";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "cuSolver execution failed";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "cuSolver internal error";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "cuSolver matrix type not supported error";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "cuSolver not supported error";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "cuSolver zero pivot error";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "cuSolver invalid license error";
    default:
      return absl::StrCat("Unknown cuSolver error: ", status);
  }
}

std::string ErrorString(gpublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "cuBlas success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "cuBlas has not been initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "cuBlas allocation failure";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "cuBlas invalid value error";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "cuBlas architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "cuBlas mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "cuBlas execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "cuBlas internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "cuBlas not supported error";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "cuBlas license error";
    default:
      return "Unknown cuBlas error";
  }
}

std::string ErrorString(CUptiResult error) {
#if CUPTI_API_VERSION >= 20
  const char* str;
  CUptiResult result = cuptiGetErrorMessage(error, &str);
  if (result == CUPTI_SUCCESS) {
    return str;
  }
#endif  // CUPTI_API_VERSION >= 20
  return absl::StrFormat(
      "Unknown CUPTI error %d. This probably means that JAX was unable to load "
      "cupti.",
      error);
}

std::string ErrorString(cufftResult status) {
  switch (status) {
    case CUFFT_SUCCESS:
      return "cuFFT success";
    case CUFFT_INVALID_PLAN:
      return "cuFFT invalid plan";
    case CUFFT_ALLOC_FAILED:
      return "cuFFT allocation failed";
    case CUFFT_INVALID_TYPE:
      return "cuFFT invalid type";
    case CUFFT_INVALID_VALUE:
      return "cuFFT invalid value";
    case CUFFT_INTERNAL_ERROR:
      return "cuFFT internal error";
    case CUFFT_EXEC_FAILED:
      return "cuFFT execution failed";
    case CUFFT_SETUP_FAILED:
      return "cuFFT setup failed";
    case CUFFT_INVALID_SIZE:
      return "cuFFT invalid size";
    case CUFFT_UNALIGNED_DATA:
      return "cuFFT unaligned data";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "cuFFT incomplete parameter list";
    case CUFFT_INVALID_DEVICE:
      return "cuFFT invalid device";
    case CUFFT_PARSE_ERROR:
      return "cuFFT parse error";
    case CUFFT_NO_WORKSPACE:
      return "cuFFT no workspace";
    case CUFFT_NOT_IMPLEMENTED:
      return "cuFFT not implemented";
    case CUFFT_LICENSE_ERROR:
      return "cuFFT license error";
    case CUFFT_NOT_SUPPORTED:
      return "cuFFT not supported";
    default:
      return "Unknown cuFFT error";
  }
}



template <typename T>
std::string ErrorString(T status, const char* file, std::int64_t line,
                        const char* expr) {
  return absl::StrFormat("%s:%d: operation %s failed: %s", file, line, expr,
                         ErrorString(status));
}
}  // namespace

absl::Status AsStatus(gpuError_t error, const char* file, std::int64_t line,
                      const char* expr) {
  if (ABSL_PREDICT_FALSE(error != gpuSuccess))
    return absl::InternalError(ErrorString(error, file, line, expr));
  return absl::OkStatus();
}

absl::Status AsStatus(gpusolverStatus_t status, const char* file,
                      std::int64_t line, const char* expr) {
  if (ABSL_PREDICT_FALSE(status != GPUSOLVER_STATUS_SUCCESS))
    return absl::InternalError(ErrorString(status, file, line, expr));
  return absl::OkStatus();
}

absl::Status AsStatus(gpusparseStatus_t status, const char* file,
                      std::int64_t line, const char* expr) {
  if (ABSL_PREDICT_FALSE(status != GPUSPARSE_STATUS_SUCCESS))
    return absl::InternalError(ErrorString(status, file, line, expr));
  return absl::OkStatus();
}

absl::Status AsStatus(gpublasStatus_t status, const char* file,
                      std::int64_t line, const char* expr) {
  if (ABSL_PREDICT_FALSE(status != GPUBLAS_STATUS_SUCCESS))
    return absl::InternalError(ErrorString(status, file, line, expr));
  return absl::OkStatus();
}


absl::Status AsStatus(CUresult error, const char* file, std::int64_t line,
                      const char* expr) {
  if (ABSL_PREDICT_FALSE(error != CUDA_SUCCESS))
    return absl::InternalError(ErrorString(error, file, line, expr));
  return absl::OkStatus();
}

absl::Status AsStatus(CUptiResult error, const char* file, std::int64_t line,
                      const char* expr) {
  if (ABSL_PREDICT_FALSE(error != CUPTI_SUCCESS))
    return absl::InternalError(ErrorString(error, file, line, expr));
  return absl::OkStatus();
}

absl::Status AsStatus(cufftResult error, const char* file, std::int64_t line,
                      const char* expr) {
  if (ABSL_PREDICT_FALSE(error != CUFFT_SUCCESS))
    return absl::InternalError(ErrorString(error, file, line, expr));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<void*[]>> MakeBatchPointers(
    gpuStream_t stream, void* buffer, void* dev_ptrs, int batch,
    int batch_elem_size) {
  char* ptr = static_cast<char*>(buffer);
  auto host_ptrs = absl::make_unique<void*[]>(batch);
  for (int i = 0; i < batch; ++i) {
    host_ptrs[i] = ptr;
    ptr += batch_elem_size;
  }
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpuMemcpyAsync(dev_ptrs, host_ptrs.get(), sizeof(void*) * batch,
                     gpuMemcpyHostToDevice, stream)));
  return std::move(host_ptrs);
}

}  // namespace XLA_GPU_NAMESPACE
}  // namespace torch_xla
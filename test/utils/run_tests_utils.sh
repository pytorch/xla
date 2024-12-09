#!/bin/bash
set -exo pipefail

# Run a test with tensor saving enabled, using a specified graph format. The
# graph dump files are cleaned after the test. In case the test crashes, the
# file is retained.
#
# Usage: run_save_tensor <exec> <format> [test arguments...]
#
# Arguments:
#   exec: The executable or function to run the test (python3 or any function)
#   format: The graph format to use with XLA_SAVE_TENSORS_FMT
#   test arguments: Arguments to pass to the test
#
# Environment:
#   Sets XLA_SAVE_TENSORS_FILE and XLA_SAVE_TENSORS_FMT
function run_save_tensor {
  local run_test_func="$1" ; local file_graph_format="$2" ; shift 2

  echo "Running in save tensor file mode: $@"
  local base_file="/tmp/xla_test_save_ir.txt"

  # Check if the file already exists, for any device ordinal number.
  if ls "${base_file}"* 1> /dev/null 2>&1; then
    echo "Error: File ${base_file} or a numbered version already exists. Please remove it before running the test."
    return 1
  fi

  XLA_SAVE_TENSORS_FILE="$base_file" XLA_SAVE_TENSORS_FMT="$file_graph_format" $run_test_func "$@"
  local test_status=$?

  # Clean up the file once the test finalizes.
  local actual_file
  actual_file=$(ls "${base_file}"* 2>/dev/null | head -n1)
  if [ -f "$actual_file" ]; then
    echo "Cleaning up temporary file: $actual_file"
    rm "$actual_file"
  else
    echo "Warning: Expected output file not found"
  fi
  return $test_status
}

function run_save_tensor_ir {
  local run_test_func="$1"
  shift
  echo "Running in save tensor file mode: $@"
  run_save_tensor "$run_test_func" "text" "$@"
}

function run_save_tensor_hlo {
  local run_test_func="$1"
  shift
  echo "Running in save tensor file mode: $@"
  run_save_tensor "$run_test_func" "hlo" "$@"
}

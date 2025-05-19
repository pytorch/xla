#!/bin/bash
set -exo pipefail

# Parses the commandline flags and sets correponding environment variables.
function parse_options_to_vars {
  # Default option values. Can be overridden via commandline flags.
  LOGFILE=/tmp/pytorch_py_test.log
  MAX_GRAPH_SIZE=500
  GRAPH_CHECK_FREQUENCY=100
  VERBOSITY=2

  # Parse commandline flags:
  #   -L
  #      disable writing to the log file at $LOGFILE.
  #   -M max_graph_size
  #   -C graph_check_frequency
  #   -V verbosity
  #   -h
  #      print the help string
  while getopts 'LM:C:V:h' OPTION
  do
    case $OPTION in
      L)
        LOGFILE=
        ;;
      M)
        MAX_GRAPH_SIZE=$OPTARG
        ;;
      C)
        GRAPH_CHECK_FREQUENCY=$OPTARG
        ;;
      V)
        VERBOSITY=$OPTARG
        ;;
      h)
        echo -e "Usage: $0 TEST_FILTER...\nwhere TEST_FILTERs are globs match .py test files. If no test filter is provided, runs all tests."
        exit 0
        ;;
      \?)  # This catches all invalid options.
        echo "ERROR: Invalid commandline flag."
        exit 1
    esac
  done
}

# Given $1 as a (possibly not normalized) test filepath, returns successfully
# if it matches any of the space-separated globs $_TEST_FILTER. If
# $_TEST_FILTER is empty, returns successfully.
function test_is_selected {
  if [[ -z "$_TEST_FILTER" ]]; then
    return 0 # success
  fi

  # _TEST_FILTER is a space-separate list of globs. Loop through the
  # list elements.
  for _FILTER in $_TEST_FILTER; do
    # realpath normalizes the paths (e.g. resolving `..` and relative paths)
    # so that they can be compared.
    case $(realpath $1) in
    $(realpath $_FILTER))
      return 0 # success
      ;;
    *)
      # No match
      ;;
    esac
  done

  return 1 # failure
}

# Sets $_TEST_FILTER based on the commandline arguments.
function set_test_filter {
  if [[ $# -ge 1 ]]; then
    # There are positional arguments - set $_TEST_FILTER to them.
    _TEST_FILTER=$@
    # Sometimes a test may fail even if it doesn't match _TEST_FILTER. Therefore,
    # we need to set this to be able to get to the test(s) we want to run.
    CONTINUE_ON_ERROR=1
  else
    # No positional argument - run all tests.
    _TEST_FILTER=""
  fi
}

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

function run_xla_ir_debug {
  local run_test_func="$1"
  shift
  echo "Running with XLA_IR_DEBUG: $@"
  XLA_IR_DEBUG=1 "$run_test_func" "$@"
}

function run_xla_hlo_debug {
  local run_test_func="$1"
  shift
  echo "Running with XLA_HLO_DEBUG: $@"
  XLA_HLO_DEBUG=1 "$run_test_func" "$@"
}

function run_xla_ir_hlo_debug {
  local run_test_func="$1"
  shift
  echo "Running with XLA_IR_DEBUG and XLA_HLO_DEBUG: $@"
  XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 "$run_test_func" "$@"
}

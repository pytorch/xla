#!/bin/bash
set -ex
RUNDIR="$(cd "$(dirname "$0")" ; pwd -P)"
BUILDTYPE="opt"
VERB=
FILTER=
LOGFILE=/tmp/pytorch_cpp_test.log
XLA_EXPERIMENTAL="nonzero:masked_select"
BAZEL_REMOTE_CACHE="0"
BAZEL_VERB="test"

# See Note [Keep Going]
CONTINUE_ON_ERROR=false
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  set +e
fi

if [ "$DEBUG" == "1" ]; then
  BUILDTYPE="dbg"
fi

while getopts 'VLDKBF:X:RC' OPTION
do
  case $OPTION in
    V)
      VERB="VERBOSE=1"
      ;;
    L)
      LOGFILE=
      ;;
    D)
      BUILDTYPE="dbg"
      ;;
    F)
      FILTER="--test_filter=$OPTARG"
      ;;
    X)
      XLA_EXPERIMENTAL="$OPTARG"
      ;;
    R)
      BAZEL_REMOTE_CACHE="1"
      ;;
    C)
      BAZEL_VERB="coverage"
      ;;
  esac
done
shift $(($OPTIND - 1))

# Set XLA_EXPERIMENTAL var to subsequently executed commands.
export XLA_EXPERIMENTAL

EXTRA_FLAGS=""

if [[ "$TPUVM_MODE" == "1" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --config=tpu"
fi

# Handle remote builds and remote cache. Use a CI-private cache silo to avoid cachce polution.
if [[ "$BAZEL_REMOTE_CACHE" == "1" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --config=remote_cache"
  if [[ ! -z "$GCLOUD_SERVICE_KEY_FILE" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --google_credentials=$GCLOUD_SERVICE_KEY_FILE"
  fi
  if [[ ! -z "$SILO_NAME" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --remote_default_exec_properties=cache-silo-key=$SILO_NAME"
  fi
fi
if [[ "$XLA_CUDA" == "1" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --config=cuda"
fi
if [[ "$BAZEL_VERB" == "coverage" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --remote_download_outputs=all" # for lcov symlink
fi


if [ "$LOGFILE" != "" ]; then
  bazel $BAZEL_VERB $EXTRA_FLAGS //third_party/xla_client:all //test/cpp:all ${FILTER:+"$FILTER"} 2> $LOGFILE
else
  bazel $BAZEL_VERB $EXTRA_FLAGS //third_party/xla_client:all //test/cpp:all ${FILTER:+"$FILTER"}
fi

#!/bin/bash
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
export XRT_WORKERS="localservice:0;grpc://localhost:40934"
export XLA_EXPERIMENTAL="nonzero:masked_select"

BAZEL_REMOTE_CACHE_CONFIG="--config=remote_cache --remote_default_exec_properties=cache-silo-key=cache-silo-coverage"
if [ ! -z "$GCLOUD_SERVICE_KEY_FILE" ]; then
  file_size=$(stat -c%s "$GCLOUD_SERVICE_KEY_FILE")
  if [ "$file_size" -le 1 ]; then
    BAZEL_REMOTE_CACHE_CONFIG=""
  fi
fi

bazel coverage $BAZEL_REMOTE_CACHE_CONFIG //...
cp "$(bazel info output_path)/_coverage/_coverage_report.dat" /tmp/cov_xrt.dat

export PJRT_DEVICE="CPU"
bazel coverage $BAZEL_REMOTE_CACHE_CONFIG //test/...
cp "$(bazel info output_path)/_coverage/_coverage_report.dat" /tmp/cov_pjrt.dat

# requires `apt-get install lcov`
lcov --add-tracefile /tmp/cov_xrt.dat -a /tmp/cov_pjrt.dat -o /tmp/merged.dat
genhtml /tmp/merged.dat -o CodeCoveragn m
BASEDIR="$(dirname $(dirname $(realpath $0)))"
export PYTHONPATH="$BASEDIR"

function run_test {
    pushd "$BASEDIR"
    python3 "$@"
    popd
}

if [[ "$RUN_XLA_OP_TESTS1" == "xla_op1" ]]; then
    run_test test/test_result_analyzer.py
fi

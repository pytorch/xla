BUILD_CPP_TESTS=0 python setup.py install
cd test
#TF_CPP_VMODULE=tensor=5,computation_client=5,xrt_computation_client=5,aten_xla_type=1 XLA_SAVE_TENSORS_FILE=/tmp/graphs.hlo XLA_SAVE_TENSORS_FMT=hlo TPU_STDERR_LOG_LEVEL=0 XLA_USE_EAGER_DEBUG_MODE=1 python3 test_operations.py TestAtenXlaTensor.test_index_put  --verbosity=2
#XLA_USE_EAGER_DEBUG_MODE=1 python3 test_operations.py TestAtenXlaTensor.test_index_put  --verbosity=2
#XLA_USE_EAGER_DEBUG_MODE=1 python3 test_operations.py TestAtenXlaTensor.test_inplace_view_makes_base_require_grad  --verbosity=2
XLA_USE_EAGER_DEBUG_MODE=1 python3 test_operations.py --verbosity=2
cd -

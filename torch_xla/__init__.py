import os
GRPC_OPTIONS = [
    'grpc.keepalive_time_ms=60000',  # 1 min
    'grpc.keepalive_timeout_ms=14400000',  # 4 hrs
    'grpc.http2.max_pings_without_data=0',  # unlimited
    'grpc.http2.min_ping_interval_without_data_ms=300000',  # 5 min
]
os.environ['TF_GRPC_DEFAULT_OPTIONS'] = ','.join(GRPC_OPTIONS)

import torch
import _XLAC

_XLAC._initialize_aten_bindings()

import torch
import torch_xla
import os

os.environ['TPU_SKIP_MDS_QUERY'] = 'True'
os.environ['TPU_ACCELERATOR_TYPE'] = 'tpu7x-8'
os.environ['TPU_PROCESS_BOUNDS'] = '2,2,1'
os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '1,1,1'
os.environ['TPU_WORKER_ID'] = '0'
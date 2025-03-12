# Running SPMD on GPU

PyTorch/XLA supports SPMD on NVIDIA GPU (single-node or multi-nodes).
The training/inference script remains the same as the one used for TPU,
such as this [ResNet
script](https://github.com/pytorch/xla/blob/1dc78948c0c9d018d8d0d2b4cce912552ab27083/test/spmd/test_train_spmd_imagenet.py).
To execute the script using SPMD, we leverage `torchrun`:

    PJRT_DEVICE=CUDA \
    torchrun \
    --nnodes=${NUM_GPU_MACHINES} \
    --node_rank=${RANK_OF_CURRENT_MACHINE} \
    --nproc_per_node=1 \
    --rdzv_endpoint="<MACHINE_0_IP_ADDRESS>:<PORT>" \
    training_or_inference_script_using_spmd.py

-   `--nnodes`: how many GPU machines to be used.
-   `--node_rank`: the index of the current GPU machines. The value can
    be 0, 1, ..., \${NUMBER_GPU_VM}-1.
-   `--nproc_per_node`: the value must be 1 due to the SPMD requirement.
-   `--rdzv_endpoint`: the endpoint of the GPU machine with
    node_rank==0, in the form `host:port`. The host will be the internal
    IP address. The `port` can be any available port on the machine. For
    single-node training/inference, this parameter can be omitted.

For example, if you want to train a ResNet model on 2 GPU machines using
SPMD, you can run the script below on the first machine:

    XLA_USE_SPMD=1 PJRT_DEVICE=CUDA \
    torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --rdzv_endpoint="<MACHINE_0_INTERNAL_IP_ADDRESS>:12355" \
    pytorch/xla/test/spmd/test_train_spmd_imagenet.py --fake_data --batch_size 128

and run the following on the second machine:

    XLA_USE_SPMD=1 PJRT_DEVICE=CUDA \
    torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=1 \
    --rdzv_endpoint="<MACHINE_0_INTERNAL_IP_ADDRESS>:12355" \
    pytorch/xla/test/spmd/test_train_spmd_imagenet.py --fake_data --batch_size 128

For more information, please refer to the [SPMD support on GPU
RFC](https://github.com/pytorch/xla/issues/6256).

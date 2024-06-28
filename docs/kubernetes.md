# Distributed training on GKE

PyTorch/XLA supports distributed training on GKE via [indexed
`Job`s](https://kubernetes.io/docs/tasks/job/job-with-pod-to-pod-communication/)
and `torchrun`. For more information about creating a GKE cluster with
accelerators, see the documentation for
[TPUs](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus) and
[GPUs](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus),
respectively.

## GPU Example

GKE is the recommended platform for distributed training with GPUs. This example
uses two hosts, each with two NVidia v100 GPUs. Adjust the values in according
to the comments for a larger or smaller cluster.

Create a new file `gpu_test.yaml` with the following:

```yaml
# Headless service used for service discovery.
# See https://kubernetes.io/docs/concepts/services-networking/service/#headless-services
apiVersion: v1
kind: Service
metadata:
  name: headless-svc
spec:
  selector:
    headless-svc: 'true'
  clusterIP: None
---
apiVersion: batch/v1
kind: Job
metadata:
  generateName: torch-xla-resnet50-v100-x2x2-
spec:
  # Don't retry upon failure
  backoffLimit: 0
  # Indexed jobs pass rank to each replica
  completionMode: Indexed
  # Set `completions` and `parallelism` to the number of hosts in the cluster
  completions: 2
  parallelism: 2
  template:
    metadata:
      creationTimestamp: null
      labels:
        headless-svc: "true"
    spec:
      containers:
      - name: main
        image: us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.3.0_3.10_cuda_12.1
        command:
        - bash
        - -cxue
        - |
          export PATH=/usr/local/nvidia/bin${PATH:+:${PATH}}
          export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

          nvidia-smi

          mkdir -p pytorch/xla
          git clone -b r2.3 https://github.com/pytorch/xla.git pytorch/xla


          torchrun \
            # Set this to the number of hosts
            --nnodes=2 \
            # Index provided by Job
            --node_rank=$(JOB_COMPLETION_INDEX) \
            # Set this to the number of GPUs per host
            --nproc_per_node=2 \
            # Coordinator always runs on 0th instance of job
            --rdzv_endpoint=$(JOB_NAME)-0.headless-svc:12355 \
            # Replace this with your script and flags
            pytorch/xla/test/test_train_mp_imagenet.py \
            --model=resnet50 \
            --log_steps=200 \
            --fake_data \
            --pjrt_distributed \
            --nometrics_debug \
            --num_epochs=1
        env:
        - name: JOB_NAME
          valueFrom:
            fieldRef:
              apiVersion: v1
              fieldPath: metadata.labels['job-name']
        - name: PJRT_DEVICE
          value: CUDA
        resources:
          limits:
            nvidia.com/gpu: "2"
        # PyTorch requires a large `shm`
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      # Change the node selector if you're using a different GPU type
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-v100
      volumes:
      - emptyDir:
          medium: Memory
        name: dshm
```

Once the job schedules, you should start seeing logs like this:

```
$ kubectl logs job/torch-xla-resnet50-v100-x2x2
...
+ nvidia-smi
Fri Jun 28 20:15:43 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-SXM2-16GB           Off |   00000000:00:04.0 Off |                    0 |
| N/A   35C    P0             33W /  300W |       0MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla V100-SXM2-16GB           Off |   00000000:00:05.0 Off |                    0 |
| N/A   35C    P0             33W /  300W |       0MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
+ mkdir -p pytorch/xla
+ git clone -b r2.3 https://github.com/pytorch/xla.git pytorch/xla
Cloning into 'pytorch/xla'...
+ torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 --rdzv_endpoint=torch-xla-resnet50-v100-x2x2-0.headless-svc:12355 pytorch/xla/test/test_train_mp_imagenet.py --model=resnet50 --log_steps=200 --fake_data --pjrt_distributed --nometrics_debug --num_epochs=1
...
I0000 00:00:1719605752.973014      55 service.cc:145] XLA service 0x59c1a6e31500 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1719605752.973052      55 service.cc:153]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
...
==> Preparing data..
==> Preparing data..
Epoch 1 train begin 20:15:56
| Training Device=xla:0/1 Epoch=1 Step=0 Loss=6.89059 Rate=3.13 GlobalRate=3.13 Time=20:16:36
| Training Device=xla:0/0 Epoch=1 Step=0 Loss=6.89059 Rate=3.14 GlobalRate=3.14 Time=20:16:36
...
| Training Device=xla:0/0 Epoch=1 Step=1800 Loss=0.00135 Rate=332.54 GlobalRate=314.11 Time=20:28:09
| Training Device=xla:0/1 Epoch=1 Step=1800 Loss=0.00135 Rate=332.54 GlobalRate=314.06 Time=20:28:09
...
| Training Device=xla:0/0 Epoch=1 Step=2200 Loss=0.00135 Rate=336.66 GlobalRate=318.00 Time=20:30:42
| Training Device=xla:0/1 Epoch=1 Step=2200 Loss=0.00135 Rate=336.66 GlobalRate=317.96 Time=20:30:42
Epoch 1 train end 20:31:36
| Test Device=xla:0/0 Step=0 Epoch=1 Time=20:31:42
| Test Device=xla:0/1 Step=0 Epoch=1 Time=20:31:42
Epoch 1 test end 20:31:47, Accuracy=100.00
Max Accuracy: 100.00%
...
```

## TPUs

For TPUs, the same steps for `torchrun` apply: create one job instance per host
(`--nnodes`) and one process per chip (`--nprocs_per_node`). For more
information about TPU GKE clusters, see [GKE's official
docs](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus).

TODO: TPU example

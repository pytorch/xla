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
uses two hosts, each with two NVidia v100 GPUs. Adjust the values according
to the comments in the example for a larger or smaller cluster.

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
    headless-svc: "true"
  clusterIP: None
---
apiVersion: batch/v1
kind: Job
metadata:
  name: torch-xla-resnet50-v100-x2x2
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
      subdomain: headless-svc
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

          # Run `args` here
          "${@:0}"
        args:
        - torchrun
        # Set this to the number of hosts
        - --nnodes=2
        # Index provided by Job
        - --node_rank=$(JOB_COMPLETION_INDEX)
        # Create one process per local GPU
        - --nproc_per_node=2
        # Coordinator always runs on 0th instance of job
        - --rdzv_endpoint=$(JOB_NAME)-0.headless-svc:12355
        # Replace this with your script and flags
        - pytorch/xla/test/test_train_mp_imagenet.py
        - --model=resnet50
        - --log_steps=200
        - --fake_data
        - --pjrt_distributed
        - --nometrics_debug
        - --num_epochs=1
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
          # Change this to the number of GPUs per host
            nvidia.com/gpu: "2"
        # PyTorch requires a large `shm`
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      restartPolicy: Never
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

Training on TPU is similar to training on GPU in GKE, the same steps for `torchrun` apply. For more
information about TPU GKE clusters, see [GKE's official
docs](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus).

The example below use two ct5lp-hightpu-4t VMs, with 4 v5e TPU each to construct a 2x4 topology nodepool.
You can adjust the values accordingly to match the training requirement.

Create a new file `tpu_test.yaml` with the following:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: headless-svc
  namespace: default
spec:
  clusterIP: None
  selector:
    job-name: torch-xla-tpu-2x4
---
apiVersion: batch/v1
kind: Job
metadata:
  name: torch-xla-tpu-2x4
  labels:
spec:
  parallelism: 2 # num of nodes
  completions: 2 # num of nodes
  backoffLimit: 0 # default, no retries
  completionMode: Indexed
  template:
    spec:  # pod-spec:
      serviceAccountName: default
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions: # need to be specified to get tpu resources
              - key: cloud.google.com/gke-tpu-accelerator
                operator: "In"
                values:
                - "tpu-v5-lite-podslice"
              - key: cloud.google.com/gke-tpu-topology
                operator: "In"
                values:
                - "2x4"  # 2 nodes of 4 tpu's
      tolerations:
      - key: "google.com/tpu"
        operator: "Equal"
        value: "present"
        effect: "NoSchedule"
      restartPolicy: Never # look in https://kubernetes.io/docs/concepts/workloads/controllers/job/
      subdomain: headless-svc
      volumes:
        # Increase size of tmpfs /dev/shm to avoid OOM.
      - name: shm
        emptyDir:
          medium: Memory
      containers:
      - name: training
        image: us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm
        command:
        - bash
        - -cxue
        - |
          apt-get update -y
          apt-get install libomp5 numactl libopenblas-dev -y
          pip3 install mkl mkl-include tf-nightly tb-nightly tbp-nightly numpy
          
          ln -s /usr/local/lib/libmkl_intel_ilp64.so.2 /usr/local/lib/libmkl_intel_ilp64.so.1

          mkdir -p pytorch/xla
          git clone -b r2.3 https://github.com/pytorch/xla.git pytorch/xla

          # Run `args` here
          "${@:0}"
        args:
        - torchrun
        # Set this to the number of hosts
        - --nnodes=2
        # Index provided by Job
        - --node_rank=$(JOB_COMPLETION_INDEX)
        # Create one process per local TPU
        - --nproc_per_node=4
        # Coordinator always runs on 0th instance of job
        - --rdzv_endpoint=$(JOB_NAME)-0.headless-svc:12355
        # Replace this with your script and flags
        - pytorch/xla/test/test_train_mp_imagenet.py
        - --model=resnet50
        - --log_steps=200
        - --fake_data
        - --pjrt_distributed
        - --nometrics_debug
        - --num_epochs=1
        ports:
        - containerPort: 8471  # 8471 is the default port for the TPU VMs communication
        - containerPort: 12355  # used by the code
        - containerPort: 8479
        - containerPort: 8478
        - containerPort: 8477
        - containerPort: 8476
        - containerPort: 8431 # Port to export TPU usage metrics, if supported.
        volumeMounts:
        - mountPath: /dev/shm
          name: shm
        env:
        - name: PJRT_DEVICE
          value: 'TPU'
        - name: XLA_USE_BF16
          value: '1'
        - name: USE_TORCH
          value: 'ON'
        - name: JOB_NAME
          value: 'torch-xla-tpu-2x4'
        resources:
          requests:
            google.com/tpu: 4
            memory: 16G
          limits:
            google.com/tpu: 4
```

Once the job schedules, you should start seeing logs like this:

```
$ kubectl logs job/torch-xla-tpu-2x4
...
Cloning into 'pytorch/xla'...
+ torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 --rdzv_endpoint=torch-xla-tpu-2x4-0.headless-svc:12355 pytorch/xla/test/test_train_mp_imagenet.py --model=resnet50 --log_steps=200 --fake_data --pjrt_distributed --nometrics_debug --num_epochs=1
...
==> Preparing data..
==> Preparing data..
Epoch 1 train begin 23:10:22
|| Training Device=xla:0/3 Epoch=1 Step=0 Loss=6.89059 Rate=4.64 GlobalRate=4.64 Time=23:10:54
 Training Device=xla:0/0 Epoch=1 Step=0 Loss=6.89059 Rate=3.97 GlobalRate=3.97 Time=23:10:54
| Training Device=xla:0/1 Epoch=1 Step=0 Loss=6.89059 Rate=4.13 GlobalRate=4.13 Time=23:10:54
| Training Device=xla:0/2 Epoch=1 Step=0 Loss=6.89059 Rate=3.99 GlobalRate=3.99 Time=23:10:54
...\
| Training Device=xla:0/3 Epoch=1 Step=1000 Loss=0.00139 Rate=1343.24 GlobalRate=864.39 Time=23:12:54
| Training Device=xla:0/2 Epoch=1 Step=1000 Loss=0.00139 Rate=1343.23 GlobalRate=839.12 Time=23:12:54
Epoch 1 train end 23:13:07
| Test Device=xla:0/1 Step=0 Epoch=1 Time=23:13:11
| Test Device=xla:0/3 Step=0 Epoch=1 Time=23:13:11
...
Epoch 1 test end 23:13:16, Accuracy=100.00
Max Accuracy: 100.00%
```

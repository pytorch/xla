# PyTorch/XLA Glossary

This glossary defines common terms used in the PyTorch/XLA documentation.

## A

**Accelerator** - A specialized hardware component designed to accelerate specific computational tasks, such as deep learning. Examples include GPUs and TPUs.

## B

**Barrier** - In the context of PyTorch/XLA, a synchronization point that ensures all operations on XLA tensors have completed before proceeding. It's often used to ensure the host (CPU) and device (TPU/GPU) are synchronized.

**bfloat16** - A 16-bit floating-point data type commonly used on TPUs for faster training.

## C

**CUDA** -  A parallel computing platform and programming model developed by NVIDIA for use with their GPUs.

**Core Aten Op Set** - A collection of fundamental operations from PyTorch's ATen library considered essential for core functionality and model export.

## D

**Data Parallelism** - A parallelization strategy where the same model is replicated on multiple devices, with each device processing a different subset of the training data.

**Device Mesh** - A logical representation of the interconnected devices (TPUs/GPUs) used for distributed training, defining the arrangement and communication paths between them.

**DistributedDataParallel (DDP)** - A PyTorch module that enables data-parallel training across multiple devices, typically used in conjunction with torch.distributed.

**Distributed Tensor** - A PyTorch API for representing tensors distributed across multiple devices, facilitating parallel and distributed computation.

**Dynamo** (See **TorchDynamo**)

## E

**Eager Execution** -  A computational model where operations are executed immediately as they are encountered in the code, as opposed to graph execution.

**Environment Variables** - Variables that can be set outside of a program to control its behavior, often used in PyTorch/XLA to configure runtime options.

## F

**FSDP (Fully Sharded Data Parallel)** - A data-parallel training technique that shards model parameters, gradients, and optimizer states across devices

**FX (TorchFX)** -  An intermediate representation (IR) format used in PyTorch for representing computation graphs in a more structured way.

**Functionalization** - A process of converting eager execution code into a functional representation, allowing for greater optimization and compilation opportunities.

## G

**GSPMD (General and Scalable Parallelization for ML Computation Graphs)** - A single API that enables a large variety of parallelism algorithms (including data parallelism, fully sharded data parallelism, spatial partitioning tensor and pipeline parallelism, as well as combinations of these algorithms) for different ML workloads and model architectures.

## H

**HLO (High-Level Optimizer)** - An intermediate representation (IR) format used by the XLA compiler, representing a computation graph at a higher level than machine code.

**Hugging Face** - A community and platform providing tools and resources for natural language processing, including pre-trained models and a popular Trainer API.

## I

**IR (Intermediate Representation)** - A representation of a program or computation graph that is more abstract than machine code but closer to it than the original source code.

## J

**JAX** - A high-performance numerical computation library developed by Google, known for its automatic differentiation and XLA integration.

**JIT (Just-in-Time Compilation)** -  A compilation strategy where code is compiled at runtime, as needed, offering flexibility and potential optimizations based on runtime information.

## K

**Kaggle** - An online community and platform for machine learning practitioners to share code and solutions.

## L

**Lazy Tensor** -  A type of tensor in PyTorch/XLA that delays operation execution until the results are explicitly needed, allowing for graph optimization and XLA compilation.

**Lit-GPT** - Implements open-source large language models in XLA and supports fine-tuning

## M

**Model Parallelism** - A parallelization strategy where different parts of a model are distributed across multiple devices, enabling training of models too large to fit on a single device.

**Multiprocessing** -  A programming technique for running multiple processes concurrently, often used in PyTorch/XLA to utilize multiple TPU cores.

**MpDeviceLoader** - A PyTorch/XLA utility for efficiently loading and distributing data across multiple devices during training.

## N

**NCCL (NVIDIA Collective Communications Library)** - A library for efficient collective communication operations (e.g., all-reduce, all-gather) on NVIDIA GPUs.

## O

**OpenXLA** - An open-source project aimed at developing and maintaining XLA, the deep learning compiler.

**Ordinal** - A unique identifier for a device (TPU/GPU) within a distributed training setup, often used to determine the device's role and data partitioning.

## P

**Partition Spec** -  In GSPMD, a specification that defines how a tensor is sharded across a device mesh.

**PJRT (Portable JAX Runtime)** - A runtime environment for JAX that supports multiple backends.

**Pod** - A group of interconnected TPU hosts, offering massive scale for training large models.

**Preemption** - An event where a Cloud TPU is reclaimed by the cloud provider, requiring checkpointing to avoid losing training progress.

## R

**Rendezvous** - Used by Torch Distributed Elastic to gather participants of a training job (i.e. nodes) such that they all agree on the same list of participants and everyone’s roles, as well as make a consistent collective decision on when training can begin/resume.

**Replication** - A data distribution strategy where a tensor is fully copied to all devices in a mesh, ensuring all devices have the same data.

## S

**Sharding** - The process of dividing a tensor into smaller pieces (shards) and distributing them across devices, commonly used to reduce memory footprint and enable parallel computation.

**SPMD (Single Program, Multiple Data)** - A parallel programming model where the same program is executed on multiple devices.

**State Dict**- A Python dictionary object that maps each layer to its parameter tensor. It is used for saving or loading models.

## T

**TensorBoard** - A visualization tool for monitoring and analyzing training progress, including performance metrics and computation graphs.

**TorchDynamo** - A Python-level JIT compiler for PyTorch, dynamically modifying bytecode to enable graph capture and optimization.

**TPU (Tensor Processing Unit)** - A custom-designed machine learning accelerator developed by Google, offering high performance for deep learning workloads.

## X

**XLA (Accelerated Linear Algebra)** - A deep learning compiler developed by Google.

**XLATensor** - A tensor type in PyTorch/XLA representing data on an XLA device, enabling lazy execution and XLA compilation.

**xla_device()** - A PyTorch/XLA function for retrieving the current XLA device.

**xm (xla_model)** - A module in PyTorch/XLA providing core functions for interacting with XLA devices and executing computations.

**xmp (xla_multiprocessing)** - A module in PyTorch/XLA for launching distributed training processes across multiple XLA devices.

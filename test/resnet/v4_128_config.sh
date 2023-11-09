#!/bin/bash

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --project $PROJECT  --worker=all --command "date && LIBTPU_INIT_ARGS='--xla_jf_auto_cross_replica_sharding --xla_jf_bounds_check=false --xla_tpu_prefer_binomial_single_phase_ring_emitter=true --xla_jf_single_phase_ring_max_kib=40 --xla_tpu_megacore_fusion_scaling_factor=2.3 --xla_tpu_nd_short_transfer_max_chunks=1536 --xla_tpu_megacore_fusion_latency_bound_ar_fusion_size=6291456' \
  PJRT_DEVICE=TPU XLA_DISABLE_FUNCTIONALIZATION=1    python3 \
  test_resnet.py --model=resnet50 --datadir=/mnt/disks/persist/imagenet \
  --num_epochs=200 --log_steps=76 --profile \
  --host_to_device_transfer_threads=1   --loader_prefetch_size=8 \
  --device_prefetch_size=4 --prefetch_factor=32 --num_workers=16 \
  --persistent_workers --drop_last --base_lr=17.0  --warmup_epochs=5 \
  --weight_decay=2e-4 --eeta=1e-3 --epsilon=0.0 --momentum=0.9 --amp \
  --eval_batch_size=256 --train_batch_size=512 --lmdb"


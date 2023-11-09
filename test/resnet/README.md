<h1> Instructions for running ResNet model </h1>


- <h2> Setup TPU VM Environment using the following instructions </h2>

  <h3>1. Setup env variables </h3>

        
        TPU_NAME=<Your TPU name>
        ACCELERATOR_TYPE=<Your accelerator type>
        ZONE=<TPU zone>
        PROJECT=<PROJECT ID>
  <h3>2. Create TPU </h3>

        gcloud alpha compute tpus tpu-vm create $TPU_NAME \
        --zone $ZONE \
        --accelerator-type $ACCELERATOR_TYPE \ 
        --version tpu-ubuntu2204-base \
        --project $PROJECT

- <h2> Attach SSD Disk </h2>
   TPU only supports attaching disks in read-only mode. Therefore we will follow
   the following strategy.

     - create a fresh disk 
     - attach the disk in read-write mode to TPU VM node
     - download data to the disk 
     - detach disk
     - reattach the disk in read-only mode to all the nodes

      [Detailed Instructions to attach a persistent disk to TPUVM](https://cloud.google.com/tpu/docs/setup-persistent-disk)
  <h3>1. Create a fresh disk </h3>

        `gcloud compute disks create lmdb-imagenet \
        --size 200G \
        --zone $ZONE \
        --type pd-ssd \
        --project $PROJECT`
  <h3>2. attach the disk in read-write mode to TPU VM node </h3>

        `gcloud  alpha compute tpus tpu-vm attach-disk $TPU_NAME \
        --zone=$ZONE \
        --disk=lmdb-imagenet2 \
        --mode=read-write \
        --project$PROJECT`

      Login to TPU VM using

      `gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --project $PROJECT`

      -  Format the disk
         
         `sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb`

      -  Mount the disk to a path

        `sudo mkdir -p /mnt/disks/persist`
      
        `sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist`


  <h3>3. download data to the disk </h3>

      `gsutil -m cp -r gs://imagenet-lmdb/imagenet /mnt/disks/persist`

  <h3>4. detach disk </h3>

      `gcloud alpha compute tpus tpu-vm detach-disk $TPU_NAME  --zone=$ZONE --project=$PROJECT  --disk=lmdb-imagenet`

  <h3>5. reattach the disk in read-only mode to all the TPU nodes </h3>

      `gcloud  alpha compute tpus tpu-vm attach-disk $TPU_NAME \
        --zone=$ZONE \
        --disk=lmdb-imagenet \
        --mode=read-only \
        --project=$PROJECT`

      ` gcloud  alpha compute tpus tpu-vm ssh $TPU_NAME \
        --zone=u$ZONE \
        --worker=all \
        --project=$PROJECT \
        --command "sudo mkdir -p /mnt/disks/persist && \
        sudo mount -o ro,noload /dev/sdb /mnt/disks/persist" `

- <h2> Run the training workload </h2>

      `gcloud  alpha compute tpus tpu-vm ssh $TPU_NAME \
        --zone=u$ZONE \
        --worker=all \
        --project=$PROJECT \
        --command "date && LIBTPU_INIT_ARGS="--xla_jf_auto_cross_replica_sharding --xla_jf_bounds_check=false --xla_tpu_prefer_binomial_single_phase_ring_emitter=true --xla_jf_single_phase_ring_max_kib=40 --xla_tpu_megacore_fusion_scaling_factor=2.3 --xla_tpu_nd_short_transfer_max_chunks=1536 --xla_tpu_megacore_fusion_latency_bound_ar_fusion_size=6291456" \
        PJRT_DEVICE=TPU XLA_DISABLE_FUNCTIONALIZATION=1    python3 \
        test_resnet.py --model=resnet50 --datadir=/mnt/disks/persist/ \
        imagenet/  --num_epochs=46 --log_steps=312 --profile \
        --host_to_device_transfer_threads=1   --loader_prefetch_size=64 \
        --device_prefetch_size=32 --prefetch_factor=32 --num_workers=16 \
        --persistent_workers --drop_last --base_lr=8.0  --warmup_epochs=5 \
        --weight_decay=1e-4 --eeta=1e-3 --epsilon=0.0 --momentum=0.9 --amp \
        --eval_batch_size=256 --train_batch_size=256" `


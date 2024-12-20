Training based on torchtitan llama model
====================================

## Install dependencies

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install optax fire tensorflow tensorboard-plugin-profile
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

cd ~
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
pip install -r requirements.txt
pip install .

cd ~
git clone https://github.com/pytorch/xla.git
cd xla/experimental/torch_xla2
pip install -e .
```

Run the train script

```bash
export LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

python train_llama.py
```



## Detailed numbers

### v5p-8

seqlen = 8192
bs = 8

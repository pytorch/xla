sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly%2B20230222-cp38-cp38-linux_x86_64.whl
sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly%2B20230222-cp38-cp38-linux_x86_64.whl
sudo pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-nightly%2B20230222-cp38-cp38-linux_x86_64.whl

gsutil cp gs://test-example-123/v5/pjrt/0201_torch_pin_libtpu.so ~/.local/lib/python3.8/site-packages/libtpu/libtpu.so

git apply patch_for_torch_serve.diff

git clone -b einsum https://github.com/AlexWertheim/transformers.git
cd transformers
gsutil cp gs://test-example-123/v5/pjrt/gpt/capture_profile.py .
gsutil cp -r gs://test-example-123/my_config_*.json examples/pytorch/language-modeling/

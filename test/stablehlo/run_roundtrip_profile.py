import os

# This is a test script for HLO<->StableHLO roundtrip performance evaluation.
# The script is a driver to run the exisitng test scripts with different networks.
# The metrics are enabled and redirect to the log file.

test_script = "PJRT_DEVICE=TPU python test/spmd/test_train_spmd_imagenet.py --fake_data --num_epochs=1 --metrics_debug --sharding=batch --model="

models = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]

log_base_path = "/workspaces/work/roundtrip_log/profile/"
roundtrip_type = "mhlo"

for model in models:
    cmd = test_script + model
    redirect = " > {} 2>&1".format(os.path.join(log_base_path, "{}_spmd_{}.log".format(roundtrip_type, model)))
    cmd += redirect
    print(cmd)
    os.system(cmd)

import os
import subprocess

def test_local_launch_allreduce():
    # This test launches a allreduce using torchrun launcher
    cmd = "torchrun --nproc_per_node=2  --master_addr=127.0.0.1 --master_port=2020 allreduce_torchrun.py "

    proc0 = subprocess.Popen(cmd, shell=True)

    return_code = proc0.wait()
    assert return_code == 0

def test_local_launch_allreduce_cores():
    # This test launches a allreduce using torchrun launcher
    cmd = "torchrun --nproc_per_node=2  --master_addr=127.0.0.1 --master_port=2020 allreduce_torchrun.py "

    new_env0 = os.environ.copy()
    new_env0['NEURON_RT_VISIBLE_CORES'] = '0,1'

    proc0 = subprocess.Popen(cmd, env=new_env0, shell=True)

    return_code = proc0.wait()
    assert return_code == 0

if __name__ == '__main__':
    test_local_launch_allreduce()
    test_local_launch_allreduce_cores()



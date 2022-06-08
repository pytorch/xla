import torch.distributed as dist
import os
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_env_vars as xenv
from torch_xla.utils.utils import get_free_tcp_ports
import socket

_STATIC_PORT_GRPC = 30000
_NODE_LIST = None
_TCP_STORE = None
_use_tcp_store = True
_INIT_XRT_ALREADY_CALLED = False


def _get_hostnames_slurm():
    # SLURM populates this environment variable
    input_names = os.environ['SLURM_JOB_NODELIST']
    if input_names is None:
        print('SLURM_JOB_NODELIST is None, cannot populate XRT workers')
        return None
    try:
        cmd_str = ['scontrol', 'show', 'hostnames', input_names]
        out = subprocess.run(cmd_str, stdout=subprocess.PIPE, text=True)
    except RuntimeError:
        print('scontrol not present or fails to produce hostnames')
        return None
    return out.stdout.split('\n')[:-1]


def _create_devices(dev_kind, world_size):
    # Create global XLA devices. Adapted from xmp.spawn() to function across nodes
    devices = []
    dev_type = 'NEURONT' if dev_kind == 'NEURON' else 'GPU'

    for gindex in range(0, world_size):
        tfdevice = f'{dev_type}:{gindex};/job:localservice/replica:0/task:{gindex}/device:XLA_{dev_type}:0'
        devices.append(tfdevice)
    os.environ[xenv.DEVICE_MAP] = '|'.join(devices)


def _setup_workers(world_size, rank, use_static_ports, node_list):
    if use_static_ports:
        # Using static ports across all processes. There is risk of port conflicts, however should
        # work with xla backend as no prior communication is needed for xla CC ops to work
        all_workers = []
        # Assuming equal workers per node
        num_workers_per_node = int(world_size) // len(node_list)
        gindex = 0
        assert node_list is not None
        for node in node_list:
            for _ in range(0, num_workers_per_node):
                worker = '{}:{};grpc://{}:{}'.format('localservice', gindex, node, _STATIC_PORT_GRPC + gindex)
                all_workers.append(worker)
                gindex += 1
    else:
        # Set up workers across nodes. xmp.spawn() does this locally by figuring out free ports on the node
        # We do this globally by doing an allgather of locally obtained free socket addresses
        port = get_free_tcp_ports()[0]
        host = socket.getfqdn()
        my_worker = '{}:{};grpc://{}:{}'.format('localservice', rank, host, port)
        all_workers = []
        if _use_tcp_store:
            for i in range(0, world_size):
                if rank == i:
                    _TCP_STORE.set(f'worker:{i}', my_worker)
                    all_workers.append(my_worker)
                else:
                    worker = _TCP_STORE.get(f'worker:{i}').decode('UTF-8')
                    all_workers.append(worker)
        else:
            all_workers = [None for _ in range(0, world_size)]
            dist.all_gather_object(all_workers, my_worker)
    os.environ['XRT_WORKERS'] = '|'.join(all_workers)


def _setup_nccl_service(dev_kind, rank, use_static_ports):
    # Set up NCCL COMM ID required for NCCL communicator IDs
    if use_static_ports:
        assert os.environ['MASTER_ADDR'] is not None
        assert os.environ['MASTER_PORT'] is not None
        if dev_kind == 'NEURON':
            os.environ['NEURON_RT_ROOT_COMM_ID'] = '{}:{}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        else:
            os.environ['XRT_MESH_SERVICE_ADDRESS'] = '{}:{}'.format(os.environ['MASTER_ADDR'],
                                                                    os.environ['MASTER_PORT'])
    else:
        if rank == 0:
            port = get_free_tcp_ports()[0]
            host = socket.getfqdn()
            if _use_tcp_store:
                service_addr = '{}:{}'.format(host, port)
                _TCP_STORE.set('nccl_info', service_addr)
            else:
                service_addr = ['{}:{}'.format(host, port)]
        else:
            if _use_tcp_store:
                service_addr = _TCP_STORE.get('nccl_info').decode('UTF-8')
            else:
                service_addr = [None]
        if not _use_tcp_store:
            dist.broadcast_object_list(service_addr, src=0)
        if dev_kind == 'NEURON':
            os.environ['NEURON_RT_ROOT_COMM_ID'] = service_addr if _use_tcp_store else service_addr[0]
            os.environ["NCCL_COMM_ID"] = os.environ['NEURON_RT_ROOT_COMM_ID']
        else:
            os.environ['XRT_MESH_SERVICE_ADDRESS'] = service_addr[0]


def set_xrt_envs(dev_kind, world_size, rank, local_rank):
    # Set up all the XRT specific env variables, adapted from xmp.spawn()
    os.environ[xenv.WORLD_SIZE] = world_size
    os.environ[xenv.ORDINAL] = rank
    os.environ[xenv.LOCAL_ORDINAL] = local_rank
    os.environ[xenv.LOCAL_WORKER] = 'localservice:' + rank

    if dev_kind == 'NEURON':
        os.environ[xenv.MP_DEVICE] = 'NEURONT:' + rank
        os.environ["NEURON_USE_LOAD_COLLECTIVES"] = "1"
        os.environ['NEURON_GLOBAL_DEVICE_ID'] = rank
        os.environ['NEURON_GLOBAL_DEVICE_COUNT'] = world_size
        cores_to_use = os.environ.get('NEURON_RT_VISIBLE_CORES') 
        if cores_to_use is not None:
            # If a the cores are set by a scheduling entity (eg. SLURM) we index into 
            # comma separated string containing numbered cores
            cores_to_use_list = cores_to_use.split(',')
            os.environ['NEURON_RT_VISIBLE_CORES'] = cores_to_use_list[int(local_rank)]
        else:
            # If no explicit visible cores are provided, local_rank is used to identify
            # the core used by this process
            os.environ['NEURON_RT_VISIBLE_CORES'] = local_rank
    else:
        os.environ[xenv.MP_DEVICE] = 'GPU:' + rank
        gpus_to_use = os.environ.get('CUDA_VISIBLE_DEVICES')
        if gpus_to_use is not None:
            # If gpu devices are set by a scheduling entity (eg. SLURM) we index into 
            # comma separated string containing numbered gpu devies 
            gpus_to_use_list = gpus_to_use.split(',')
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use_list[int(local_rank)]
        else:
            # If no explicit visible devices are provided, local_rank is used to identify
            # the gpu used by this process
            os.environ['CUDA_VISIBLE_DEVICES'] = local_rank


def _init_xrt_context(dev_kind='NEURON', master_addr=None, master_port=None, store=None):
    '''
    Initializes the XLA runtime on a given device. Or is a no-op if init_xrt_context
    has already been called.

    Keyword Arguments:
    dev_kind    -- 'NEURON' or 'GPU' (default 'NEURON') -- specifies device type
    master_addr -- int(SomeAddrNumber) (default tries to get addr from
                    environment variable, used to setup TCPStore)
                    Not needed if store is provided.
    master_port -- int(SomePortNumber) (default tries to get port from
                    environment variable, used to setup TCPStore)
                    Not needed if store is provided.
    store       -- A TCPstore object (default None) -- If None a TCPStore object
                    will be setup for you
    '''
    global _INIT_XRT_ALREADY_CALLED

    if _INIT_XRT_ALREADY_CALLED:
        return

    # Call this in the actual test case, to work with torch/xla workers
    # This internally creates a process group using "Gloo" backend which is used only
    # for exchanging the IP and port addresses.
    rank = os.environ.get('RANK')
    assert rank is not None

    local_rank = os.environ.get('LOCAL_RANK')
    assert local_rank is not None

    world_size = os.environ.get('WORLD_SIZE')
    assert world_size is not None

    if master_addr==None:
        master_addr = os.environ.get('MASTER_ADDR')
        assert master_addr is not None

    if master_port==None:
        master_port = os.environ.get('MASTER_PORT')
        assert master_port is not None

    set_xrt_envs(dev_kind, world_size, rank, local_rank)
    _create_devices(dev_kind, int(world_size))
    # follow torch_xla/distributed/xla_multiprocessing.py#L276 to correctly set workers
    os.environ.pop('NEURON_NUM_DEVICES', None)

    # This is required if we want to dynamically grab free ports.
    # Useful in shared settings when we cannot predetermine what ports are taken.
    # Cannot use xla_backend now as it requires xrt worker set-up first, which again
    # requires communication set-up/
    if _use_tcp_store:
        is_server = True if rank is '0' else False
        global _TCP_STORE
        if store is None:
            assert master_addr is not None
            assert master_port is not None
            _TCP_STORE = dist.TCPStore(master_addr, int(master_port), int(world_size), is_server)
        else:
            _TCP_STORE = store
    else:
        dist.init_process_group("gloo")
    node_list = None

    _setup_workers(int(world_size), int(rank), False, node_list)
    _setup_nccl_service(dev_kind, int(rank), False)

    dev = xm.xla_device()
    xm.set_replication(dev, [dev])

    # if we get to this point, we know the function completed successfully
    # and we can switch the flag to True
    _INIT_XRT_ALREADY_CALLED = True


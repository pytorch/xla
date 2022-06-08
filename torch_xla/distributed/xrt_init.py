import torch.distributed as dist
import os
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_env_vars as xenv
from torch_xla.utils.utils import get_free_tcp_ports
import socket

_STATIC_PORT_GRPC = 30000
_NODE_LIST = None
_TCP_STORE = None
_INIT_XRT_ALREADY_CALLED = False


def _create_devices(dev_kind, world_size):
    # Create global XLA devices. Adapted from xmp.spawn() to function across nodes
    devices = []
    dev_type = 'NEURONT' if dev_kind == 'NEURON' else 'GPU'

    for gindex in range(0, world_size):
        tfdevice = f'{dev_type}:{gindex};/job:localservice/replica:0/task:{gindex}/device:XLA_{dev_type}:0'
        devices.append(tfdevice)
    os.environ[xenv.DEVICE_MAP] = '|'.join(devices)


def _setup_workers(world_size, rank, local_world_size, local_rank, node_list):
    # Set up workers across nodes. xmp.spawn() does this locally by figuring out free ports on the node
    # We do this globally by doing an allgather of locally obtained free socket addresses
    host = socket.gethostname()
    if local_rank == 0:
        ports = [str(i) for i in get_free_tcp_ports(local_world_size)]
        _TCP_STORE.set(host, ' '.join(ports))
    else:
        ports_str = _TCP_STORE.get(host).decode('UTF-8')
        ports = list(ports_str.split(' '))

    my_worker = '{}:{};grpc://{}:{}'.format('localservice', rank, host, ports[local_rank])
    all_workers = []
    for i in range(0, world_size):
        if rank == i:
            _TCP_STORE.set(f'worker:{i}', my_worker)
            all_workers.append(my_worker)
        else:
            worker = _TCP_STORE.get(f'worker:{i}').decode('UTF-8')
            all_workers.append(worker)
    os.environ['XRT_WORKERS'] = '|'.join(all_workers)


def _setup_nccl_service(dev_kind, rank, use_static_ports):
    # Set up NCCL COMM ID required for NCCL communicator IDs
    if rank == 0:
        port = get_free_tcp_ports()[0]
        host = socket.getfqdn()
        service_addr = '{}:{}'.format(host, port)
        _TCP_STORE.set('nccl_info', service_addr)
    else:
        service_addr = _TCP_STORE.get('nccl_info').decode('UTF-8')

    if dev_kind == 'NEURON':
        os.environ['NEURON_RT_ROOT_COMM_ID'] = service_addr
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
        os.environ['NEURON_RT_VISIBLE_CORES'] = local_rank
    else:
        os.environ[xenv.MP_DEVICE] = 'GPU:' + rank
        os.environ['CUDA_VISIBLE_DEVICES'] = local_rank


def _init_xrt_context(dev_kind='NEURON', master_addr=None, master_port=None, store=None):
    '''
    Initializes the XLA device depending on the kind of the device. Or is a no-op if init_xrt_context
    has already been called.

    Keyword Arguments:
    dev_kind    -- 'NEURON' or 'GPU' (default 'NEURON') -- specifies device kind
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

    local_world_size = os.environ.get('LOCAL_WORLD_SIZE')
    assert local_world_size is not None

    set_xrt_envs(dev_kind, world_size, rank, local_rank)
    _create_devices(dev_kind, int(world_size))
    # follow torch_xla/distributed/xla_multiprocessing.py#L276 to correctly set workers
    os.environ.pop('NEURON_NUM_DEVICES', None)

    # This is required if we want to dynamically grab free ports.
    # Useful in shared settings when we cannot predetermine what ports are taken.
    is_server = True if rank is '0' else False
    global _TCP_STORE
    if store is None:
        assert master_addr is not None
        assert master_port is not None
        _TCP_STORE = dist.TCPStore(master_addr, int(master_port), int(world_size), is_server)
    else:
        _TCP_STORE = store

    node_list = None

    _setup_workers(int(world_size), int(rank), int(local_world_size), int(local_rank), node_list)
    _setup_nccl_service(dev_kind, int(rank), False)

    dev = xm.xla_device()
    xm.set_replication(dev, [dev])

    # if we get to this point, we know the function completed successfully
    # and we can switch the flag to True
    _INIT_XRT_ALREADY_CALLED = True


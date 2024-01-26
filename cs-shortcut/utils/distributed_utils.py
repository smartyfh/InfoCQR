import torch.distributed as dist
import torch
import os
import datetime
import pickle


def is_main():
    return get_rank() == 0


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def init_distributed(args):
    if args.distributed:
        print("initiating distributed system...")
        print("master address: {}:{}".format(os.environ.get('MASTER_ADDR'),
              os.environ.get('MASTER_PORT')))
        print("local rank: {}".format(args.local_rank))
        os.environ['NCCL_DEBUG']='INFO'
        n_gpu = 1
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                timeout=datetime.timedelta(0, 60 * 60 * 12))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    args.n_all_gpu = torch.cuda.device_count()
    return args


def dist_print(text, logger=None):
    if not is_main():
        return
    
    if logger:
        logger.info(text)
    else:
        print(text)


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError(
            'encoded data exceeds max_size, this can be fixed by increasing buffer size: {}'.format(enc_size))

    rank = get_rank()
    world_size = get_world_size()
    buffer_size = max_size * world_size

    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    assert enc_size < 256 ** SIZE_STORAGE_BYTES, 'Encoded object size should be less than {} bytes'.format(
        256 ** SIZE_STORAGE_BYTES)

    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder='big')

    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES: enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start: start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder='big')
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES: size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )

        
def data_sharding(dataset, world_size=1, rank=0):
    if world_size == 1:
        return dataset
    
    rest = len(dataset) % world_size
    shard_size = int(len(dataset) / world_size)
    
    sharded_dataset = []
    
    i = 0
    for i in range(world_size):
        shard = dataset[shard_size * i: shard_size * (i + 1)]
        sharded_dataset.append(shard)
        
    if rest:
        d = dataset[shard_size * (i+1):]
        if isinstance(sharded_dataset[-1], list):
            sharded_dataset[-1].extend(d)
        else:
            t = torch.cat([sharded_dataset[-1], d], 0)
            sharded_dataset[-1] = t
    return sharded_dataset[rank]


def all_gather_items(local_items, world_size=1, rank=-1, max_buffer_size=592000, add_offset=False):
    need_to_concat = []
    device = None
    for idx, item in enumerate(local_items):
        if isinstance(item, torch.Tensor):
            device = item.device
            local_items[idx] = torch.empty_like(item).cpu().copy_(item).detach_()
            need_to_concat.append(idx)
    
    if world_size <= 1:
        return local_items
    
    global_outputs_list = all_gather_list(
        local_items,
        max_size=max_buffer_size,
    )
    
    global_items = [[] for _ in  range(len(local_items))]
    offset = 0
    for i, gathered_outputs in enumerate(global_outputs_list):
        if i != rank:
            items = gathered_outputs
        else:
            items = local_items

        for idx, item in enumerate(items):
            if isinstance(item, torch.Tensor):
                global_items[idx].append(item.to(device))
            else:
#                 if add_offset:
#                     item = [v + offset for v in item]
                global_items[idx].extend(item)
#         offset += items[1].size(0)  # context vectors
    for idx in need_to_concat:
        global_items[idx] = torch.cat(global_items[idx], dim=0)
    return global_items

import torch 


# images usally stored as [n, h, w, ch]
# pytorch format: [ch, h, w] -> e.g., tensor.permute(2, 0, 1)
trsfrm_to_tensor        = lambda x: torch.as_tensor(x, dtype=torch.float32)
trsfrm_tensor_resize    = lambda x, dim: x.reshape(*dim)
trsrm_tensor_format     = lambda x, dim: x.permute(*dim)    
trsfrm_tensor_to_device = lambda x, device: x.to(device)

get_device_from_tensor  = lambda x: x.device
get_type_from_tensor    = lambda x: x.type()
get_shape_from_tensor   = lambda x: x.shape
get_value_from_tensor   = lambda x: x.item()

get_device              = lambda: torch.device("cuda" if torch.cuda.is_available()  else "cpu")
is_gpu_avail            = lambda: torch.cuda.is_available()


def torch_seed_init(seed:int=42) -> None:
    """seeds torch environment and device as applicable

    Args:
        seed (int, optional): seed to configure. Defaults to 42.
    """
    torch.manual_seed(seed)
    if is_gpu_avail():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = True



def batch_to_device(batch: dict, target_device: torch.device):
    """Send a pytorch batch to a device (CPU/GPU).

    Args:
        batch (dict): batch of data
        target_device (torch.device): device as cpu or gpu

    Returns:
        torch.Tensor: send batches of data to device
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

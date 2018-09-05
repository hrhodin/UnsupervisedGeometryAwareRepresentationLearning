import torch
import numpy as np

def nestedDictToDevice(nested_dict_list, device):
    """
    Converts a nested list or dict of Tensors to CPU Tensors
    :param nested_list: A list or dict of list or dict of Tensors
    :return: Same dtatastructure converted to the target device
    """
    
    if isinstance(nested_dict_list, dict):
        nested_dict_device = {}
        for key, val in nested_dict_list.items():
            if isinstance(val, (dict,list)):
                nested_dict_device[key] = nestedDictToDevice(val, device)
            else:
                nested_dict_device[key] = val.to(device=device)
        return nested_dict_device
    else:
        nested_list_device = []
        for val in nested_dict_list:
            if isinstance(val, (dict,list)):
                nested_list_device.append(nestedDictToDevice(val, device))
            else:
                nested_list_device.append(val.to(device=device))
        return nested_list_device

# TODO: use alternative collate function
class PostFlattenInputSubbatchTensorIter(object):
    def __init__(self, data_loader_orig):
        self.data_loader_orig = data_loader_orig
        self.iterator = iter(data_loader_orig)

    def __iter__(self):
        return self

    def __next__(self):
        inputs, labels = next(self.iterator)

        inputs_flat = {}
        for key in inputs:
            x = inputs[key]
            if isinstance(x, torch._C._TensorBase):
                inputs_flat[key] = x.view([-1, *list(x.size())[2:]]) # squeeze first dimension
            elif type(x).__module__ == 'numpy' and type(x).__name__ == 'ndarray':
            #if x.dtype.kind in {'U', 'S'}:
                inputs_flat[key] = np.reshape(x,[-1, *x.shape[2:]]) # merge first two dimensions into one
            else:
                inputs_flat[key] = x

        labels_flat = {}
        for key in labels:
            x = labels[key]
            if isinstance(x, torch._C._TensorBase):
                labels_flat[key] = x.view([-1, *list(x.size())[2:]]) # squeeze first dimension
            elif type(x).__module__ == 'numpy' and type(x).__name__ == 'ndarray':
            #if x.dtype.kind in {'U', 'S'}:
                labels_flat[key] = np.reshape(x,[-1, *x.shape[2:]]) # merge first two dimensions into one
            else:
                labels_flat[key] = x

        return inputs_flat, labels_flat

    def __len__(self):
        return len(self.iterator)

    def __del__(self):
        del self.iterator

class PostFlattenInputSubbatchTensor(object):
    """Flattens the first two dimensions (batch and sub-batch) to make subbatch data compatible with networks accepting a single image"""
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        return PostFlattenInputSubbatchTensorIter(self.data_loader)

import collections
def default_collate_with_string(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    _use_shared_memory = False
    numpy_type_map = {
        'float64': torch.DoubleTensor,
        'float32': torch.FloatTensor,
        'float16': torch.HalfTensor,
        'int64': torch.LongTensor,
        'int32': torch.IntTensor,
        'int16': torch.ShortTensor,
        'int8': torch.CharTensor,
        'uint8': torch.ByteTensor,
    }
    string_classes = (str, bytes)
    if torch.is_tensor(batch[0]):
        #print("IN","torch.is_tensor(batch[0])")
        #IPython.embed()
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        #print("batch:",[e.numpy().shape for e in batch])
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        #print("IN", "type(batch[0]).__module__ == 'numpy'")
        #IPython.embed()
        if type(elem).__name__ == 'ndarray':
            if elem.dtype.kind in {'U', 'S'}:
                return np.stack(batch, 0)
            else:
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate_with_string([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate_with_string(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

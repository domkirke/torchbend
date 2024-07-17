import numpy as np, os, torch, torch.nn as nn, sys, copy, bisect, random
from . import distributions as dist
from collections import OrderedDict
from typing import List, Union, Tuple


class TestModule(nn.Module):
    def __init__(self, nlayers=3):
        super().__init__()
        self.nlayers = nlayers
        self.pre_conv = nn.Conv1d(1, 16, 1)
        self.conv_modules = nn.ModuleList([nn.Conv1d(16, 16, 3) for _ in range(nlayers)])
        self.activations = nn.ModuleList([nn.Tanh() for _ in range(nlayers)])

    def forward(self, x):
        out = self.pre_conv(x)
        for i in range(self.nlayers):
            out = self.conv_modules[i](out)
            out = self.activations[i](out)
        return out
        # return torch.distributions.Normal(out, torch.ones_like(out))


def parse_slice(item, length):
    start = item.start or 0
    stop = item.stop or length
    start = start if start >= 0 else length + start
    stop = start if stop >= 0 else length + stop
    return start, stop, item.step

def parse_files_in_folder(dirpath, ext):
    ext = checklist(ext)
    listfiles = os.listdir(dirpath)
    return list(filter(lambda x: os.path.splitext(x)[1] in ext, listfiles))

def checkdist(obj):
    if obj is None:
        return obj
    elif isinstance(obj, str):
        return getattr(dist, obj)
    elif issubclass(obj, dist.Distribution):
        return obj
    else:
        raise TypeError('obj %s does not seem to be a distribution')

def check_shape(shape, fill_value: int = 1):
    shape = list(shape)
    for i, s in enumerate(shape):
        if s is None:
            shape[i] = fill_value
    return tuple(shape)

def checksize(obj):
    if obj is None:
        return None
    elif isinstance(obj, int):
        return torch.Size([obj])
    else:
        try:
            return torch.Size(obj)
        except:
            raise TypeError("could not convert %s in torch.Size"%type(obj))


def filter_nans(x: np.ndarray, y: np.ndarray = None):
    idxs = np.where(np.isnan(x))
    if y is None:
        x[idxs] = 0.
    else:
        x[idxs] = y[idxs]


def checklist(item, n=1, copy=False):
    """Repeat list elemnts
    """
    if not isinstance(item, (list, )):
        if copy:
            item = [copy.deepcopy(item) for _ in range(n)]
        elif isinstance(item, torch.Size):
            item = [i for i in item]
        else:
            item = [item]*n
    return item

def checktuple(item, n=1, copy=False, list_ok=False):
    """Check tuple"""
    if not isinstance(item, tuple):
        if isinstance(item, list) and list_ok:
            item = tuple(*item)
        if copy:
            item = tuple([copy.deepcopy(item) for _ in range(n)])
        else:
            item = tuple([item]*n)
    return item


def checkdir(directory):
    """Check directory existence. Create a directory if necessary
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)


def checktensor(tensor, dtype=None, allow_0d=True):
    if isinstance(tensor, list):
        return [checktensor(t, dtype=dtype) for t in tensor]
    elif isinstance(tensor, tuple):
        return tuple([checktensor(t, dtype=dtype) for t in tensor])
    elif isinstance(tensor, dict):
        return {k: checktensor(v, dtype=dtype) for k, v in tensor.items()}
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor).to(dtype=dtype)
    elif torch.is_tensor(tensor):
        tensor = tensor.to(dtype=dtype)
        if tensor.ndim == 0 and not allow_0d:
            tensor = torch.Tensor([tensor])
        return tensor
    else:
        if hasattr(tensor, "__iter__"):
            tensor = torch.Tensor(tensor, dtype=dtype)
        else:
            tensor = torch.tensor(tensor, dtype=dtype)
        if tensor.ndim == 0 and not allow_0d:
            tensor = torch.Tensor([tensor])
        return tensor


def checknumpy(tensor):
    if isinstance(tensor, list):
        return [checknumpy(t) for t in tensor]
    elif isinstance(tensor, tuple):
        return tuple([checknumpy(t) for t in tensor])
    elif isinstance(tensor, dict):
        return {k: checknumpy(v) for k, v in tensor.items()}
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()

def flatten_batch(x: torch.Tensor, dim: int = -1):
    batch_size = x.shape[:dim]
    event_size = x.shape[dim:]
    reshape_size = torch.Size([-1]) + event_size
    return x.reshape(reshape_size), batch_size

def reshape_batch(x: torch.Tensor, batch_size: List[int], dim: int = -1):
    event_size = x.shape[dim:]
    return x.reshape(batch_size + event_size)


def print_stats(k, v):
    print(f"{k}: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}, std={v.std():.3f}")


def print_module_grads(module):
    for k, v in module.named_parameters():
        v = v.grad
        if v is None:
            print(f'{k}: None')
        else:
            print_stats(k, v.grad)

def trace_distribution(distribution, name="", scatter_dim=False):
    if name != "":
        name = name + "_"
    if isinstance(distribution, dist.Normal):
        if scatter_dim:
            return {**{f'{name}mean/dim_{i}': distribution.mean[..., i] for i in range(distribution.mean.shape[-1])},
                    **{f'{name}std/dim_{i}': distribution.stddev[..., i] for i in range(distribution.stddev.shape[-1])}}
        else:
            return {f"{name}mean": distribution.mean, f"{name}std": distribution.stddev}
    elif isinstance(distribution, (dist.Bernoulli, dist.Categorical)):
        if scatter_dim:
            return {**{f'{name}probs/dim_{i}': distribution.probs[..., i] for i in range(distribution.probs.shape[-1])}}
        else:
            return {f"{name}probs": distribution.probs}
    elif torch.is_tensor(distribution):
        return {f"{name}": distribution}


def get_shape_from_ratio(n_item, target_ratio):
    i = np.ceil(np.sqrt(n_item / np.prod(target_ratio)))
    h, w = target_ratio[1] * i, target_ratio[0] * i
    return int(h), int(w)


def kronecker(A, B):
    """
    Kronecker product of two incoming matrices A and B
    Args:
        A (torch.Tensor): b x _ x _ matrix
        B (torch.Tensor): b x _ x _ matrix

    Returns:
        out (torch.Tensor): Kronkecker product
    """
    assert len(A.shape) == len(B.shape) == 3, "kroncker takes b x _ x _ matrices"
    requires_grad = A.requires_grad or B.requires_grad
    out = torch.zeros(A.shape[0], A.size(1)*B.size(1),  A.size(2)*B.size(2), requires_grad=requires_grad, device=A.device)
    for i in range(A.shape[0]):
        out[i] =  torch.einsum("ab,cd->acbd", A[i], B[i]).contiguous().view(A.size(1)*B.size(1),  A.size(2)*B.size(2))
    return out


def pad_array(arr: np.array, target_size: int, dim: int):
    if arr.shape[dim] > target_size:
        return arr
    tensor_size = list(arr.shape)
    tensor_size[dim] = target_size - arr.shape[dim]
    cat_tensor = np.zeros(tensor_size, dtype=arr.dtype)
    return np.concatenate([arr, cat_tensor], axis=dim)


def pad(tensor: torch.Tensor, target_size: int, dim: int):
    if tensor.size(dim) > target_size:
        return tensor
    tensor_size = list(tensor.shape)
    tensor_size[dim] = target_size - tensor.shape[dim]
    cat_tensor = torch.zeros(
        tensor_size, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, cat_tensor], dim=dim)

def frame(tensor: torch.Tensor, wsize: int, hsize: int, dim: int):
    if dim < 0:
        dim = tensor.ndim + dim
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    n_windows = tensor.shape[dim] // hsize
    tensor = pad(tensor, n_windows * hsize + wsize,  dim)
    shape = list(tensor.shape)
    shape[dim] = n_windows
    shape.insert(dim+1, wsize)
    # shape = shape[:dim] + (n_windows, wsize) + shape[dim+1:]
    strides = [tensor.stride(i) for i in range(tensor.ndim)]
    strides.insert(dim, hsize)
    # strides = strides[:dim] + (hsize,) + strides[dim:]
    # strides = list(strides[dim:], (strides[dim]*hsize) + [hsize * new_stride] + list(strides[dim+1:])
    return torch.as_strided(tensor, shape, strides)


def overlap_add(array, wsize, hsize, window=None):
    batch_shape = array.shape[:-2]
    array_shape = (*batch_shape, hsize * array.shape[-2] + wsize)
    buffer = np.zeros(*batch_shape, array_shape)
    for i in range(array.shape[-2]):
        idx = [slice(None)] * len(batch_shape) + [i, slice(None)]
        current_slice = array.__getitem__(idx)
        buffer[..., i * hsize:i*hsize + wsize] += current_slice
    return buffer


def unwrap(tensor: torch.Tensor):
    """
    unwrap phase for tensors
    :param tensor: phase to unwrap (seq x spec_bin)
    :return: unwrapped phase
    """
    if isinstance(tensor, list):
        return [unwrap(t) for t in tensor]
    if tensor.ndimension() == 2:
        unwrapped = tensor.clone()
        diff = tensor[1:] - tensor[:-1]
        ddmod = (diff + torch.pi)%(2 * torch.pi) - torch.pi
        mask = (ddmod == -torch.pi).bitwise_and(diff > 0)
        ddmod[mask] = torch.pi
        ph_correct = ddmod - diff
        ph_correct[diff.abs() < torch.pi] = 0
        unwrapped[1:] = tensor[1:] + torch.cumsum(ph_correct, 1)
        return unwrapped
    else:
        return torch.stack([unwrap(tensor[i]) for i in range(tensor.size(0))], dim=0)


def fdiff(x, order=2):
    if order == 1:
        inst_f = torch.cat([x[0].unsqueeze(0), (x[1:] - x[:-1])/2], axis=0)
    elif order == 2:
        inst_f = torch.cat([x[0].unsqueeze(0), (x[2:] - x[:-2])/4, x[-1].unsqueeze(0)], axis=0)
    return inst_f


def fint(x, order=1):
    if order == 1:
        out = x
        out[1:] = out[1:] * 2
        if torch.is_tensor(x):
            out = torch.cumsum(out, axis=0)
        else:
            out = torch.cumsum(out, axis=0)
    elif order == 2:
        out = torch.zeros_like(x)
        out[0] = x[0]; out[-1] = x[-1]

        for i in range(2, x.shape[0], 2):
            out[i] = out[i-2] + 4 * x[i-1]
        for i in reversed(range(1, x.shape[0], 2)):
            out[i-2] = out[i] - 4 * x[i-1]
    return out

def crop_data(tensor_list):
    assert len(tensor_list) != 0
    assert [tensor_list[0].ndim == tensor_list[i].ndim for i in range(len(tensor_list))]
    target_shape = []
    for i in range(tensor_list[0].ndim):
        target_shape.append(min([tensor_list[j].shape[i] for j in range(len(tensor_list))]))
    cropped_list = []
    for i in range(len(tensor_list)):
        cropped_list.append(tensor_list[i].as_strided(target_shape, tensor_list[i].stride()))
    return torch.stack(cropped_list)

def pad_data(tensor_list):
    assert len(tensor_list) != 0
    assert [tensor_list[0].ndim == tensor_list[i].ndim for i in range(len(tensor_list))]
    target_shape = []
    for i in range(tensor_list[0].ndim):
        target_shape.append(max([tensor_list[j].shape[i] for j in range(len(tensor_list))]))
    padded_list = []
    for i in range(len(tensor_list)):
        pad_length = []
        for j in range(len(target_shape)-1,-1,-1):
            pad_length.extend([0, target_shape[j] - tensor_list[i].shape[j]])
        padded_list.append(torch.nn.functional.pad(tensor_list[i], pad_length))
    return torch.stack(padded_list)
    


class ContinuousSlice(object):
    def __init__(self, *args):
        self.start = None
        self.stop = None
        self.step = None
        assert len(args) > 1
        if len(args) == 2:
            self.start, self.stop = args
        elif len(args) == 3:
            self.start, self.stop, self.step = args


class ContinuousList(object):
    def __init__(self, *args, append=False, drop_with_offset=None, default_values=None):
        if len(args) == 0:
            self._hash = {}
            self._ordered_values = []
        elif len(args) == 1:
            if isinstance(args[0], ContinuousList):
                self._hash = copy.copy(args[0]._hash)
                self._ordered_values = copy.copy(args[0]._ordered_values)
            elif isinstance(args[0], dict):
                # assert functools.reduce(lambda x, y: x and isinstance(y, float), args[0].keys(), True), \
                #     "dict keys must be floats"
                self._hash = args[0]
                self._ordered_values = sorted(list(self._hash.keys()))
            else:
                raise TypeError("ContinuousList must be initialised with ContinousList, dict, or a list of values.")
        else:
            self._hash = {i: args[i] for i in range(len(args))}
        self.append = append
        self.drop_with_offset = drop_with_offset

    def get_idx(self, key):
        idx = bisect.bisect_left(self._ordered_values, key)
        return idx

    def __iter__(self):
        _ord = self._ordered_values + [None]
        for i in range(len(self._ordered_values)):
            yield _ord[i], _ord[i+1], self._hash[_ord[i]]

    def __contains__(self, item):
        return item in self._ordered_values

    def __setitem__(self, key, value):
        # try:
        #     key = float(key)
        # except TypeError:
        #     raise TypeError('item assignement for ContinuousList is only valid for float')

        if self._hash.get(key) is None:
            idx = self.get_idx(key)
            self._ordered_values.insert(idx, key)
        else:
            if self.append:
                if isinstance(self._hash[key], list):
                    self._hash[key].append(value)
                    return
        self._hash[key] = value

    def __getitem__(self, item, drop_with_offset=None):
        drop_with_offset = drop_with_offset or self.drop_with_offset
        if isinstance(item, tuple):
            item = ContinuousSlice(*item)
        if torch.is_tensor(item):
            if len(item.shape) == 0:
                item = item.item()
        if hasattr(item, "__iter__"):
            if drop_with_offset == 'absolute':
                idxs = [self.get_idx(i) for i in item]
                return {i: self._hash[i] for i in idxs}
            elif drop_with_offset == 'relative':
                idxs = [self.get_idx(i) for i in item]
                return {i-min(idxs): self._hash[i] for i in idxs}
            else:
                return [self.__getitem__(i) for i in item]
        elif isinstance(item, (slice, ContinuousSlice)):
            start = item.start; end = item.stop
            if item.step is not None:
                raise NotImplementedError
            if start is None:
                hash_keys = self._ordered_values[:self.get_idx(end)]
            elif end is None:
                hash_keys = self._ordered_values[self.get_idx(start):]
            else:
                start_idx = self.get_idx(start); end_idx = self.get_idx(end)
                if start_idx == end_idx:
                    if start_idx < len(self._ordered_values):
                        hash_keys = [self._ordered_values[self.get_idx(start)]]
                    else:
                        hash_keys = [self._ordered_values[-1]]
                else:
                    hash_keys = self._ordered_values[self.get_idx(start):self.get_idx(end)]
            if drop_with_offset == "absolute":
                return {k: self._hash[k] for k in hash_keys}
            if drop_with_offset == "relative":
                return {k-hash_keys[0]: self._hash[k] for k in hash_keys}
            else:
                return [self._hash[k] for k in hash_keys]
        else:
            # item = float(item)
            idx = max(min(self.get_idx(item)-1, len(self._ordered_values)), 0)
            # print(idx)
            if drop_with_offset == "absolute":
                return {self._ordered_values[idx]: self._hash[self._ordered_values[idx]]}
            elif drop_with_offset == "relative":
                return {0: self._hash[self._ordered_values[idx]]}
            else:
                return self._hash[self._ordered_values[idx]]

    def __repr__(self):
        return str({i: self._hash[i] for i in self._ordered_values})


def _recursive_to(obj, device, clone=False):
    if isinstance(obj, OrderedDict):
        return OrderedDict({k: _recursive_to(v, device, clone) for k, v in obj.items()})
    if isinstance(obj, dict):
        return {k: _recursive_to(v, device, clone) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_recursive_to(o, device, clone) for o in obj]
    elif torch.is_tensor(obj):
        if clone:
            return obj.to(device=device).clone()
        else:
            return obj.to(device=device)
    else:
        raise TypeError('type %s not handled by _recursive_to'%type(obj))


def is_abstract(obj):
    if hasattr(obj, "__isabstractmethod__"):
        return True
    else:
        return False


__COPY_LIST = [
    "_parameters",
    "_buffers",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_is_full_backward",
    "_forward_hooks",
    "_forward_hooks_with_kw",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kw",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_modules",
    "_versions"
]


def _get_model_copy(model, copy_parameters=False):
    """Make a bendable copy of a model, just copying internal dicts of submodules
       without deep-copying parameters"""
    model_copy = copy.copy(model)
    for attr in dir(model_copy):
        if attr == "_modules":
            continue
        if attr in __COPY_LIST:
            setattr(model_copy, attr, getattr(model_copy, attr).copy())
    if copy_parameters:
        parameters = OrderedDict([(k, copy.copy(m)) for k, m in model._parameters.items()])
        model_copy._parameters = parameters
    else:
        model_copy._parameters = model._parameters.copy()
    
    model_copy._buffers = model._buffers.copy()
    model_copy._modules = {}
    for name, mod in model._modules.items():
        if mod is not None:
            model_copy._modules[name] = _get_model_copy(mod)
    return model_copy


def get_random_hash(n=8):
    return "".join([chr(random.randrange(97,122)) for i in range(n)])



__all__= ['TestModule', 'get_random_hash']

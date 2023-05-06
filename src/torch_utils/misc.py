# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import re
import contextlib
from typing import List, Set
import numpy as np
import torch
import warnings
from src import dnnlib

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

#----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0

#----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672

@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

#----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

#----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().

def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

#----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src, dst, require_all: bool=False, verbose: bool=False):
    if isinstance(src, torch.nn.Parameter):
        assert isinstance(dst, torch.nn.Parameter), f"Wrong type: {type(dst)}"
        dst.data.copy_(src.data)
    elif isinstance(src, torch.nn.Module):
        assert isinstance(dst, torch.nn.Module), f"Wrong type: {type(src)}"
        src_tensors = dict(named_params_and_buffers(src))
        trg_tensors = dict(named_params_and_buffers(dst))
        extra_keys = [k for k in src_tensors if not k in trg_tensors]
        if len(extra_keys) > 0 and verbose:
            print('extra keys:', extra_keys)
        for name, tensor in trg_tensors.items():
            if name.endswith('w_to_style.model.0.weight') and not name in src_tensors:
                tensor.copy_(src_tensors[name.replace('w_to_style.model.0.weight', 'affine.weight')].detach()).requires_grad_(tensor.requires_grad)
            elif name.endswith('w_to_style.model.0.bias') and not name in src_tensors:
                tensor.copy_(src_tensors[name.replace('w_to_style.model.0.bias', 'affine.bias')].detach()).requires_grad_(tensor.requires_grad)
            else:
                assert (name in src_tensors) or (not require_all), f"{name} is missing among source tensors. Set require_all=False to suppress."
                if name in src_tensors:
                    try:
                        tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
                    except:
                        print('Couldnt copy', name)
                        raise
    elif src is None:
        assert dst is None or not require_all, f"Hyperaparameters mismatch: {type(dst)}"
    else:
        raise TypeError(f"Wrong type: {type(src)}")

#----------------------------------------------------------------------------

def convert_params_to_buffers(module: torch.nn.Module, param_names: List[str]=None, device=None):
    if param_names is None:
        param_names = [k[0] for k in module.named_parameters()]

    # We consider that `module` can some complicated nested nn.Module
    # That's why we need to first group the parameters by their root module
    root2subparams = {r: [] for r in set([p.split('.')[0] for p in param_names])}
    for param_name in param_names:
        param_name_decomposed: List[str] = param_name.split('.')
        if len(param_name_decomposed) > 1:
            root2subparams[param_name_decomposed[0]].append('.'.join(param_name_decomposed[1:]))

    # Now we can convert 0-level parameters to buffers or recursively call
    # the routine on the submodules
    for name, subparams in root2subparams.items():
        value = getattr(module, name)
        if isinstance(value, torch.nn.Module):
            assert len(subparams) > 0, f"Wrong subparams: {name} => {subparams}"
            convert_params_to_buffers(value, subparams, device)
        elif isinstance(value, torch.nn.Parameter):
            assert len(subparams) == 0, f"Wrong subparams: {name} => {subparams}"
            delattr(module, name)
            # This line is important. The code hangs at the broadcasting stage otherwise.
            # import gc; gc.collect(); torch.cuda.empty_cache()
            module.register_buffer(name, value.data.to(device))
        elif isinstance(value, torch.Tensor):
            # TODO: it would be good to make sure that this tensor is a buffer, and not just a random tensor,
            # but I do not know how to do this since there is no `torch.nn.Buffer` class
            assert len(subparams) == 0, f"Wrong subparams: {name} => {subparams}"
            pass # Doing nothing since it is already a buffer
        else:
            raise NotImplemented(f'Uknown value type: {type(value)}')

#------------------------------------------------------------------------------------------

def disable_grads(module: torch.nn.Module, parameters_to_freeze: Set[str]):
    for param_name, param in module.named_parameters():
        if param_name in parameters_to_freeze:
            param.requires_grad_(False)

#------------------------------------------------------------------------------------------

def print_stats(*args):
    assert len(args) > 0
    prefixes = args[:-1]
    x = args[-1]
    import torch
    if x is None:
        print(*prefixes, x)
    elif isinstance(x, torch.Tensor):
        x = x.double()
        print(*prefixes, f'avg: {x.mean().item()} | std: {x.std().item()} | min: {x.min().item()} | max: {x.max().item()} | shape: {list(x.shape)}')
    elif isinstance(x, torch.nn.Module):
        p = torch.cat([p.view(-1) for p in x.parameters()]).double()
        print_stats(p, *prefixes)
    else:
        raise NotImplementedError(f"Uknown type: {type(x)}")

#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

#----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname

#----------------------------------------------------------------------------
# Print summary table of module hierarchy.

def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True, module_kwargs={}):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, module_inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            module_inputs = list(module_inputs) if isinstance(module_inputs, (tuple, list)) else [module_inputs]
            module_inputs = [t for t in module_inputs if isinstance(t, torch.Tensor)]
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, inputs=module_inputs, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs, **module_kwargs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Input Shape', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        input_shape_str = ' + '.join([str(list(t.shape)) for t in e.inputs])
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            input_shape_str if len(input_shape_str) > 0 else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-', '-']]
    row_lengths = [len(r) for r in rows]
    assert len(set(row_lengths)) == 1, f"Summary table contains rows of different lengths: {row_lengths}"

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs

#----------------------------------------------------------------------------

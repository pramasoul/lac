# lac_llm for xp (that's numpy/cupy) inference

"""
Compress using a trained model as a predictor
The name ``lac'' is meant to suggest "LLM Arithmetic Coder"
"""

r"""Provides PredictionService, and provide_prediction_service to get one"""


import numpy as np

from xpu import cpnp, got_cp, get_array_module
from xpu import GPTConfig, forward_model_dict, load_to_xp, softmax
from xpu import device_name_to_int, device_context

import logging
import tiktoken

from contextlib import nullcontext

from config import SingletonConfig
config = SingletonConfig()


def provide_model(model_name="internal", device="cpu", threads=1):
    r"""Loads a saved module, returning a dictionary of named parameters"""
    npz = np.load(model_name + ".npz")
    if device == "cpu":
        rv = npz
    elif got_cp():
        cp = cpnp() # will return a cupy reference, since we got_cp
        with cp.cuda.Device(device_name_to_int(device)):
            rv = dict((k, cp.array(v)) for k, v in npz.items())
    else:
        raise ValueError("CuPy needed to use cuda device. Can you install it? See https://docs.cupy.dev/en/stable/install.html")
    return rv


def provide_model_on_cpu(model_name):
    return provide_model(model_name, device="cpu")


################################################################
# Prediction service

class PredictionService:
    def __init__(self):
        pass

    def restart(self):
        pass

    def accept(self, tok):
        pass

    @property
    def probabilities(self):
        return None


class LLMPredictionService(PredictionService):
    # Init with an LLMPredictor and a starting idx
    # .probabilities gets the (unnormalized) probability array for the possible token values
    # accept(tok) to update the history, before getting the new .probabilities given the new tok

    def __init__(self, llm_predictor, idx=[[198]]):
        self.p = llm_predictor
        self.xp = self.p.xp
        self.idx = self.xp.asarray(idx)
        self._temp_idx = None

    def accept(self, tok):
        assert tok < self.p.model_config.vocab_size
        #if self.idx.size(1) < self.p.model_config.block_size:
        if self.idx.size < self.p.model_config.block_size:
            #self.idx = torch.cat((self.idx, torch.tensor([[tok]]).to(self.idx.device)), 1)
            self.idx = self.xp.concatenate((self.idx, self.xp.array([[tok]])), axis=-1)
        else:
            if self._temp_idx is None:
                #self._temp_idx = self.idx.clone()


                #self._temp_idx = self.xp.asarray(self.idx, copy=True)
                self._temp_idx = self.xp.array(self.idx, copy=True)


            t = self._temp_idx
            # Shift all elements down, losing the first
            t[..., :-1] = self.idx[..., 1:]
            # Replace the last element with the new token
            t[..., -1] = tok
            #self.idx.copy_(t)   # FIXME: Should ping-pong between them instead?
            self.xp.copyto(self.idx, t)   # FIXME: Should ping-pong between them instead?

    @property
    def probabilities(self):
        probs = self.p(self.idx)
        #self.size = probs.size(0)
        self.size = probs.size
        return probs


class LLMPredictor:
    # Init with a ModelOnDeviceCache and model name, device and temperature on creation
    # Callable, returning probabilities using the MODC to transiently obtain the model
    def __init__(self, model_name, device, temperature=1.0):
        self.model_name = model_name
        # self.device = torch.device(device)
        self.device = device
        self.temperature = temperature
        self._model_config = None
        self.model_dict = provide_model(device=self.device)

        # Find array module to use
        self.xp = cpnp(*self.model_dict.values())

        self.ctx = device_context(*self.model_dict.values())

        logging.debug(f"{self.ctx=}")


    def __call__(self, idx):
        with self.ctx:
            logits = forward_model_dict(self.model_dict, self.xp.asarray(idx))

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.temperature
            if hasattr(config, "mangle_logits"):
                logits = config.mangle_logits(logits)

            # apply softmax to convert logits to (normalized) probabilities
            probabilities = softmax(logits, axis=-1)[-1]
        return probabilities  #.cpu()

    @property
    def model_config(self):
        if self._model_config is None:
            self._model_config = GPTConfig()
        return self._model_config


enc = tiktoken.get_encoding("gpt2")

def provide_prediction_service(model_name, device, threads=1, temperature=1.0) -> LLMPredictionService:
    logging.debug(f"provide_prediction_service({model_name}, {device}, {threads=}, {temperature=})")
    predictor = LLMPredictor(model_name, device, temperature)
    return LLMPredictionService(predictor)


################################################################
# Surgical tools

if False:
#if True:
    
    # torch.nn.Module hooks

    # Capture the input to modules
    # Installed with module.register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle
    # The hook will be called every time before :func:`forward` is invoked.
    # It should have the following signature::

    #     hook(module, input) -> None or modified input

    # The input contains only the positional arguments given to the module.
    # Keyword arguments won't be passed to the hooks and only to the ``forward``.
    # The hook can modify the input. User can either return a tuple or a
    # single modified value in the hook. We will wrap the value into a tuple
    # if a single value is returned(unless that value is already a tuple).


    # To include a name, this gets wrapped in a lambda at hook time:
    def record_module_input(name, module_input):
        """Record the module's input"""
        if hasattr(config, "model_record") \
           and isinstance(config.model_record, dict) \
           and 'input' in config.model_record \
           and isinstance(config.model_record['input'], list):
            config.model_record['input'].append((name, module_input))
        return None             # Don't modify the inputs for the module


    # Capture the output of modules
    # Installed with module.register_forward_hook(hook, *, prepend=False, with_kwargs=False, always_call=False)
    # Called as hook(module, args, output) -> None or modified output, if installed with_kwargs=False
    # Called as hook(module, args, kwargs, output) -> None or modified output, if installed with_kwargs=True

    # To include a name, this gets wrapped in a lambda at hook time:
    def record_module_output(name, output):
        """Record the module's output"""
        if isinstance(output, tuple):
            output = output[0] # HACK to get the logits free of the (logits, loss) tuple
        if hasattr(config, "model_record") \
           and isinstance(config.model_record, dict) \
           and 'output' in config.model_record \
           and isinstance(config.model_record['output'], list):

            if False: # Paranoid behavior
                import copy
                import numpy as np
                import cupy as cp
                if isinstance(output, Tensor):
                    output = output.cpu()
                if hasattr(output, 'get'): # cupy.array has this method, which gets an np.array
                    output = output.get()
                output = copy.deepcopy(output+0)

            config.model_record['output'].append((name, output)) # Paranoid or not, append the result

    def hook_model_for_recording(model):
        record_handles = {}
        for name, module in model.named_modules():
            pre_f = lambda module, module_input, name=name: record_module_input(name, module_input)
            pre_handle = module.register_forward_pre_hook(pre_f)
            post_f = lambda module, args, output, name=name: record_module_output(name, output)
            post_handle = module.register_forward_hook(post_f)            
            record_handles[name] = pre_handle, post_handle
        return record_handles


    def quantize_module(module, n_bits):
        """Quantize a module (which can be an entire model)"""
        def quantize_param(parameter):
            if parameter is not None:
                round_to_bits_(parameter, n_bits - parameter.abs().max().log2().ceil().int().item())

        def quantize_output(module, args, output):
            # A forward hook function, viz.
            # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
            match output:
                case torch.Tensor() as t:
                    round_to_bits_(t, n_bits) # Round in-place
                case (t, _) if isinstance(t, torch.Tensor):
                    round_to_bits_(t, n_bits) # Round in-place
            return output

        quantize_handles = {}
        for name, module in module.named_modules():
            if hasattr(module, "weight"):
                quantize_param(module.weight)
            if hasattr(module, "bias"):
                quantize_param(module.bias)
            # Hook in the quantization of the module's output
            quantize_handles[name] = module.register_forward_hook(quantize_output)
        return quantize_handles


    # Chat4, modified:
    def view_as_int32(t: torch.Tensor) -> torch.Tensor:
        """Returns a tensor of type int32 which is a view of the given tensor"""
        shape = t.shape
        device = t.device
        as_np_uint8 = t.flatten().view(torch.uint8).to('cpu').numpy()
        as_np_int32 = np.frombuffer(as_np_uint8,dtype=np.int32)
        as_torch_int32 = torch.from_numpy(as_np_int32)
        shaped = as_torch_int32.reshape(shape)
        return shaped.to(device)


    # Chat4 guided by me:
    def bitwise_or_reduce(tensor):
        # Ensure the tensor is <strike>on the GPU and </strike>flattened
        tensor = tensor.flatten()#.to('cuda')

        # Calculate the pad size to the next power of two
        current_size = tensor.numel()
        next_pow2 = 1 << current_size.bit_length()
        pad_size = next_pow2 - current_size

        # Pad the tensor if necessary
        if pad_size > 0:
            tensor = torch.cat([tensor, torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)])

        # Perform the successive folding
        while tensor.numel() > 1:
            half = tensor.numel() // 2
            left, right = tensor[:half], tensor[half:half*2]
            torch.bitwise_or(left, right, out=left)
            tensor = left  # Reuse the left half for the next iteration

        return tensor


    def round_to_bits(t: torch.Tensor, n: int):
        return (t*(1<<n)).round() * 1.0/(1<<n)


    def round_to_bits_(t: torch.Tensor, n: int):
        t *= 1<<n
        torch.round_(t)
        t *= 1.0/(1<<n)
        return t


    import torch.nn.modules.activation
    import inspect

    activation_classes = {cls_name: cls_obj for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation, inspect.isclass)}

    def is_activation_class(obj):
        return obj.__class__ in activation_classes.values()


if False: # OBSOLETE

    class Quactivation(nn.Module):
        def __init__(self, activation, n_bits) -> None:
            super().__init__()
            self.activation = activation
            self.n_bits = n_bits

        def forward(self, x: Tensor) -> Tensor:
            return round_to_bits_(self.activation(x), self.n_bits)

        def extra_repr(self) -> str:
            #return super().extra_repr() + "round_to_bits={}".format(self.n_bits)
            return "round_to_bits={}".format(self.n_bits)

    def quantize_module(module, n_bits):
        children = list(module.named_children())

        for name, child in children:
            #print(name, child)
            if isinstance(child, Quactivation):
                # Already wrapped, skip
                continue
            if hasattr(child, "weight"):
                round_to_bits_(child.weight, n_bits - child.weight.abs().max().log2().ceil().int().item())
                #print('   '.join(binstack(child.weight.flatten()[:3])))
            elif type(child) in activation_classes.values():
                # Wrap the activation function
                wrapped_activation = Quactivation(child, n_bits)
                setattr(module, name, wrapped_activation)
            else:
                # Recursively apply to child modules
                quantize_module(child, n_bits)




################################################################
# Visualization tools

if False:
    from typing import Iterable

    def about(in_x):
        x = in_x.to(torch.float64)
        print(f"{x.shape}, {in_x.dtype}, ({x.max()}, {x.mean()}, {x.min()}), {(x*x).mean(dtype=torch.float32):.3e}, {x.device}")

    def f32_delimit_single(s:str) -> str:
        return s[0] + '|' + s[1:9] + '|' + s[9:]

    def f32_delimit(arg: str | Iterable[str]) -> str | Iterable[str]:
        match arg:
            case str():
                rv =  f32_delimit_single(arg)
            case _ if isinstance(arg, Iterable) and all(isinstance(item, str) for item in arg):
                rv = [f32_delimit_single(s) for s in arg]
            case _:
                raise ValueError("Expected str or Iterable(str), got", type(arg))
        return rv

    def binstack(t: torch.Tensor) -> str:
        """Bitwise representation of a tensor"""
        t_uint8 = t.flatten().view(torch.uint8)
        t_bs = [format(x.item(), '08b') for x in t_uint8]
        n = t.element_size()
        # Assuming little-endian:
        bit_strs = [''.join(reversed(t_bs[i:i+n])) for i in range(0, len(t_bs), n)]
        if n == 4:
            bit_strs = f32_delimit(bit_strs)
        return bit_strs

    def float_to_binary(v):
        return binstack(torch.tensor(v))[0]

    def or_each(tensor_iter) -> Tensor:
        """"""
        return torch.tensor([bitwise_or_reduce(view_as_int32(p)) for p in iter(tensor_iter)],
                            dtype=torch.int32)

    def or_all(tensor_iter):
        per_param = [bitwise_or_reduce(view_as_int32(p)) for p in iter(tensor_iter)]
        return bitwise_or_reduce(torch.tensor(per_param, dtype=torch.int32))

    def binor_each(t_iter):
        return binstack(or_each(iter(t_iter)))

    def binor(t: Tensor):
        return binor_each([t])

    def binor_all(t_iter):
        return binstack(or_all(iter(t_iter)))

    def binor_model(model):
        return f32_delimit(binstack(or_all(model.parameters()))[0])

    def print_model_components(model):
        for name, module in model.named_modules():
            print([" ","A"][is_activation_class(module)], \
                  # [" ","N"][isinstance(module, LayerNorm)], \
                  [" ","W"][hasattr(module, "weight")], \
                  hasattr(module, "weight") and module.weight is not None and "%24s" % (str(module.weight.shape)) or " "*24, \
                  [" ","B"][hasattr(module, "bias")], \
                  hasattr(module, "bias") and module.bias is not None and "%22s" % (str(module.bias.shape)) or " "*22, \
                  name)

    def compare_records(record_a, record_b, device='cpu'):
        assert all(c[0]==g[0] for c,g in zip(record_a, record_b))
        assert all(c[1].dtype==g[1].dtype for c,g in zip(record_a[:-1], record_b[:-1]))
        record_names = [v[0] for v in record_a]
        close_list = [torch.allclose(c[1].to(device), g[1].to(device), atol=1e-09) for c,g in zip(record_a[:-1], record_b[:-1])]
        close_dict = dict((name, (i, close)) for i, (name, close) in enumerate(zip(record_names, close_list)))
        for name, (i, close) in close_dict.items():
            if True or not close:
                print(name)
                about(record_b[i][1].to(device) - record_a[i][1].to(device))

    def view_record(model_record):
        for name, t in model_record:
            if hasattr(t, "dtype") and t.dtype == torch.float32:
                #print(f32_delimit(binstack(bitwise_or_reduce(view_as_int32(t)))[0]), name)
                print(binor(t), name)
            else:
                print(f"{name} is {t}")

from typing import Union, List, Tuple, Optional
from dataclasses import dataclass
from scipy.special import erf

from contextlib import nullcontext
import itertools

import numpy as np
try:
    import cupy as cp
    from cupy import get_array_module
except ImportError:
    def get_array_module(*args, **kwargs):
        return np

import sys

def got_cp():
    return 'cupy' in sys.modules

def cpnp(*args, **kwargs):
    if args or kwargs:
        return get_array_module(*args, **kwargs)
    return got_cp() and cp or np


NDArray = type(cpnp().ndarray)

def device_name_to_int(device_name: str) -> int:
    if not device_name.lower().startswith("cuda"):
        raise ValueError("unrecognized device name " + cuda_name)
    if device_name.lower() == "cuda":
        cuda_n = 0
    else:
        cuda_n = int(device_name.split(':')[1])
    return cuda_n


def device_context(*args, **kwargs):
    # Find appropriate context manager
    ctx = nullcontext()
    if got_cp():
        if cpnp(*args, **kwargs) == cpnp():
            devices = [v.device for v in itertools.chain(args, kwargs.values()) if hasattr(v, 'device')]
            if not all(d == devices[0] for d in devices):
                raise ValueError(f"These are spread across multiple devices: {devices}")
            ctx = cpnp().cuda.Device(devices[0])
    return ctx


def to_cpu(a):
    if hasattr(a, 'cpu'):
        return a.cpu()
    if hasattr(a, 'get'):
        return a.get()


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster



# def softmax(x):
#     max_x = xp.max(x)
#     return xp.exp(x - max_x) / xp.exp(x - max_x).sum()

def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x along the specified axis."""
    # Note: Use xp to abstract over NumPy and CuPy
    xp = cpnp(x)
    e_x = xp.exp(x - xp.max(x, axis=axis, keepdims=True))
    return e_x / xp.sum(e_x, axis=axis, keepdims=True)



def gelu(x: NDArray) -> NDArray:
    r"""Applies the Gaussian Error Linear Units function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> gelu(xp.linspace(-3, 3, 10))
        array([-0.00404969, -0.02290243, -0.07965059, -0.15865525, -0.12314711,
                0.21018622,  0.84134475,  1.58701608,  2.3104309 ,  2.99595031])
    """
    return (x * (1 + erf(x/(2**0.5))))/2

def layer_norm(
        x: NDArray,
        #normalized_shape: List[int],
        weight: Optional[NDArray] = None,
        bias: Optional[NDArray] = None,
        eps: float = 1e-5,
) -> NDArray:
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> t = xp.linspace(-3, 3, 3*4).reshape(3,4)
        >>> layer_norm(t)
        array([[-1.34162275, -0.44720758,  0.44720758,  1.34162275],
               [-1.34162275, -0.44720758,  0.44720758,  1.34162275],
               [-1.34162275, -0.44720758,  0.44720758,  1.34162275]])

    """
    xp = cpnp(x)
    x_mean, x_var = x.mean(axis=-1, keepdims=True), x.var(axis=-1, keepdims=True)
    lnorm = (x - x_mean) / xp.sqrt(x_var + eps)
    y = lnorm
    if weight is not None:
        y *= weight
    if bias is not None:
        y += bias
    return y


def savez_torch_nn_parameters(str_or_file, module, compressed=False):
    r"""Saves the named parameters of a torch.nn.Module to a file.

    A pytorch model typically inherits from nn.Module, so this will
    save all the parameters of a well-constructed model, in the order
    they were registered.

    It records nothing about how those modules are combined. 

    """
    if compressed:
        save_fun = np.savez_compressed
    else:
        save_fun = np.savez
    save_fun(str_or_file, **dict(module.named_parameters()))


def load_to_xp(filename):
    r"""Loads a saved module, returning a dictionary of named parameters"""
    npz = np.load(filename)
    return dict((k, cpnp().array(v)) for k, v in npz.items())


def forward_model_dict(d: dict, idx: NDArray, config=GPTConfig(), recorder=None, x=None):
    r"""Evaluate a GPT model
    """
    if recorder is None:
        rec = lambda name, v: None
    else:
        rec = recorder
    n_embd = config.n_embd
    n_head = config.n_head
    rec('model', idx)

    xp = cpnp(*d.values())

    with device_context(*d.values()):
        block_size = config.block_size
        lt_ones = xp.tril(xp.ones((block_size, block_size))).reshape(1, 1, block_size, block_size) # For attention mask generation
        for name, array in d.items():
            name_els = name.split('.')
            assert name_els.pop(0) == 'transformer'
            assert name_els.pop() == 'weight'
            match name_els.pop(0):
                case 'wte':
                    # print("got wte", name_els, array.shape)
                    tok_emb = array[idx]
                    x = tok_emb

                case 'wpe':
                    # print("got wpe", name_els, array.shape)
                    assert array.shape[0] == block_size
                    # print(f"{block_size=}")
                    b, t = idx.shape # dimensions of batch, tokens
                    assert t <= block_size, f"Cannot forward sequence of length {t}, block size is only {block_size}"
                    pos = xp.arange(0, t, dtype=xp.uint32) # shape (t)
                    pos_emb = array[pos]
                    #x = tok_emb + pos_emb
                    x = x + pos_emb

                case 'h':

                    # Block is residual connection:
                    # def forward(self, x):
                    #     x = x + self.attn(self.ln_1(x))
                    #     x = x + self.mlp(self.ln_2(x))
                    #     return x

                    block = int(name_els.pop(0))
                    layer = name_els.pop(0)
                    # print(f"{block=} {layer=}", end=": ")
                    match layer:
                        case _ if layer.startswith('ln_'):
                            pre_ln_x = x
                            x = layer_norm(x, weight=array)

                        case 'attn':

                            B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
                            # print(f"{B=}, {T=}, {C=}")
                            match name_els.pop(0):

                                case 'c_attn':
                                    # print(" attention", name_els, array.shape)
                                    assert config.n_embd % config.n_head == 0

                                    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
                                    qkv = xp.matmul(x, array.T)
                                    q, k, v = xp.split(qkv,
                                                       [n_embd, 2 * n_embd],
                                                       axis=2)
                                    k = k.reshape(B, T, n_head, C // n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs) 
                                    q = q.reshape(B, T, n_head, C // n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs) 
                                    v = v.reshape(B, T, n_head, C // n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs) 

                                    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
                                    att = xp.matmul(q, k.transpose(0, 1, 3, 2)) * (1.0 / xp.sqrt(k.shape[-1], dtype=x.dtype))
                                    # Manually broadcast the mask to match the shape of `att`
                                    mask = (lt_ones[:,:,:T,:T] == 0)
                                    broadcasted_mask = xp.broadcast_to(mask, att.shape)
                                    att[broadcasted_mask] = xp.full((), -xp.inf, dtype=att.dtype)
                                    att = softmax(att, axis=-1)
                                    # print(f"{att.shape=}")
                                    x = xp.matmul(att, v) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

                                    # Transpose y from (B, n_head, T, C // n_head) to (B, T, n_head, C // n_head)
                                    x = x.transpose(0, 2, 1, 3)

                                    # Reshape x to re-assemble all head outputs side by side -> (B, T, C)
                                    x = x.reshape(B, T, C)

                                case 'c_proj':
                                    # print(" projection", name_els, array.shape)
                                    x = xp.matmul(x, array.T)

                                    # x is the residual; add to it the straight path as saved by the layer norm step
                                    x += pre_ln_x

                        case 'mlp':
                            # print(" multi-layer perceptron", name_els, array.shape)
                            match name_els.pop(0):
                                case 'c_fc':
                                    x = xp.matmul(x, array.T)
                                    x = gelu(x)

                                case 'c_proj':
                                    x = xp.matmul(x, array.T)
                                    # x is the residual; add to it the straight path as saved by the layer norm step
                                    x += pre_ln_x

                case 'ln_f':
                    # print("final layer norm", name_els, array.shape)
                    x = layer_norm(x, weight=array)
                    x = xp.matmul(x[:, [-1], :], d['transformer.wte.weight'].T)

            rec(name, x)

    return x

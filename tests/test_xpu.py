# Test xpu utilities

import pytest
import torch
import ctypes
import json
import logging
import subprocess
import tempfile

from binascii import hexlify, unhexlify
from contextlib import contextmanager
from io import BytesIO
from itertools import chain
from typing import Callable, List

import cupy as cp
import numpy as np

# import unittest.mock as mock
from unittest.mock import mock_open

# import torch
# import torch.nn.functional as F
from lac_llm import *

import xpu
from xpu import NDArray
from xpu import cpnp


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)  # StreamHandler logs to console


# Now you can use logging in your tests
def test_example():
    logging.debug("This is a debug message.")


def test_create_file(tmp_path):
    CONTENT = "content"
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text(CONTENT, encoding="utf-8")
    assert p.read_text(encoding="utf-8") == CONTENT
    assert len(list(tmp_path.iterdir())) == 1


# A tmp filename maker operating outside the scope rules
@contextmanager
def tmp_filename():
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    try:
        yield filename
    finally:
        os.unlink(filename)


def test_tmp_filename():
    with tmp_filename() as fname:
        logging.debug("tmp_filename provided %s", fname)
    # FIXME: test it


@contextmanager
def mock_file(initial_data=None):
    """A context manager for mocking file operations using BytesIO."""
    # Create a BytesIO object with optional initial data
    if initial_data is not None:
        if isinstance(initial_data, str):
            initial_data = initial_data.encode()  # Convert string to bytes
        file_obj = BytesIO(initial_data)
    else:
        file_obj = BytesIO()

    try:
        yield file_obj
    finally:
        # Perform any necessary cleanup (if any)
        file_obj.close()


def test_mock_file():
    # Usage example
    with mock_file("Initial text") as mock_file_obj:
        # Perform file operations
        mock_file_obj.write(b" More data")
        mock_file_obj.seek(0)
        content = mock_file_obj.read()
        assert content.decode() == " More dataxt"


def test_read_file(mocker):
    mock_file_contents = "mock file data"
    mocker.patch("builtins.open", mock_open(read_data=mock_file_contents))

    with open("mock_file.txt", "r") as file:
        data = file.read()

    assert data == mock_file_contents


def test_write_file(mocker):
    mock_write = mock_open()
    mocker.patch("builtins.open", mock_write)

    data_to_write = "data to be written"
    with open("mock_file.txt", "w") as file:
        file.write(data_to_write)

    mock_write.assert_called_once_with("mock_file.txt", "w")
    mock_write().write.assert_called_once_with(data_to_write)


def test_multiple_file_mocks(mocker):
    # Define the mock file contents for different files
    file_mocks = {
        "foo": mock_open(read_data="data from foo").return_value,
        "bar": mock_open(read_data="data from bar").return_value,
    }

    # Side effect function to return different mocks based on filename
    def side_effect(filename, *args, **kwargs):
        return file_mocks[filename]

    # Patch 'open' with the side effect
    mocker.patch("builtins.open", side_effect=side_effect)

    # Test reading from 'foo'
    with open("foo", "r") as file:
        data = file.read()
    assert data == "data from foo"

    # Test reading from 'bar'
    with open("bar", "r") as file:
        data = file.read()
    assert data == "data from bar"


def my_function():
    # Example function that reads from one file and writes to another
    with open("read_file.txt", "r") as f:
        logging.debug(f"my_function read_file.txt is {f}")
        data = f.read()

    logging.debug(f"my_function data is {data}")

    with open("write_file.txt", "w") as f:
        logging.debug(f"my_function write_file.txt is {f}")
        f.write(data[::-1])  # Just an example transformation


def test_my_function(mocker):
    mock_file_handles = {
        ("read_file.txt", "r"): mock_open(read_data="Hello World").return_value,
        ("write_file.txt", "w"): mock_open().return_value,
    }

    def side_effect(filename, mode):
        rv = mock_file_handles[(filename, mode)]
        logging.debug(f"side_effect({filename}, {mode}) returning {rv}")
        return rv

    mocker.patch("builtins.open", side_effect=side_effect, create=True)

    my_function()

    # Check if the read file was opened correctly
    logging.debug(f"{mock_file_handles[('read_file.txt', 'r')]=}")
    mock_file_handles[("read_file.txt", "r")].read.assert_called_once_with()

    # Add any other assertions you need for the write file
    mock_file_handles[("write_file.txt", "w")].write.assert_called_once_with(
        "dlroW olleH"
    )


# @pytest.fixture(scope="session")
# def brief_text_file(tmp_path):
#     return tmp_file


################
# xpu test fixtures

# @pytest.fixture
# def xp():
#     return xpu.xp

# @pytest.fixture
# def config():
#     return xpu.GPTConfig

@pytest.fixture(scope="session")
def medium_text():
    return r"""A variety of specific techniques for arithmetic coding have
historically been covered by US patents, although various well-known
methods have since passed into the public domain as the patents have
expired. Techniques covered by patents may be essential for
implementing the algorithms for arithmetic coding that are specified
in some formal international standards. When this is the case, such
patents are generally available for licensing under what is called
"reasonable and non-discriminatory" (RAND) licensing terms (at least
as a matter of standards-committee policy). In some well-known
instances, (including some involving IBM patents that have since
expired), such licenses were available for free, and in other
instances, licensing fees have been required. The availability of
licenses under RAND terms does not necessarily satisfy everyone who
might want to use the technology, as what may seem "reasonable" for a
company preparing a proprietary commercial software product may seem
much less reasonable for a free software or open source project."""

@pytest.fixture(scope="session")
def brief_text():
    return r"""You will rejoice to hear that no disaster has accompanied the
commencement of an enterprise which you have regarded with such evil
forebodings. I arrived here yesterday, and my first task is to assure
my dear sister of my welfare and increasing confidence in the success
of my undertaking.
"""

@pytest.fixture(scope="session")
def brief_toks(brief_text):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(brief_text)

def test_brief_toks(brief_toks):
    assert brief_toks[-1] == 198
    assert len(brief_toks) == 62
    

@pytest.fixture(scope="session")
def reference_model():
    make_torch_deterministic()
    model_name = 'internal'
    #model_name = 'gpt2'
    reference_model = provide_model_on_cpu(model_name)
    return reference_model

@pytest.fixture(scope="module")
def model(reference_model):
    from copy import deepcopy
    model = deepcopy(reference_model)
    return model

def test_model(model):
    assert isinstance(model, nn.Module)

#idx = torch.tensor([toks], dtype=torch.int64)

@pytest.fixture
def miked_model_and_handles(reference_model):
    from copy import deepcopy
    model = deepcopy(reference_model)
    record_handles = hook_model_for_recording(model)
    config.model_record = {'input': [], 'output': []}
    return model, record_handles


################
# Save and load

@pytest.fixture(scope="module")
def savez_loaded_to_xp_dict(model): # We don't need the miked model, just the named parameters
    with tmp_filename() as fname:
        xpu.savez_torch_nn_parameters(fname, model)
        md = xpu.load_to_xp(fname + ".npz")
    return md


def test_savez_loaded_to_xp_dict(savez_loaded_to_xp_dict, model):
    md = savez_loaded_to_xp_dict
    assert isinstance(md, dict)
    assert len(list(md.keys())) == 75
    assert all(k.startswith('transformer.') for k in md.keys())
    assert all(isinstance(v, cpnp().ndarray) for k, v in md.items())
    for name, t in model.named_parameters():
        assert (md[name] == t2x(t)).all(), name


################
# torch <-> xp conversions

def x2t(x):
    xp = cpnp()
    if xp == cp:
        x = xp.asnumpy(x)
    else:
        x = xp.array(x)
    return torch.from_numpy(x)

def test_x2t():
    a = cpnp().arange(3)
    t = x2t(a)
    assert all(t == torch.arange(3))


def t2x(t):
    return xpu.cpnp().asarray(t)

def test_t2x():
    t = torch.arange(3)
    assert cpnp().all(t2x(t) == cpnp().arange(3))

def xify(f):
    return lambda x: t2x(f(x2t(x)))

def test_xify():
    ft_arange = xify(torch.arange)
    assert cpnp().all(ft_arange(3) == cpnp().arange(3))
    

################
# Model inference component tests

def test_NDArray_exists():
    assert type(NDArray) in (type(np.ndarray), type(cp.ndarray))


def test_GPTConfig():
    assert xpu.GPTConfig.vocab_size == 50304
    assert xpu.GPTConfig.dropout == 0.0


def test_softmax():
    assert cpnp().all(xpu.softmax(cpnp().ones(3)) == cpnp().ones(3)/3)
    x = cpnp().arange(3, dtype=cpnp().float32)
    assert cpnp().allclose(xpu.softmax(x), t2x(F.softmax(x2t(x), dim=0)))


def test_gelu():
    x = cpnp().linspace(-3, 3)
    y = xpu.gelu(x)
    assert cpnp().allclose(y, xify(F.gelu)(x))


def test_layer_norm():
    x = cpnp().linspace(-2, 4)
    y = xpu.layer_norm(x)
    # torch.layer_norm(input, normalized_shape, weight, bias, eps)
    t_lyn = lambda x: t2x(F.layer_norm(x2t(x), x.shape))
    assert cpnp().allclose(y, t_lyn(x))
    w = cpnp().linspace(1, 2)
    yw = xpu.layer_norm(x, w)
    t_lynw = lambda x: t2x(F.layer_norm(x2t(x), x.shape, x2t(w)))
    assert cpnp().allclose(yw, t_lynw(x))    



################
# ***


################
# Inference correspondence

def logits_and_recording_torch(model, idx):
    idx_torch = torch.tensor([idx], dtype=torch.int64)
    logits, loss = model(idx_torch)
    rv = logits, config.model_record
    config.model_record = None
    return rv


def test_logits_and_recording_torch(miked_model_and_handles, brief_toks):
    model, handles = miked_model_and_handles
    logits, torch_model_record = logits_and_recording_torch(model, brief_toks)
    assert logits.shape == (1,1,50304)
    assert len(torch_model_record['input'])  == 150
    assert len(torch_model_record['output']) == 150
    assert torch_model_record['input'][1][0] == 'transformer.wte'
    tmrd = dict(torch_model_record['input'])
    assert 'lm_head' in tmrd.keys()


def logits_and_recording_xp(md, brief_toks):
    idx_xp = cpnp().asarray([brief_toks]) # one batch
    logging.debug("idx_xp %r", idx_xp)
    config.model_record = {'output': []}
    y_xp = xpu.forward_model_dict(md, idx_xp, recorder=record_module_output)
    xp_model_record = config.model_record['output']
    config.model_record = None
    return y_xp, xp_model_record


def test_logits_and_recording_xp(savez_loaded_to_xp_dict, brief_toks):
    md = savez_loaded_to_xp_dict
    logits, recording = logits_and_recording_xp(md, brief_toks)
    assert type(recording[0][1]) == cpnp().ndarray
    xmrd = dict(recording)
    assert 'transformer.wte.weight' in xmrd.keys()


@pytest.fixture
def trid_trod_xrd(miked_model_and_handles, savez_loaded_to_xp_dict, brief_toks):
    # Get the torch model, miked for recording, and run it
    model, handles = miked_model_and_handles
    torch_logits, torch_record = logits_and_recording_torch(model, brief_toks)

    # Get the xp model as loaded from torch save, and run it with recording
    md = savez_loaded_to_xp_dict
    xp_logits, xp_record = logits_and_recording_xp(md, brief_toks)

    # Make convenient dictionaries of both
    trid = dict((v[0], v[1][0]) for v in torch_record['input'])
    trod = dict(torch_record['output'])
    xrd = dict((k.removesuffix('.weight'), v) for k,v in xp_record)
    
    return trid, trod, xrd


@pytest.fixture
def trd_xrd(trid_trod_xrd):
    trid, trod, xrd = trid_trod_xrd
    return trod, xrd


################
# Check the dimensions and data types of the models and the recordings

def test_shapes_of_models_match(savez_loaded_to_xp_dict, model):
    md = savez_loaded_to_xp_dict
    for name, t in model.named_parameters():
        assert (md[name] == t2x(t)).all(), name
    

def test_shape_of_xp_model_matches_recording(trd_xrd, savez_loaded_to_xp_dict, model):
    trd, xrd = trd_xrd
    md = savez_loaded_to_xp_dict
    model_rec_shapes_mismatch = dict((k, (md[k].shape, xrd[k].shape)) for k in md.keys() if k in xrd.keys() and md[k].shape != xrd[k].shape)
    assert not model_rec_shapes_mismatch # FIXME: Is this really what we should expect?


expected_shape_mismatches = set(f'transformer.h.{h}.attn.c_attn' for h in range(12)) | set(['transformer.wpe', 'transformer.ln_f'])

def test_shapes_of_recordings_match(trd_xrd, savez_loaded_to_xp_dict, model): # models included for pdb debugging
    trd, xrd = trd_xrd
    md = savez_loaded_to_xp_dict
    recordings_shape_mismatch = dict((k, (trd[k].shape, xrd[k].shape)) for k in xrd.keys() if k in trd.keys() and t2x(trd[k]).shape != xrd[k].shape)
    assert set(recordings_shape_mismatch.keys()) == expected_shape_mismatches
    #assert not recordings_shape_mismatch

def test_dtypes_of_recordings(trd_xrd):
    trd, xrd = trd_xrd
    # Check the data types in the recordings are consistent as expected
    assert all(v.dtype == torch.float32 for v in trd.values() if type(v) is torch.Tensor)
    assert all(v.dtype == cpnp().float32 for k, v in xrd.items() if k != 'model') # The 'model' entry records the tokens to the model forward
    assert xrd['model'].dtype == cpnp().int64 # The 'model' entry records the tokens to the model forward


if False:
    # For reference, the LayerNorm class:
    class LayerNorm(nn.Module):
        """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

        def __init__(self, ndim, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

        def forward(self, input):
            return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


    # For reference, excerpt from the self-attention module:
    class CausalSelfAttention(nn.Module):
        "[...]"
        def forward(self, x):
            "[...]"
            y = (
                y.transpose(1, 2).contiguous().view(B, T, C)
            )  # re-assemble all head outputs side by side

            # output projection
            y = self.resid_dropout(self.c_proj(y))
            return y


    # For reference, the definition of the 12x repeated transformer block in the GPT model
    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


def test_transformer_h_0_ln_2(trid_trod_xrd, savez_loaded_to_xp_dict, model):
    # What's up with the calculation of the first Block's second LayerNorm?
    trid, trd, xrd = trid_trod_xrd

    md = savez_loaded_to_xp_dict # The dictionary of named parameters in cpnp().ndarray form

    # for DEBUG convenience:
    ta, tb = t2x(trd['transformer.h.0.attn']), t2x(trd['transformer.h.0.ln_2'])
    na, nb = xrd['transformer.h.0.attn.c_proj'], xrd['transformer.h.0.ln_2']

    if not cpnp().allclose(xrd['transformer.h.0.ln_2'], t2x(trd['transformer.h.0.ln_2']), atol=1e-4, rtol=1e-4):
        logging.debug("Mean squared difference of %s is %f" %
                      ('transformer.h.0.ln_2',
                       ((xrd['transformer.h.0.ln_2'] - t2x(trd['transformer.h.0.ln_2']))**2).mean()))

        # Reproduce the layer norm calculation:

        # Get the input ("x") to the layer norm from the output of the preceeding stage in the recordings
        x = xrd['transformer.h.0.attn.c_proj'] # From the xp recording, the c_proj result is input to ln_2
        putative_t_x = trd['transformer.h.0.attn.c_proj'] # From the torch recording, for comparison
        assert cpnp().allclose(x, t2x(putative_t_x), rtol=1e-06, atol=1e-6) # which should match +-
        t_x = trid['transformer.h.0.ln_2'] # The recorded input to the torch ln_2
        assert (putative_t_x == t_x).all()

        # Get the weight vector ("w") from the respective models
        w = md['transformer.h.0.ln_2.weight'] # From the xp model, the weight vector to use
        t_w = model.transformer.h[0].ln_2.weight # From the torch model
        # which should match the torch model exactly, as there should be no loss in going from torch to numpy
        assert cpnp().all(w == t2x(t_w)) # xp way
        assert (x2t(w) == t_w).all()  # torch way

        assert x.shape[-1] == w.shape[-1] # and which must match the shape of what it's scaling

        # Compare xpu and torch's layer norm calculation from their respective inputs
        xw = xpu.layer_norm(x, w)   # The layer norm calculation as we do it
        t_xw = F.layer_norm(t_x, t_w.shape, t_w, None, 1e-5)  # Layer norm as pytorch calculates it from its input
        assert cpnp().allclose(xw, t2x(t_xw), rtol=1e-04, atol=1e-4) # Expect reasonable correspondence

        # Compare our and torch's layer norm calculation from the same input
        t_xpxw = F.layer_norm(x2t(x), t_w.shape, x2t(w), None, 1e-5)  # Layer norm as pytorch calculates it from xp's inputs
        assert cpnp().allclose(xw, t2x(t_xpxw), rtol=1e-05, atol=1e-5) # Agrees more tightly with our calculation
        
        # Calculate using the torch ln_2 model forward
        t_h0_ln_2 = model.transformer.h[0].ln_2(t_x)
        assert (t_xw == t_h0_ln_2).all() # The F.layer_norm result from above should agree exactly

        # Compare these calculations with the results recorded during inference
        assert (xw == xrd['transformer.h.0.ln_2']).all() # The xp calculation should match the recording
        assert (t_xw == trd['transformer.h.0.ln_2']).all() # The torch calculation should match the recording

        # Compare with the torch model's calculation as recorded
        # FAILS: assert cpnp().allclose(trd['transformer.h.0.ln_2'], t_h0_ln_2)



################
# The self-attention is the most complicated translation from torch,
# so is deserving of special scrutiny


def xpu_attn(x: NDArray, md: dict, h_ix: int):
    # Run just the attention at Block h_ix
    keys = (f'transformer.h.{h_ix}.attn.c_attn.weight',
            f'transformer.h.{h_ix}.attn.c_proj.weight',
    )
    md_selection = {}
    md_selection[f'transformer.h.{h_ix}.ln_1.weight'] = cpnp().ones_like(md[f'transformer.h.{h_ix}.ln_1.weight'])
    md_selection.update((k,v) for k,v in md.items() if k in keys)

    # We use a fake ln_1 with identity weights so the forward_model_dict has the value to add the residual to

    return xpu.forward_model_dict(md_selection, cpnp().array([[198]]), x=x)


@pytest.mark.skip(reason="FIXME")
def test_xpu_attn(trd_xrd, savez_loaded_to_xp_dict):
    trd, xrd = trd_xrd
    md = savez_loaded_to_xp_dict

    # Verify the extract reproduces the recorded results
    for i in range(12):
        x = xrd[f'transformer.h.{i}.ln_1']
        y = xpu_attn(x, md, i)
        assert cpnp().all(y == xrd[f'transformer.h.{i}.attn.c_proj'])


@pytest.mark.skip(reason="FIXME")
def test_xpu_attn_v_torch(trd_xrd, savez_loaded_to_xp_dict, model):
    trd, xrd = trd_xrd
    md = savez_loaded_to_xp_dict

    # Test correspondence of xp and torch attention
    for i in range(12):
        x = xrd[f'transformer.h.{i}.ln_1']
        y = xpu_attn(x, md, i)
        t_y = model.transformer.h[i].attn(x2t(x))
        assert cpnp().allclose(y, t2x(t_y), rtol=1e-05, atol=1e-05)


################
# How do the recordings diverge? How does their separation go?

def diverg(a, b):
    if isinstance(a, torch.Tensor):
        a = t2x(a)
    if isinstance(b, torch.Tensor):
        b = t2x(b)
    return ((a-b)**2).mean()/((a**2).mean() + (b**2).mean())


def test_diverg():
    assert diverg(cpnp().arange(4), cpnp().arange(4)) == 0
    assert diverg(torch.arange(4), cpnp().arange(4)) == 0
    assert diverg(torch.arange(4), cpnp().arange(1,5)) == 1 / 11


@pytest.mark.skip(reason="FIXME")
def test_recordings_divergence(trd_xrd):
    trd, xrd = trd_xrd
    divergence = dict((k, diverg(trd[k], xrd[k])) for k in xrd.keys() if k in (xrd.keys() & trd.keys()) - expected_shape_mismatches)
    assert all(v < 0.001 for v in divergence.values())


################
# Which of the recorded values are "close enough" to be considered to match?

def closenuf(a, b):
    return cpnp().allclose(a, b, rtol=1e-04, atol=1e-04, equal_nan=True)


class Clump:
    def __init__(self, name, a):
        self.orig_a = a
        if isinstance(a, tuple):
            if isinstance(a[0], Tensor):
                a = a[0]
        if isinstance(a, Tensor):
            a = t2x(a)
            name = f"t.{name}"
        self.ref_a = a
        self.names = [name]

    def takeit(self, name, a):
        if a is None:
            return True
        if isinstance(a, Tensor):
            a = t2x(a)
            name = f"t.{name}"
        
        if isinstance(a, cpnp().ndarray) \
           and a.shape == self.ref_a.shape \
           and closenuf(a, self.ref_a):
            self.names.append(name)
            return True
        else:
            return False

    def __repr__(self):
        return "<" +  ", ".join(self.names) + ">"

    def __len__(self):
        return len(self.names)


def test_torch_xp_correspondence(trd_xrd, savez_loaded_to_xp_dict, model, brief_toks):
    # pdb helpers

    def c_match():
        print('\n'.join(str(c) for c in clumps if len(c)>1))

    def cx_match():
        print('\n'.join(str(c) for c in clumps if len(c)>1 and not c.names[0].startswith('t.')))

    def k_about(k):
        about(x2t(xrd[k]) - trd[k])

    # model, handles = miked_model_and_handles
    # torch_logits, torch_record = logits_and_recording_torch(model, brief_toks)
    # md = savez_loaded_to_xp_dict
    # trd = dict(torch_record)
    # xrd = dict((k.removesuffix('.weight'), v) for k,v in xp_record)

    trd, xrd = trd_xrd          # The recordings of torch model and xp model, in dictionary form
    md = savez_loaded_to_xp_dict # The dictionary of named parameters in cpnp().ndarray form
    idx_torch = torch.tensor([brief_toks], dtype=torch.int64)
    torch_logits, _ = model(idx_torch)
    idx_xp = cpnp().asarray([brief_toks]) # one batch
    xp_logits = xpu.forward_model_dict(md, idx_xp)
    
    clumps = []
    for kx, ax in chain(xrd.items(), trd.items()):
        taken = False
        for clump in clumps:
            if clump.takeit(kx, ax):
                taken = True
                break
        if not taken:
            clumps.append(Clump(kx, ax))


    # close = [kx for kx, ax in xrd.items() if kx in trd.keys() and closenuf(ax, t2x(trd[kx]))]

    assert closenuf(xp_logits, t2x(torch_logits))


################


if False:
    #del(config.model_record)
    cpu_model_record = config.model_record
    config.model_record = []
    y_ref, loss = model(idx)
    about(y_ref)
    y_ref
    ref_model_record = config.model_record
    config.model_record = []
    idx_xp = cpnp().array(idx)
    y_xp = forward_model_dict(md, idx_xp, recorder=record_module_output)
    xp_model_record = config.model_record
    config.model_record = []
    len(xp_model_record)











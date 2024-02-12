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
from xpu import xp


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
    config.model_record = []
    return model, record_handles


@pytest.fixture(scope="module")
def savez_loaded_to_xp_dict(model):
    with tmp_filename() as fname:
        xpu.savez_torch_nn_parameters(fname, model)
        md = xpu.load_to_xp(fname + ".npz")
    return md


################
# Save and load tests

@pytest.mark.slow
def test_savez_loaded_to_xp_dict(savez_loaded_to_xp_dict, model):
    md = savez_loaded_to_xp_dict
    assert isinstance(md, dict)
    assert len(list(md.keys())) == 75
    assert all(k.startswith('transformer.') for k in md.keys())
    assert all(isinstance(v, xp.ndarray) for k, v in md.items())
    for name, t in model.named_parameters():
        assert (md[name] == t2x(t)).all(), name


################
# torch <-> xp conversions

def x2t(x):
    return torch.from_numpy(xpu.xp.asnumpy(x))

def test_x2t():
    a = xp.arange(3)
    t = x2t(a)
    assert all(t == torch.arange(3))


def t2x(t):
    return xpu.xp.asarray(t)

def test_t2x():
    t = torch.arange(3)
    assert xp.all(t2x(t) == xp.arange(3))

def xify(f):
    return lambda x: t2x(f(x2t(x)))

def test_xify():
    ft_arange = xify(torch.arange)
    assert xp.all(ft_arange(3) == xp.arange(3))
    

################
# Model inference component tests

def test_NDArray_exists():
    assert type(NDArray) in (type(np.ndarray), type(cp.ndarray))


def test_GPTConfig():
    assert xpu.GPTConfig.vocab_size == 50304


def test_softmax():
    assert xp.all(xpu.softmax(xp.ones(3)) == xp.ones(3)/3)
    x = xp.arange(3, dtype=xp.float32)
    assert xp.allclose(xpu.softmax(x), t2x(F.softmax(x2t(x), dim=0)))


def test_gelu():
    x = xp.linspace(-3, 3)
    y = xpu.gelu(x)
    assert xp.allclose(y, xify(F.gelu)(x))


def test_layer_norm():
    x = xp.linspace(-2, 4)
    y = xpu.layer_norm(x)
    # torch.layer_norm(input, normalized_shape, weight, bias, eps)
    t_lyn = lambda x: t2x(F.layer_norm(x2t(x), x.shape))
    assert xp.allclose(y, t_lyn(x))
    w = xp.linspace(1, 2)
    yw = xpu.layer_norm(x, w)
    t_lynw = lambda x: t2x(F.layer_norm(x2t(x), x.shape, x2t(w)))
    assert xp.allclose(yw, t_lynw(x))    



################
# ***


################
# Inference correspondence

def logits_and_recording_torch(model, idx):
    idx_torch = torch.tensor([idx], dtype=torch.int64)
    config.model_record = []
    logits, loss = model(idx_torch)
    torch_model_record = config.model_record
    config.model_record = []
    return logits, torch_model_record


def test_logits_and_recording_torch(miked_model_and_handles, brief_toks):
    model, handles = miked_model_and_handles
    logits, torch_model_record = logits_and_recording_torch(model, brief_toks)
    assert logits.shape == (1,1,50304)
    assert len(torch_model_record) == 150
    assert torch_model_record[0][0] == 'transformer.wte'
    tmrd = dict(torch_model_record)
    assert 'lm_head' in tmrd.keys()


def logits_and_recording_xp(md, brief_toks):
    idx_xp = xp.asarray([brief_toks]) # one batch
    logging.debug("idx_xp %r", idx_xp)
    config.model_record = []
    y_xp = xpu.forward_model_dict(md, idx_xp, recorder=record_module_output)
    xp_model_record = config.model_record
    config.model_record = []
    return y_xp, xp_model_record


def test_logits_and_recording_xp(savez_loaded_to_xp_dict, brief_toks):
    md = savez_loaded_to_xp_dict
    logits, recording = logits_and_recording_xp(md, brief_toks)
    assert type(recording[0][1]) == xp.ndarray
    xmrd = dict(recording)
    assert 'transformer.wte.weight' in xmrd.keys()


def test_transformer_h_0_ln_2(savez_loaded_to_xp_dict, brief_toks):
    md = savez_loaded_to_xp_dict
    logits, recording = logits_and_recording_xp(md, brief_toks)
    xmrd = dict(recording)
    assert 'transformer.h.0.attn.c_proj.weight' in xmrd.keys() # recorded output of c_proj
    assert 'transformer.h.0.ln_2.weight' in md.keys() # model weight for ln_2

    x = xmrd['transformer.h.0.attn.c_proj.weight'] # c_proj result is input to ln_2
    w = md['transformer.h.0.ln_2.weight'] # weights to use
    assert x.shape[-1] == w.shape[-1]

    yw = xpu.layer_norm(x, w)
    t_lynw = lambda x: t2x(F.layer_norm(x2t(x), w.shape, x2t(w)))
    assert xp.allclose(yw, t_lynw(x))    


def closenuf(a, b):
    return xp.allclose(a, b, rtol=1e-04, atol=1e-04, equal_nan=True)


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
        
        if isinstance(a, xp.ndarray) \
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


def test_torch_xp_correspondence(miked_model_and_handles, savez_loaded_to_xp_dict, brief_toks):
    # pdb helpers

    def c_match():
        return [c for c in clumps if len(c)>1 and not c.names[0].startswith('t.')]

    def k_about(k):
        return about(x2t(xrd[k]) - trd[k])

    model, handles = miked_model_and_handles
    torch_logits, torch_record = logits_and_recording_torch(model, brief_toks)
    md = savez_loaded_to_xp_dict
    xp_logits, xp_record = logits_and_recording_xp(md, brief_toks)
    trd = dict(torch_record)
    xrd = dict((k.removesuffix('.weight'), v) for k,v in xp_record)
    assert closenuf(xrd['tok_emb'], t2x(trd['transformer.wte']))
    assert closenuf(xrd['pos_emb'], t2x(trd['transformer.wpe']))


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
    idx_xp = xp.array(idx)
    y_xp = forward_model_dict(md, idx_xp, recorder=record_module_output)
    xp_model_record = config.model_record
    config.model_record = []
    len(xp_model_record)











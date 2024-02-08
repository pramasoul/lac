# Test xpu utilities

import pytest
import torch
import ctypes
import json
import logging
import subprocess

from binascii import hexlify, unhexlify
from contextlib import contextmanager
from io import BytesIO
from typing import Callable, List

import cupy as cp
import numpy as np

# import unittest.mock as mock
from unittest.mock import mock_open

import xpu


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
def brief_text_file(tmp_path):
    return tmp_file


################
# xpu-specific fixtures

@pytest.fixture
def xp():
    return xpu.xp

@pytest.fixture
def config():
    return xpu.GPTConfig

################
# Tests

from xpu import NDArray
import torch
import torch.nn.functional as F

def x2t(x):
    return torch.from_numpy(xpu.xp.asnumpy(x))

def test_x2t(xp):
    a = xp.arange(3)
    t = x2t(a)
    assert all(t == torch.arange(3))


def t2x(t):
    return xpu.xp.asarray(t)

def test_t2x(xp):
    t = torch.arange(3)
    assert xp.all(t2x(t) == xp.arange(3))



def test_NDArray_exists():
    assert type(NDArray) in (type(np.ndarray), type(cp.ndarray))


def test_GPTConfig(config):
    assert config.vocab_size == 50304


def test_softmax(xp):
    assert xp.all(xpu.softmax(xp.ones(3)) == xp.ones(3)/3)
    x = xp.arange(3, dtype=xp.float32)
    assert xp.allclose(xpu.softmax(x), t2x(F.softmax(x2t(x), dim=0)))


def xify(f):
    return lambda x: t2x(f(x2t(x)))

def test_xify(xp):
    ft_arange = xify(torch.arange)
    assert xp.all(ft_arange(3) == xp.arange(3))
    

def test_gelu(xp):
    x = xp.linspace(-3, 3)
    y = xpu.gelu(x)
    assert xp.allclose(y, xify(F.gelu)(x))


def test_layer_norm(xp):
    x = xp.linspace(-3, 3)
    y = xpu.layer_norm(x)
    assert xp.allclose(y, xify(F.layer_norm)(x))
















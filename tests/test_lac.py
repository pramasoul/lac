# Test lac compressor

import pytest
import torch
import ctypes
import json
import logging

from binascii import hexlify, unhexlify
from contextlib import contextmanager
from io import BytesIO
from typing import Callable, List

import numpy as np

import unittest.mock as mock

#from arithmetic_coding import ACSampler, packbits, unpackbits
from ac_for_z import ACSampler, packbits, unpackbits

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])  # StreamHandler logs to console

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
    """ A context manager for mocking file operations using BytesIO. """
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
    mocker.patch('builtins.open', mock.mock_open(read_data=mock_file_contents))

    with open('mock_file.txt', 'r') as file:
        data = file.read()

    assert data == mock_file_contents

def test_write_file(mocker):
    mock_write = mock.mock_open()
    mocker.patch('builtins.open', mock_write)

    data_to_write = "data to be written"
    with open('mock_file.txt', 'w') as file:
        file.write(data_to_write)

    mock_write.assert_called_once_with('mock_file.txt', 'w')
    mock_write().write.assert_called_once_with(data_to_write)


def test_multiple_file_mocks(mocker):
    # Define the mock file contents for different files
    file_mocks = {
        'foo': mock.mock_open(read_data='data from foo').return_value,
        'bar': mock.mock_open(read_data='data from bar').return_value
    }

    # Side effect function to return different mocks based on filename
    def side_effect(filename, *args, **kwargs):
        return file_mocks[filename]

    # Patch 'open' with the side effect
    mocker.patch('builtins.open', side_effect=side_effect)

    # Test reading from 'foo'
    with open('foo', 'r') as file:
        data = file.read()
    assert data == 'data from foo'

    # Test reading from 'bar'
    with open('bar', 'r') as file:
        data = file.read()
    assert data == 'data from bar'



def test_lacz_header():
    raise NotImplementedError

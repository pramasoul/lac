"""Test script for atok_compressor"""

#import array
#import functools
#import io
import logging
#import os
#import pathlib
import random
import struct
import sys

import pytest

from binascii import hexlify, unhexlify
from contextlib import contextmanager
from io import BytesIO, DEFAULT_BUFFER_SIZE
from typing import Callable, List, Tuple

from unittest.mock import mock_open

import tok_compressor as tc
import atok_compressor as atc

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

def test_bytes_to_bits():
    y2b = atc.bytes_to_bits
    assert y2b(b'') == []
    assert y2b(b'0') == [0,0,1,1, 0,0,0,0]
    assert y2b(unhexlify('cafe')) == [1,1,0,0, 1,0,1,0, 1,1,1,1, 1,1,1,0]

def test_bits_to_bytes():
    b2y = atc.bits_to_bytes
    y2b = atc.bytes_to_bits
    assert b2y([]) == (b'', [])
    assert b2y([1]) == (b'', [1])
    assert b2y([0]*7) == (b'', [0] * 7)
    assert b2y([1]*7) == (b'', [1] * 7)
    assert b2y([1]*8) == (b'\xff', [])
    assert b2y([1]*9) == (b'\xff', [1])
    for s in ('a5c369', 'deadbeefcafebabe'):
        assert b2y(y2b(unhexlify(s))) == (unhexlify(s), [])
    for s in ('foo', 'bar', 'foobar'):
        b = s.encode('utf8')
        assert b2y(y2b((b))) == (b, [])

def test_bits_to_bytes_to_bits():
    b2y = atc.bits_to_bytes
    y2b = atc.bytes_to_bits
    random.seed(42)
    for l in (0, 1, 10, 1024, 1<<20-1):
        b = random.randbytes(1024)
        assert b2y(y2b((b))) == (b, [])


def test_compress_empty():
    c = atc.ACTokCompressor()
    compressed = c.compress("") + c.flush()
    assert compressed == b"\xfe\xfe\xff\xffThat's all, folks!"
    
def test_decompress_empty():
    compressed = b"\xfe\xfe\xff\xffThat's all, folks!"
    d = atc.ACTokDecompressor()
    decompressed = d.decompress(compressed)
    assert decompressed == b""


@pytest.fixture
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

def compress_decompress_test(text):
    c = atc.ACTokCompressor()
    d = atc.ACTokDecompressor()
    compressed = c.compress(text) + c.flush()
    decompressed = d.decompress(compressed)
    if type(text) is str: text = text.encode("utf8")
    assert text == decompressed
    
def test_cd_short():
    compress_decompress_test(b"")
    compress_decompress_test(b"Hi!")
    compress_decompress_test(b"Hi!Hi!")
    compress_decompress_test(b"The quick brown fox, et al.")

def test_cd_short_nl():
    for c,d in ((tc.TokCompressor(),tc.TokDecompressor()), (atc.ACTokCompressor(),atc.ACTokDecompressor())):
        text = b"\n"
        zbody = c.compress(text)
        ztail = c.flush()
        reconstructed = d.decompress(zbody + ztail)
        assert text == reconstructed

def test_cd_brief():
    compress_decompress_test(b"The quick brown fox jumped over the lazy dogs.\n")

def test_cd_medium(medium_text):
    compress_decompress_test(medium_text)

def cd_char_at_a_time_test(text):
    comp = atc.ACTokCompressor()
    compressed = b"".join(comp.compress(char) for char in text) + comp.flush()
    decompressed = atc.ACTokDecompressor().decompress(compressed)
    if type(text) is str: text = text.encode("utf8")
    assert text == decompressed
    decomp = atc.ACTokDecompressor()
    decompressed = b"".join(decomp.decompress(bytes([b])) for b in compressed)
    assert text == decompressed

    
def test_cd_caat(medium_text):
    cd_char_at_a_time_test("")
    cd_char_at_a_time_test("Hi!")
    cd_char_at_a_time_test("The quick brown fox jumped over the lazy dogs.\n")
    cd_char_at_a_time_test(medium_text)

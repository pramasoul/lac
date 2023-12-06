"""Test script for the blacz module."""

import array
import functools
import io
import logging
import os
import pathlib
import struct
import sys
import pytest
from subprocess import PIPE, Popen

from binascii import hexlify, unhexlify
from contextlib import contextmanager
from io import BytesIO, DEFAULT_BUFFER_SIZE
from typing import Callable, List

from unittest.mock import mock_open

from blacz import LacFile

data1 = b"""  int length=DEFAULTALLOC, err = Z_OK;
  PyObject *RetVal;
  int flushmode = Z_FINISH;
  unsigned long start_total_out;

"""

data2 = b"""/* zlibmodule.c -- lac-compatible data compression */
/* See http://www.lac.org/zlib/
/* See http://www.winimage.com/zLibDll for Windows */
"""



class UnseekableIO(io.BytesIO):
    def seekable(self):
        return False

    def tell(self):
        raise io.UnsupportedOperation

    def seek(self, *args):
        raise io.UnsupportedOperation


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


def test_write(tmp_path):
    filename = tmp_path / "foo"
    with LacFile(filename, 'wb') as f:
        f.write(data1 * 50)

        # Try flush and fileno.
        f.flush()
        f.fileno()
        if hasattr(os, 'fsync'):
            os.fsync(f.fileno())
        f.close()

    # Test multiple close() calls.
    f.close()


def test_write_read_with_pathlike_file_short(tmp_path):
    filename = tmp_path / "foo"
    data1 = b"Hi"
    with LacFile(filename, 'w') as f:
        f.write(data1 * 2)
    assert isinstance(f.name, str)
    with LacFile(filename, 'a') as f:
        f.write(data1)
    with LacFile(filename) as f:
        d = f.read()
    assert d == data1 * 3
    assert isinstance(f.name, str)

def test_write_read_with_pathlike_file(tmp_path):
    filename = tmp_path / "foo"
    with LacFile(filename, 'w') as f:
        f.write(data1 * 50)
    assert isinstance(f.name, str)
    with LacFile(filename, 'a') as f:
        f.write(data1)
    with LacFile(filename) as f:
        d = f.read()
    assert d == data1 * 51
    assert isinstance(f.name, str)

def test_bad_args():
    with pytest.raises(TypeError):
        LacFile(123.456)
    with pytest.raises(ValueError):
        LacFile(os.devnull, "z")
    with pytest.raises(ValueError):
        LacFile(os.devnull, "rx")
    with pytest.raises(ValueError):
        LacFile(os.devnull, "rbt")
    with pytest.raises(ValueError):
        LacFile(os.devnull, compresslevel=0)
    with pytest.raises(ValueError):
        LacFile(os.devnull, compresslevel=10)

    # compresslevel is keyword-only
    with pytest.raises(TypeError):
        LacFile(os.devnull, "r", 3)

    
from test import support
from test.support import bigmemtest, _4G

import pickle
import glob
import tempfile
import random
import shutil
import subprocess
import threading
from test.support import import_helper
from test.support import threading_helper
from test.support.os_helper import unlink
import _compression

# import bz2
import blacz as lac
from blacz import LacFile, LacCompressor, LacDecompressor

has_cmdline_bunzip2 = False

def ext_decompress(data):
    global has_cmdline_bunzip2
    if has_cmdline_bunzip2 is None:
        has_cmdline_bunzip2 = bool(shutil.which('bunzip2'))
    if has_cmdline_bunzip2:
        return subprocess.check_output(['bunzip2'], input=data)
    else:
        # return LacDecompressor.decompress(data)
        return lac.decompress(data)

class BaseTest:
    "Base for other testcases."

    TEXT_LINES = [
        b'root:x:0:0:root:/root:/bin/bash\n',
        b'bin:x:1:1:bin:/bin:\n',
        b'daemon:x:2:2:daemon:/sbin:\n',
        b'adm:x:3:4:adm:/var/adm:\n',
        b'lp:x:4:7:lp:/var/spool/lpd:\n',
        b'sync:x:5:0:sync:/sbin:/bin/sync\n',
        b'shutdown:x:6:0:shutdown:/sbin:/sbin/shutdown\n',
        b'halt:x:7:0:halt:/sbin:/sbin/halt\n',
        b'mail:x:8:12:mail:/var/spool/mail:\n',
        b'news:x:9:13:news:/var/spool/news:\n',
        b'uucp:x:10:14:uucp:/var/spool/uucp:\n',
        b'operator:x:11:0:operator:/root:\n',
        b'games:x:12:100:games:/usr/games:\n',
        b'gopher:x:13:30:gopher:/usr/lib/gopher-data:\n',
        b'ftp:x:14:50:FTP User:/var/ftp:/bin/bash\n',
        b'nobody:x:65534:65534:Nobody:/home:\n',
        b'postfix:x:100:101:postfix:/var/spool/postfix:\n',
        b'niemeyer:x:500:500::/home/niemeyer:/bin/bash\n',
        b'postgres:x:101:102:PostgreSQL Server:/var/lib/pgsql:/bin/bash\n',
        b'mysql:x:102:103:MySQL server:/var/lib/mysql:/bin/bash\n',
        b'www:x:103:104::/var/www:/bin/false\n',
        ]
    TEXT = b''.join(TEXT_LINES)
    # DATA = b'BZh91AY&SY.\xc8N\x18\x00\x01>_\x80\x00\x10@\x02\xff\xf0\x01\x07n\x00?\xe7\xff\xe00\x01\x99\xaa\x00\xc0\x03F\x86\x8c#&\x83F\x9a\x03\x06\xa6\xd0\xa6\x93M\x0fQ\xa7\xa8\x06\x804hh\x12$\x11\xa4i4\xf14S\xd2<Q\xb5\x0fH\xd3\xd4\xdd\xd5\x87\xbb\xf8\x94\r\x8f\xafI\x12\xe1\xc9\xf8/E\x00pu\x89\x12]\xc9\xbbDL\nQ\x0e\t1\x12\xdf\xa0\xc0\x97\xac2O9\x89\x13\x94\x0e\x1c7\x0ed\x95I\x0c\xaaJ\xa4\x18L\x10\x05#\x9c\xaf\xba\xbc/\x97\x8a#C\xc8\xe1\x8cW\xf9\xe2\xd0\xd6M\xa7\x8bXa<e\x84t\xcbL\xb3\xa7\xd9\xcd\xd1\xcb\x84.\xaf\xb3\xab\xab\xad`n}\xa0lh\tE,\x8eZ\x15\x17VH>\x88\xe5\xcd9gd6\x0b\n\xe9\x9b\xd5\x8a\x99\xf7\x08.K\x8ev\xfb\xf7xw\xbb\xdf\xa1\x92\xf1\xdd|/";\xa2\xba\x9f\xd5\xb1#A\xb6\xf6\xb3o\xc9\xc5y\\\xebO\xe7\x85\x9a\xbc\xb6f8\x952\xd5\xd7"%\x89>V,\xf7\xa6z\xe2\x9f\xa3\xdf\x11\x11"\xd6E)I\xa9\x13^\xca\xf3r\xd0\x03U\x922\xf26\xec\xb6\xed\x8b\xc3U\x13\x9d\xc5\x170\xa4\xfa^\x92\xacDF\x8a\x97\xd6\x19\xfe\xdd\xb8\xbd\x1a\x9a\x19\xa3\x80ankR\x8b\xe5\xd83]\xa9\xc6\x08\x82f\xf6\xb9"6l$\xb8j@\xc0\x8a\xb0l1..\xbak\x83ls\x15\xbc\xf4\xc1\x13\xbe\xf8E\xb8\x9d\r\xa8\x9dk\x84\xd3n\xfa\xacQ\x07\xb1%y\xaav\xb4\x08\xe0z\x1b\x16\xf5\x04\xe9\xcc\xb9\x08z\x1en7.G\xfc]\xc9\x14\xe1B@\xbb!8`'
    DATA = b'\x00' + TEXT + b'\xff'
    # EMPTY_DATA = b'BZh9\x17rE8P\x90\x00\x00\x00\x00'
    EMPTY_DATA = b'\x00\xff'
    BAD_DATA = b'this is not a valid bzip2 file'

    # Some tests need more than one block of uncompressed data. Since one block
    # is at least 100,000 bytes, we gather some data dynamically and compress it.
    # Note that this assumes that compression works correctly, so we cannot
    # simply use the bigger test data for all tests.
    test_size = 0
    BIG_TEXT = bytearray(128*1024)
    for fname in glob.glob(os.path.join(glob.escape(os.path.dirname(__file__)), '*.py')):
        with open(fname, 'rb') as fh:
            test_size += fh.readinto(memoryview(BIG_TEXT)[test_size:])
        if test_size > 128*1024:
            break
    BIG_DATA = lac.compress(BIG_TEXT, compresslevel=1)

    def setup_method(self):
        fd, self.filename = tempfile.mkstemp()
        os.close(fd)

    def teardown_method(self):
        unlink(self.filename)


class Test_LacFile(BaseTest):
    "Test the LacFile class."

    def createTempFile(self, streams=1, suffix=b""):
        with open(self.filename, "wb") as f:
            f.write(self.DATA * streams)
            f.write(suffix)

    def test_bad_args(self):
        pytest.raises(TypeError, LacFile, 123.456)
        pytest.raises(ValueError, LacFile, os.devnull, "z")
        pytest.raises(ValueError, LacFile, os.devnull, "rx")
        pytest.raises(ValueError, LacFile, os.devnull, "rbt")
        pytest.raises(ValueError, LacFile, os.devnull, compresslevel=0)
        pytest.raises(ValueError, LacFile, os.devnull, compresslevel=10)

        # compresslevel is keyword-only
        pytest.raises(TypeError, LacFile, os.devnull, "r", 3)

    def test_read(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.read, float())
            assert bz2f.read() == self.TEXT

    def test_read_bad_file(self):
        self.createTempFile(streams=0, suffix=self.BAD_DATA)
        with LacFile(self.filename) as bz2f:
            pytest.raises(OSError, bz2f.read)

    def test_read_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.read, float())
            assert bz2f.read() == self.TEXT * 5

    def test_read_monkey_multi_stream(self):
        # Test LacFile.read() on a multi-stream archive where a stream
        # boundary coincides with the end of the raw read buffer.
        buffer_size = _compression.BUFFER_SIZE
        _compression.BUFFER_SIZE = len(self.DATA)
        try:
            self.createTempFile(streams=5)
            with LacFile(self.filename) as bz2f:
                pytest.raises(TypeError, bz2f.read, float())
                assert bz2f.read() == self.TEXT * 5
        finally:
            _compression.BUFFER_SIZE = buffer_size

    def test_read_trailing_junk(self):
        self.createTempFile(suffix=self.BAD_DATA)
        with LacFile(self.filename) as bz2f:
            assert bz2f.read() == self.TEXT

    def test_read_multi_stream_trailing_junk(self):
        self.createTempFile(streams=5, suffix=self.BAD_DATA)
        with LacFile(self.filename) as bz2f:
            assert bz2f.read() == self.TEXT * 5

    def test_read0(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.read, float())
            assert bz2f.read(0) == b""

    def test_read_chunk10(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            text = b''
            while True:
                str = bz2f.read(10)
                if not str:
                    break
                text += str
            assert text == self.TEXT

    def test_read_chunk10_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            text = b''
            while True:
                str = bz2f.read(10)
                if not str:
                    break
                text += str
            assert text == self.TEXT * 5

    def test_read100(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            assert bz2f.read(100) == self.TEXT[:100]

    def test_peek(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pdata = bz2f.peek()
            assert len(pdata) != 0
            assert self.TEXT.startswith(pdata)
            assert bz2f.read() == self.TEXT

    def test_read_into(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            n = 128
            b = bytearray(n)
            assert bz2f.readinto(b) == n
            assert b == self.TEXT[:n]
            n = len(self.TEXT) - n
            b = bytearray(len(self.TEXT))
            assert bz2f.readinto(b) == n
            assert b[:n] == self.TEXT[-n:]

    def test_read_line(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.readline, None)
            for line in self.TEXT_LINES:
                assert bz2f.readline() == line

    def test_read_line_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.readline, None)
            for line in self.TEXT_LINES * 5:
                assert bz2f.readline() == line

    def test_read_lines(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.readlines, None)
            assert bz2f.readlines() == self.TEXT_LINES

    def test_read_lines_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.readlines, None)
            assert bz2f.readlines() == self.TEXT_LINES * 5

    def test_iterator(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            assert list(iter(bz2f)) == self.TEXT_LINES

    def test_iterator_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            assert list(iter(bz2f)) == self.TEXT_LINES * 5

    def test_closed_iterator_deadlock(self):
        # Issue #3309: Iteration on a closed LacFile should release the lock.
        self.createTempFile()
        bz2f = LacFile(self.filename)
        bz2f.close()
        pytest.raises(ValueError, next, bz2f)
        # This call will deadlock if the above call failed to release the lock.
        pytest.raises(ValueError, bz2f.readlines)

    def test_write(self):
        with LacFile(self.filename, "w") as bz2f:
            pytest.raises(TypeError, bz2f.write)
            bz2f.write(self.TEXT)
        with open(self.filename, 'rb') as f:
            assert ext_decompress(f.read()) == self.TEXT

    def test_write_chunks10(self):
        with LacFile(self.filename, "w") as bz2f:
            n = 0
            while True:
                str = self.TEXT[n*10:(n+1)*10]
                if not str:
                    break
                bz2f.write(str)
                n += 1
        with open(self.filename, 'rb') as f:
            assert ext_decompress(f.read()) == self.TEXT

    def test_write_non_default_compress_level(self):
        expected = lac.compress(self.TEXT, compresslevel=5)
        with LacFile(self.filename, "w", compresslevel=5) as bz2f:
            bz2f.write(self.TEXT)
        with open(self.filename, "rb") as f:
            assert f.read() == expected

    def test_write_lines(self):
        with LacFile(self.filename, "w") as bz2f:
            pytest.raises(TypeError, bz2f.writelines)
            bz2f.writelines(self.TEXT_LINES)
        # Issue #1535500: Calling writelines() on a closed LacFile
        # should raise an exception.
        pytest.raises(ValueError, bz2f.writelines, ["a"])
        with open(self.filename, 'rb') as f:
            assert ext_decompress(f.read()) == self.TEXT

    def test_write_methods_on_read_only_file(self):
        with LacFile(self.filename, "w") as bz2f:
            bz2f.write(b"abc")

        with LacFile(self.filename, "r") as bz2f:
            pytest.raises(OSError, bz2f.write, b"a")
            pytest.raises(OSError, bz2f.writelines, [b"a"])

    def test_append(self):
        with LacFile(self.filename, "w") as bz2f:
            pytest.raises(TypeError, bz2f.write)
            bz2f.write(self.TEXT)
        with LacFile(self.filename, "a") as bz2f:
            pytest.raises(TypeError, bz2f.write)
            bz2f.write(self.TEXT)
        with open(self.filename, 'rb') as f:
            assert ext_decompress(f.read()) == self.TEXT * 2

    def test_seek_forward(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.seek)
            bz2f.seek(150)
            assert bz2f.read() == self.TEXT[150:]

    def test_seek_forward_across_streams(self):
        self.createTempFile(streams=2)
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.seek)
            bz2f.seek(len(self.TEXT) + 150)
            assert bz2f.read() == self.TEXT[150:]

    def test_seek_backwards(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            bz2f.read(500)
            bz2f.seek(-150, 1)
            assert bz2f.read() == self.TEXT[500-150:]

    def test_seek_backwards_across_streams(self):
        self.createTempFile(streams=2)
        with LacFile(self.filename) as bz2f:
            readto = len(self.TEXT) + 100
            while readto > 0:
                readto -= len(bz2f.read(readto))
            bz2f.seek(-150, 1)
            assert bz2f.read() == self.TEXT[100-150:] + self.TEXT

    def test_seek_backwards_from_end(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            bz2f.seek(-150, 2)
            assert bz2f.read() == self.TEXT[len(self.TEXT)-150:]

    def test_seek_backwards_from_end_across_streams(self):
        self.createTempFile(streams=2)
        with LacFile(self.filename) as bz2f:
            bz2f.seek(-1000, 2)
            assert bz2f.read() == (self.TEXT * 2)[-1000:]

    def test_seek_post_end(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            bz2f.seek(150000)
            assert bz2f.tell() == len(self.TEXT)
            assert bz2f.read() == b""

    def test_seek_post_end_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            bz2f.seek(150000)
            assert bz2f.tell() == len(self.TEXT) * 5
            assert bz2f.read() == b""

    def test_seek_post_end_twice(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            bz2f.seek(150000)
            bz2f.seek(150000)
            assert bz2f.tell() == len(self.TEXT)
            assert bz2f.read() == b""

    def test_seek_post_end_twice_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            bz2f.seek(150000)
            bz2f.seek(150000)
            assert bz2f.tell() == len(self.TEXT) * 5
            assert bz2f.read() == b""

    def test_seek_pre_start(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            bz2f.seek(-150)
            assert bz2f.tell() == 0
            assert bz2f.read() == self.TEXT

    def test_seek_pre_start_multi_stream(self):
        self.createTempFile(streams=2)
        with LacFile(self.filename) as bz2f:
            bz2f.seek(-150)
            assert bz2f.tell() == 0
            assert bz2f.read() == self.TEXT * 2

    def test_fileno(self):
        self.createTempFile()
        with open(self.filename, 'rb') as rawf:
            bz2f = LacFile(rawf)
            try:
                assert bz2f.fileno() == rawf.fileno()
            finally:
                bz2f.close()
        pytest.raises(ValueError, bz2f.fileno)

    def test_seekable(self):
        bz2f = LacFile(BytesIO(self.DATA))
        try:
            assert bz2f.seekable()
            bz2f.read()
            assert bz2f.seekable()
        finally:
            bz2f.close()
        pytest.raises(ValueError, bz2f.seekable)

        bz2f = LacFile(BytesIO(), "w")
        try:
            assert not bz2f.seekable()
        finally:
            bz2f.close()
        pytest.raises(ValueError, bz2f.seekable)

        src = BytesIO(self.DATA)
        src.seekable = lambda: False
        bz2f = LacFile(src)
        try:
            assert not bz2f.seekable()
        finally:
            bz2f.close()
        pytest.raises(ValueError, bz2f.seekable)

    def test_readable(self):
        bz2f = LacFile(BytesIO(self.DATA))
        try:
            assert bz2f.readable()
            bz2f.read()
            assert bz2f.readable()
        finally:
            bz2f.close()
        pytest.raises(ValueError, bz2f.readable)

        bz2f = LacFile(BytesIO(), "w")
        try:
            assert not bz2f.readable()
        finally:
            bz2f.close()
        pytest.raises(ValueError, bz2f.readable)

    def test_writable(self):
        bz2f = LacFile(BytesIO(self.DATA))
        try:
            assert not bz2f.writable()
            bz2f.read()
            assert not bz2f.writable()
        finally:
            bz2f.close()
        pytest.raises(ValueError, bz2f.writable)

        bz2f = LacFile(BytesIO(), "w")
        try:
            assert bz2f.writable()
        finally:
            bz2f.close()
        pytest.raises(ValueError, bz2f.writable)

    def test_open_del(self):
        self.createTempFile()
        for i in range(10000):
            o = LacFile(self.filename)
            del o

    def test_open_nonexistent(self):
        pytest.raises(OSError, LacFile, "/non/existent")

    def test_readlines_no_newline(self):
        # Issue #1191043: readlines() fails on a file containing no newline.
        # data = b'BZh91AY&SY\xd9b\x89]\x00\x00\x00\x03\x80\x04\x00\x02\x00\x0c\x00 \x00!\x9ah3M\x13<]\xc9\x14\xe1BCe\x8a%t'
        data = b'\x00Test\xff'
        with open(self.filename, "wb") as f:
            f.write(data)
        with LacFile(self.filename) as bz2f:
            lines = bz2f.readlines()
        assert lines == [b'Test']
        with LacFile(self.filename) as bz2f:
            xlines = list(bz2f.readlines())
        assert xlines == [b'Test']

    def test_context_protocol(self):
        f = None
        with LacFile(self.filename, "wb") as f:
            f.write(b"xxx")
        f = LacFile(self.filename, "rb")
        f.close()
        try:
            with f:
                pass
        except ValueError:
            pass
        else:
            pytest.fail("__enter__ on a closed file didn't raise an exception")
        try:
            with LacFile(self.filename, "wb") as f:
                1/0
        except ZeroDivisionError:
            pass
        else:
            pytest.fail("1/0 didn't raise an exception")

    def test_threading(self):
        # Issue #7205: Using a LacFile from several threads shouldn't deadlock.
        data = b"1" * 2**20
        nthreads = 10
        with LacFile(self.filename, 'wb') as f:
            def comp():
                for i in range(5):
                    f.write(data)
            threads = [threading.Thread(target=comp) for i in range(nthreads)]
            with threading_helper.start_threads(threads):
                pass

    def test_mixed_iteration_and_reads(self):
        self.createTempFile()
        linelen = len(self.TEXT_LINES[0])
        halflen = linelen // 2
        with LacFile(self.filename) as bz2f:
            bz2f.read(halflen)
            assert next(bz2f) == self.TEXT_LINES[0][halflen:]
            assert bz2f.read() == self.TEXT[linelen:]
        with LacFile(self.filename) as bz2f:
            bz2f.readline()
            assert next(bz2f) == self.TEXT_LINES[1]
            assert bz2f.readline() == self.TEXT_LINES[2]
        with LacFile(self.filename) as bz2f:
            bz2f.readlines()
            pytest.raises(StopIteration, next, bz2f)
            assert bz2f.readlines() == []

    def test_multi_stream_ordering(self):
        # Test the ordering of streams when reading a multi-stream archive.
        data1 = b"foo" * 1000
        data2 = b"bar" * 1000
        with LacFile(self.filename, "w") as bz2f:
            bz2f.write(data1)
        with LacFile(self.filename, "a") as bz2f:
            bz2f.write(data2)
        with LacFile(self.filename) as bz2f:
            assert bz2f.read() == data1 + data2

    @pytest.mark.skip(reason="Binary data with our flags in it")
    def test_open_bytes_filename(self):
        str_filename = self.filename
        try:
            bytes_filename = str_filename.encode("ascii")
        except UnicodeEncodeError:
            pytest.skip("Temporary file name needs to be ASCII")
        with LacFile(bytes_filename, "wb") as f:
            f.write(self.DATA)
        with LacFile(bytes_filename, "rb") as f:
            assert f.read() == self.DATA
        # Sanity check that we are actually operating on the right file.
        with LacFile(str_filename, "rb") as f:
            assert f.read() == self.DATA

    @pytest.mark.skip(reason="Binary data with our flags in it")
    def test_open_path_like_filename(self):
        filename = pathlib.Path(self.filename)
        with LacFile(filename, "wb") as f:
            f.write(self.DATA)
        with LacFile(filename, "rb") as f:
            assert f.read() == self.DATA

    @pytest.mark.skip(reason="Binary data with our flags in it")
    def test_decompress_limited(self):
        """Decompressed data buffering should be limited"""
        bomb = lac.compress(b'\0' * int(2e6), compresslevel=9)
        assert len(bomb) < _compression.BUFFER_SIZE

        decomp = LacFile(BytesIO(bomb))
        assert decomp.read(1) == b'\0'
        max_decomp = 1 + DEFAULT_BUFFER_SIZE
        assert decomp._buffer.raw.tell() <= max_decomp, \
            "Excessive amount of data was decompressed"


    # Tests for a LacFile wrapping another file object:

    @pytest.mark.skip(reason="Binary data with our flags in it")
    def test_read_bytes_io(self):
        with BytesIO(self.DATA) as bio:
            with LacFile(bio) as bz2f:
                pytest.raises(TypeError, bz2f.read, float())
                assert bz2f.read() == self.TEXT
            assert not bio.closed

    def test_peek_bytes_io(self):
        with BytesIO(self.DATA) as bio:
            with LacFile(bio) as bz2f:
                pdata = bz2f.peek()
                assert len(pdata) != 0
                assert self.TEXT.startswith(pdata)
                assert bz2f.read() == self.TEXT

    def test_write_bytes_io(self):
        with BytesIO() as bio:
            with LacFile(bio, "w") as bz2f:
                pytest.raises(TypeError, bz2f.write)
                bz2f.write(self.TEXT)
            assert ext_decompress(bio.getvalue()) == self.TEXT
            assert not bio.closed

    def test_seek_forward_bytes_io(self):
        with BytesIO(self.DATA) as bio:
            with LacFile(bio) as bz2f:
                pytest.raises(TypeError, bz2f.seek)
                bz2f.seek(150)
                assert bz2f.read() == self.TEXT[150:]

    def test_seek_backwards_bytes_io(self):
        with BytesIO(self.DATA) as bio:
            with LacFile(bio) as bz2f:
                bz2f.read(500)
                bz2f.seek(-150, 1)
                assert bz2f.read() == self.TEXT[500-150:]

    @pytest.mark.skip(reason="bz2 specific")
    def test_read_truncated(self):
        # Drop the eos_magic field (6 bytes) and CRC (4 bytes).
        truncated = self.DATA[:-10]
        with LacFile(BytesIO(truncated)) as f:
            pytest.raises(EOFError, f.read)
        with LacFile(BytesIO(truncated)) as f:
            assert f.read(len(self.TEXT)) == self.TEXT
            pytest.raises(EOFError, f.read, 1)
        # Incomplete 4-byte file header, and block header of at least 146 bits.
        for i in range(22):
            with LacFile(BytesIO(truncated[:i])) as f:
                pytest.raises(EOFError, f.read, 1)

    def test_issue44439(self):
        q = array.array('Q', [1, 2, 3, 4, 5])
        LENGTH = len(q) * q.itemsize

        with LacFile(BytesIO(), 'w') as f:
            assert f.write(q) == LENGTH
            assert f.tell() == LENGTH


class TestLacCompressor(BaseTest):
    def test_compress(self):
        bz2c = LacCompressor()
        pytest.raises(TypeError, bz2c.compress)
        data = bz2c.compress(self.TEXT)
        data += bz2c.flush()
        assert ext_decompress(data) == self.TEXT

    def test_compress_empty_string(self):
        bz2c = LacCompressor()
        data = bz2c.compress(b'')
        data += bz2c.flush()
        assert data == self.EMPTY_DATA

    def test_compress_chunks10(self):
        bz2c = LacCompressor()
        n = 0
        data = b''
        while True:
            str = self.TEXT[n*10:(n+1)*10]
            if not str:
                break
            data += bz2c.compress(str)
            n += 1
        data += bz2c.flush()
        assert ext_decompress(data) == self.TEXT

    @support.skip_if_pgo_task
    @bigmemtest(size=_4G + 100, memuse=2)
    def test_compress4_g(self, size):
        # "Test LacCompressor.compress()/flush() with >4GiB input"
        bz2c = LacCompressor()
        data = b"x" * size
        try:
            compressed = bz2c.compress(data)
            compressed += bz2c.flush()
        finally:
            data = None  # Release memory
        data = lac.decompress(compressed)
        try:
            assert len(data) == size
            assert len(data.strip(b"x")) == 0
        finally:
            data = None

    def test_pickle(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with pytest.raises(TypeError):
                pickle.dumps(LacCompressor(), proto)


class TestLacDecompressor(BaseTest):
    def test_constructor(self):
        pytest.raises(TypeError, LacDecompressor, 42)

    def test_decompress(self):
        bz2d = LacDecompressor()
        pytest.raises(TypeError, bz2d.decompress)
        text = bz2d.decompress(self.DATA)
        assert text == self.TEXT

    def test_decompress_chunks10(self):
        bz2d = LacDecompressor()
        text = b''
        n = 0
        while True:
            str = self.DATA[n*10:(n+1)*10]
            if not str:
                break
            text += bz2d.decompress(str)
            n += 1
        assert text == self.TEXT

    def test_decompress_unused_data(self):
        bz2d = LacDecompressor()
        unused_data = b"this is unused data"
        text = bz2d.decompress(self.DATA+unused_data)
        assert text == self.TEXT
        assert bz2d.unused_data == unused_data

    def test_eoferror(self):
        bz2d = LacDecompressor()
        text = bz2d.decompress(self.DATA)
        pytest.raises(EOFError, bz2d.decompress, b"anything")
        pytest.raises(EOFError, bz2d.decompress, b"")

    @pytest.mark.skip(reason="likely contains our flags in data to compress")
    @support.skip_if_pgo_task
    @bigmemtest(size=_4G + 100, memuse=3.3)
    def test_decompress4_g(self, size):
        # "Test lac.decompress() with >4GiB input"
        blocksize = 10 * 1024 * 1024
        block = random.randbytes(blocksize)
        try:
            data = block * (size // blocksize + 1)
            compressed = lac.compress(data)
            bz2d = LacDecompressor()
            decompressed = bz2d.decompress(compressed)
            assert decompressed == data
        finally:
            data = None
            compressed = None
            decompressed = None

    def test_pickle(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with pytest.raises(TypeError):
                pickle.dumps(LacDecompressor(), proto)

    def test_decompressor_chunks_maxsize(self):
        bzd = LacDecompressor()
        max_length = 100
        out = []

        # Feed some input
        len_ = len(self.BIG_DATA) - 64
        out.append(bzd.decompress(self.BIG_DATA[:len_],
                                  max_length=max_length))
        assert not bzd.needs_input
        assert len(out[-1]) == max_length

        # Retrieve more data without providing more input
        out.append(bzd.decompress(b'', max_length=max_length))
        assert not bzd.needs_input
        assert len(out[-1]) == max_length

        # Retrieve more data while providing more input
        out.append(bzd.decompress(self.BIG_DATA[len_:],
                                  max_length=max_length))
        assert len(out[-1]) <= max_length

        # Retrieve remaining uncompressed data
        while not bzd.eof:
            out.append(bzd.decompress(b'', max_length=max_length))
            assert len(out[-1]) <= max_length

        out = b"".join(out)
        assert out == self.BIG_TEXT
        assert bzd.unused_data == b""

    def test_decompressor_inputbuf_1(self):
        # Test reusing input buffer after moving existing
        # contents to beginning
        bzd = LacDecompressor()
        out = []

        # Create input buffer and fill it
        assert bzd.decompress(self.DATA[:100],
                                        max_length=0) == b''

        # Retrieve some results, freeing capacity at beginning
        # of input buffer
        out.append(bzd.decompress(b'', 2))

        # Add more data that fits into input buffer after
        # moving existing data to beginning
        out.append(bzd.decompress(self.DATA[100:105], 15))

        # Decompress rest of data
        out.append(bzd.decompress(self.DATA[105:]))
        assert b''.join(out) == self.TEXT

    def test_decompressor_inputbuf_2(self):
        # Test reusing input buffer by appending data at the
        # end right away
        bzd = LacDecompressor()
        out = []

        # Create input buffer and empty it
        assert bzd.decompress(self.DATA[:200],
                                        max_length=0) == b''
        out.append(bzd.decompress(b''))

        # Fill buffer with new data
        out.append(bzd.decompress(self.DATA[200:280], 2))

        # Append some more data, not enough to require resize
        out.append(bzd.decompress(self.DATA[280:300], 2))

        # Decompress rest of data
        out.append(bzd.decompress(self.DATA[300:]))
        assert b''.join(out) == self.TEXT

    def test_decompressor_inputbuf_3(self):
        # Test reusing input buffer after extending it

        bzd = LacDecompressor()
        out = []

        # Create almost full input buffer
        out.append(bzd.decompress(self.DATA[:200], 5))

        # Add even more data to it, requiring resize
        out.append(bzd.decompress(self.DATA[200:300], 5))

        # Decompress rest of data
        out.append(bzd.decompress(self.DATA[300:]))
        assert b''.join(out) == self.TEXT

    def test_failure(self):
        bzd = LacDecompressor()
        pytest.raises(Exception, bzd.decompress, self.BAD_DATA * 30)
        # Previously, a second call could crash due to internal inconsistency
        pytest.raises(Exception, bzd.decompress, self.BAD_DATA * 30)

    @support.refcount_test
    def test_refleaks_in___init__(self):
        gettotalrefcount = support.get_attribute(sys, 'gettotalrefcount')
        bzd = LacDecompressor()
        refs_before = gettotalrefcount()
        for i in range(100):
            bzd.__init__()
        assert gettotalrefcount() - refs_before == pytest.approx(0, abs=10)


class TestCompressDecompress(BaseTest):
    def test_compress(self):
        data = lac.compress(self.TEXT)
        assert ext_decompress(data) == self.TEXT

    def test_compress_empty_string(self):
        text = lac.compress(b'')
        assert text == self.EMPTY_DATA

    def test_decompress(self):
        text = lac.decompress(self.DATA)
        assert text == self.TEXT

    def test_decompress_empty(self):
        text = lac.decompress(b"")
        assert text == b""

    def test_decompress_to_empty_string(self):
        text = lac.decompress(self.EMPTY_DATA)
        assert text == b''

    def test_decompress_incomplete(self):
        pytest.raises(ValueError, lac.decompress, self.DATA[:-10])

    def test_decompress_bad_data(self):
        pytest.raises(OSError, lac.decompress, self.BAD_DATA)

    def test_decompress_multi_stream(self):
        text = lac.decompress(self.DATA * 5)
        assert text == self.TEXT * 5

    def test_decompress_trailing_junk(self):
        text = lac.decompress(self.DATA + self.BAD_DATA)
        assert text == self.TEXT

    def test_decompress_multi_stream_trailing_junk(self):
        text = lac.decompress(self.DATA * 5 + self.BAD_DATA)
        assert text == self.TEXT * 5


class TestOpen(BaseTest):
    "Test the open function."

    def open(self, *args, **kwargs):
        return lac.open(*args, **kwargs)

    def test_binary_modes(self):
        for mode in ("wb", "xb"):
            if mode == "xb":
                unlink(self.filename)
            with self.open(self.filename, mode) as f:
                f.write(self.TEXT)
            with open(self.filename, "rb") as f:
                file_data = ext_decompress(f.read())
                assert file_data == self.TEXT
            with self.open(self.filename, "rb") as f:
                assert f.read() == self.TEXT
            with self.open(self.filename, "ab") as f:
                f.write(self.TEXT)
            with open(self.filename, "rb") as f:
                file_data = ext_decompress(f.read())
                assert file_data == self.TEXT * 2

    def test_implicit_binary_modes(self):
        # Test implicit binary modes (no "b" or "t" in mode string).
        for mode in ("w", "x"):
            if mode == "x":
                unlink(self.filename)
            with self.open(self.filename, mode) as f:
                f.write(self.TEXT)
            with open(self.filename, "rb") as f:
                file_data = ext_decompress(f.read())
                assert file_data == self.TEXT
            with self.open(self.filename, "r") as f:
                assert f.read() == self.TEXT
            with self.open(self.filename, "a") as f:
                f.write(self.TEXT)
            with open(self.filename, "rb") as f:
                file_data = ext_decompress(f.read())
                assert file_data == self.TEXT * 2

    def test_text_modes(self):
        text = self.TEXT.decode("ascii")
        text_native_eol = text.replace("\n", os.linesep)
        for mode in ("wt", "xt"):
            if mode == "xt":
                unlink(self.filename)
            with self.open(self.filename, mode, encoding="ascii") as f:
                f.write(text)
            with open(self.filename, "rb") as f:
                file_data = ext_decompress(f.read()).decode("ascii")
                assert file_data == text_native_eol
            with self.open(self.filename, "rt", encoding="ascii") as f:
                assert f.read() == text
            with self.open(self.filename, "at", encoding="ascii") as f:
                f.write(text)
            with open(self.filename, "rb") as f:
                file_data = ext_decompress(f.read()).decode("ascii")
                assert file_data == text_native_eol * 2

    def test_x_mode(self):
        for mode in ("x", "xb", "xt"):
            unlink(self.filename)
            encoding = "utf-8" if "t" in mode else None
            with self.open(self.filename, mode, encoding=encoding) as f:
                pass
            with pytest.raises(FileExistsError):
                with self.open(self.filename, mode) as f:
                    pass

    def test_fileobj(self):
        with self.open(BytesIO(self.DATA), "r") as f:
            assert f.read() == self.TEXT
        with self.open(BytesIO(self.DATA), "rb") as f:
            assert f.read() == self.TEXT
        text = self.TEXT.decode("ascii")
        with self.open(BytesIO(self.DATA), "rt", encoding="utf-8") as f:
            assert f.read() == text

    def test_bad_params(self):
        # Test invalid parameter combinations.
        pytest.raises(ValueError,
                          self.open, self.filename, "wbt")
        pytest.raises(ValueError,
                          self.open, self.filename, "xbt")
        pytest.raises(ValueError,
                          self.open, self.filename, "rb", encoding="utf-8")
        pytest.raises(ValueError,
                          self.open, self.filename, "rb", errors="ignore")
        pytest.raises(ValueError,
                          self.open, self.filename, "rb", newline="\n")

    def test_encoding(self):
        # Test non-default encoding.
        text = self.TEXT.decode("ascii")
        text_native_eol = text.replace("\n", os.linesep)
        with self.open(self.filename, "wt", encoding="utf-16-le") as f:
            f.write(text)
        with open(self.filename, "rb") as f:
            file_data = ext_decompress(f.read()).decode("utf-16-le")
            assert file_data == text_native_eol
        with self.open(self.filename, "rt", encoding="utf-16-le") as f:
            assert f.read() == text

    @pytest.mark.skip(reason="bz2 specific")
    def test_encoding_error_handler(self):
        # Test with non-default encoding error handler.
        with self.open(self.filename, "wb") as f:
            f.write(b"foo\xffbar")
        with self.open(self.filename, "rt", encoding="ascii", errors="ignore") \
                as f:
            assert f.read() == "foobar"

    def test_newline(self):
        # Test with explicit newline (universal newline mode disabled).
        text = self.TEXT.decode("ascii")
        with self.open(self.filename, "wt", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with self.open(self.filename, "rt", encoding="utf-8", newline="\r") as f:
            assert f.readlines() == [text]


def tearDownModule():
    support.reap_children()


if __name__ == '__main__':
    unittest.main()

"""Test script for the tlacz module."""

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

from tlacz import LacFile

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



class Test_Mocks:

    def test_mock_file(self):
        # Usage example
        with mock_file("Initial text") as mock_file_obj:
            # Perform file operations
            mock_file_obj.write(b" More data")
            mock_file_obj.seek(0)
            content = mock_file_obj.read()
            assert content.decode() == " More dataxt"


    def test_read_file(self, mocker):
        mock_file_contents = "mock file data"
        mocker.patch("builtins.open", mock_open(read_data=mock_file_contents))

        with open("mock_file.txt", "r") as file:
            data = file.read()

        assert data == mock_file_contents


    def test_write_file(self, mocker):
        mock_write = mock_open()
        mocker.patch("builtins.open", mock_write)

        data_to_write = "data to be written"
        with open("mock_file.txt", "w") as file:
            file.write(data_to_write)

        mock_write.assert_called_once_with("mock_file.txt", "w")
        mock_write().write.assert_called_once_with(data_to_write)



class Test_Outsiders:

    def test_write(self, tmp_path):
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


    def test_write_read_with_pathlike_file_short(self, tmp_path):
        filename = tmp_path / "foo"
        data1 = b"Hi"
        with LacFile(filename, 'w') as f:
            f.write(data1 * 2)
        assert isinstance(f.name, str)
        with LacFile(filename) as f:
            d = f.read()
        assert d == data1 * 2
        with LacFile(filename, 'a') as f:
            f.write(data1)
        with LacFile(filename) as f:
            d = f.read()
        assert d == data1 * 3
        assert isinstance(f.name, str)

    def test_write_read_with_pathlike_file(self, tmp_path):
        filename = tmp_path / "foo"
        with LacFile(filename, 'w') as f:
            f.write(data1 * 50)
        assert isinstance(f.name, str)
        with LacFile(filename) as f:
            d = f.read()
        assert d == data1 * 50
        with LacFile(filename, 'a') as f:
            f.write(data1)
        with LacFile(filename) as f:
            d = f.read()
        assert d == data1 * 51
        assert isinstance(f.name, str)

    def test_bad_args(self):
        with pytest.raises(TypeError):
            LacFile(123.456)
        with pytest.raises(ValueError):
            LacFile(os.devnull, "z")
        with pytest.raises(ValueError):
            LacFile(os.devnull, "rx")
        with pytest.raises(ValueError):
            LacFile(os.devnull, "rbt")

    @pytest.mark.compressed_data
    def test_read(self):
        text = b"Hello world!"
        #data = b'\xfe\xfe\x88<\xe3\x03\x00\x00\xff\xff'
        #data = b"\xfe\xfeN\xef\x14$\xbb\x92\xa1\xbfThat's all, folks!"
        #data = b"\xfe\xfeN\xef\x14$\xb2\xf4T\xeeThat's all, folks!"
        #data = b"\xfe\xfeN\xef\x14$\xb2\xa1T\xeaThat's all, folks!"
        data = b"\xfe\xfe\xe8\x89\xddfThat's all, folks!"
        with mock_file(data) as f:
            with LacFile(f) as lacf:
                assert lacf.read() == text


    def test_multi_stream_ordering_without_actions(self, tmp_path):
        filename = tmp_path / "foo"
        # Test the ordering of streams when reading a multi-stream archive.
        data1 = b"foo" * 1000
        data2 = b"bar" * 1000
        with LacFile(filename, "w") as bz2f:
            pass
        with LacFile(filename, "a") as bz2f:
            pass
        with LacFile(filename) as bz2f:
            pass

    def test_multi_stream_ordering(self, tmp_path):
        filename = tmp_path / "foo"
        # Test the ordering of streams when reading a multi-stream archive.
        data1 = b"foo" * 1000
        data2 = b"bar" * 1000
        with LacFile(filename, "w") as bz2f:
            bz2f.write(data1)
        with LacFile(filename, "a") as bz2f:
            bz2f.write(data2)
        with LacFile(filename) as bz2f:
            assert bz2f.read() == data1 + data2



# ChatGPT4:
def custom_repr(byte_data):
    if isinstance(byte_data, bytes):
        return "b'" + ''.join(f'\\x{b:02x}' if b < 0x20 or b > 0x7E else chr(b) for b in byte_data) + "'"
    return repr(byte_data)


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
# import blacz as lac
# from blacz import LacFile, LacCompressor, LacDecompressor
import tlacz as lac
from tlacz import LacFile, LacCompressor, LacDecompressor
from atok_compressor import _HEADER#, _EOS

################################################################
### Fixtures

has_cmdline_bunzip2 = False

def ext_decompress(data):
    global has_cmdline_bunzip2
    if has_cmdline_bunzip2 is None:
        has_cmdline_bunzip2 = bool(shutil.which('bunzip2'))
    if has_cmdline_bunzip2:
        return subprocess.check_output(['bunzip2'], input=data)
    else:
        # return LacDecompressor().decompress(data)
        # A different cheat
        return lac.decompress(data)


@pytest.fixture(scope="session")
def TEXT_LINES():
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
    return TEXT_LINES

@pytest.fixture(scope="session")
def TEXT(TEXT_LINES):
    TEXT = b''.join(TEXT_LINES)
    return TEXT

@pytest.fixture(scope="session")
def DATA(TEXT):
    # FIXME: This is a cheat. Need another way to get it.
    DATA = lac.compress(TEXT)
    return DATA

@pytest.fixture(scope="session")
def EMPTY_DATA():
    # FIXME: Probably wrong, definitely wrong in future
    EMPTY_DATA = b"\xfe\xfe\xff\xc0That's all, folks!"
    return EMPTY_DATA


@pytest.fixture(scope="session")
def BAD_DATA():
    BAD_DATA = b'this is not a valid bzip2 file'
    return BAD_DATA


@pytest.fixture(scope="session")
def BIG_TEXT():
    # Some tests need more than one block of uncompressed data. Since one block
    # is at least 100,000 bytes [bzip2], we gather some data dynamically and compress it.
    # Note that this assumes that compression works correctly, so we cannot
    # simply use the bigger test data for all tests.
    test_size = 0
    # Too big for our slowness: BIG_TEXT = bytearray(128*1024)
    size = 8*1024
    BIG_TEXT = bytearray(size)
    for fname in glob.glob(os.path.join(glob.escape(os.path.dirname(__file__)), '*.py')):
        with open(fname, 'rb') as fh:
            test_size += fh.readinto(memoryview(BIG_TEXT)[test_size:])
        if test_size > size:
            break
    return BIG_TEXT


@pytest.fixture(scope="session")
def BIG_DATA(BIG_TEXT):
    BIG_DATA = lac.compress(BIG_TEXT, compresslevel=1)
    return BIG_DATA


@pytest.fixture
def filename():
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    yield filename
    unlink(filename)

# from Test_LacFile
def createTempFile(filename, DATA, streams=1, suffix=b""):
    with open(filename, "wb") as f:
        f.write(DATA * streams)
        f.write(suffix)

@pytest.fixture(scope="session")
def defaultTempFile(DATA):
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    createTempFile(filename, DATA)
    yield filename
    unlink(filename)
    
@pytest.fixture(scope="session")
def twoStreamTempFile(DATA):
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    createTempFile(filename, DATA, streams=2)
    yield filename
    unlink(filename)

@pytest.fixture(scope="session")
def fiveStreamTempFile(DATA):
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    createTempFile(filename, DATA, streams=5)
    yield filename
    unlink(filename)

@pytest.fixture(scope="session")
def badDataSuffixTempFile(DATA, BAD_DATA):
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    createTempFile(filename, DATA, suffix=BAD_DATA)
    yield filename
    unlink(filename)

@pytest.fixture(scope="session")
def fiveStreamBadDataSuffixTempFile(DATA, BAD_DATA):
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    createTempFile(filename, DATA, streams=5, suffix=BAD_DATA)
    yield filename
    unlink(filename)

@pytest.fixture(scope="session")
def noDataTempFile():
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    createTempFile(filename, b"")
    yield filename
    unlink(filename)


################################################################
# Tests


class Test_LacFile:
    "Test the LacFile class."
    def test_bad_args(self):
        pytest.raises(TypeError, LacFile, 123.456)
        pytest.raises(ValueError, LacFile, os.devnull, "z")
        pytest.raises(ValueError, LacFile, os.devnull, "rx")
        pytest.raises(ValueError, LacFile, os.devnull, "rbt")


    @pytest.mark.compressed_data
    def test_read(self, defaultTempFile, TEXT):
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            pytest.raises(TypeError, bz2f.read, float())
            assert bz2f.read() == TEXT


    def test_read_bad_file(self, filename, DATA, BAD_DATA):
        createTempFile(filename, DATA, streams=0, suffix=BAD_DATA)
        with LacFile(filename) as bz2f:
            pytest.raises(OSError, bz2f.read)

    @pytest.mark.compressed_data
    def test_read_multi_stream(self, fiveStreamTempFile, TEXT):
        #createTempFile(filename, DATA, streams=5)
        filename = fiveStreamTempFile
        with LacFile(filename) as bz2f:
            pytest.raises(TypeError, bz2f.read, float())
            assert bz2f.read() == TEXT * 5


    @pytest.mark.compressed_data
    def test_read_monkey_multi_stream(self, fiveStreamTempFile, DATA, TEXT):
        # Test LacFile.read() on a multi-stream archive where a stream
        # boundary coincides with the end of the raw read buffer.
        buffer_size = _compression.BUFFER_SIZE
        _compression.BUFFER_SIZE = len(DATA)
        try:
            filename = fiveStreamTempFile
            with LacFile(filename) as bz2f:
                pytest.raises(TypeError, bz2f.read, float())
                assert bz2f.read() == TEXT * 5
        finally:
            _compression.BUFFER_SIZE = buffer_size

    @pytest.mark.compressed_data
    def test_read_trailing_junk(self, badDataSuffixTempFile, TEXT):
        #self.createTempFile(suffix=self.BAD_DATA)
        filename = badDataSuffixTempFile
        with LacFile(filename) as bz2f:
            assert bz2f.read() == TEXT

    @pytest.mark.compressed_data
    def test_read_multi_stream_trailing_junk(self, fiveStreamBadDataSuffixTempFile, TEXT):
        #self.createTempFile(streams=5, suffix=self.BAD_DATA)
        filename = fiveStreamBadDataSuffixTempFile
        with LacFile(filename) as bz2f:
            assert bz2f.read() == TEXT * 5

    @pytest.mark.compressed_data
    def test_read0(self, noDataTempFile):
        filename = noDataTempFile
        with LacFile(filename) as bz2f:
            pytest.raises(TypeError, bz2f.read, float())
            assert bz2f.read(0) == b""

    @pytest.mark.compressed_data
    def test_read_chunk10(self, defaultTempFile, TEXT):
        #self.createTempFile()
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            text = b''
            while True:
                str = bz2f.read(10)
                if not str:
                    break
                text += str
            assert text == TEXT

    @pytest.mark.compressed_data
    def test_read_chunk10_multi_stream(self, fiveStreamTempFile, TEXT):
        #self.createTempFile(streams=5)
        filename = fiveStreamTempFile
        with LacFile(filename) as bz2f:
            text = b''
            while True:
                str = bz2f.read(10)
                if not str:
                    break
                text += str
            assert text == TEXT * 5

    @pytest.mark.compressed_data
    def test_read100(self, defaultTempFile, TEXT):
        #self.createTempFile()
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            assert bz2f.read(100) == TEXT[:100]

    @pytest.mark.compressed_data
    def test_peek(self, defaultTempFile, TEXT):
        #self.createTempFile()
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            pdata = bz2f.peek()
            assert len(pdata) != 0
            assert TEXT.startswith(pdata)
            assert bz2f.read() == TEXT


    @pytest.mark.compressed_data
    def test_read_into(self, defaultTempFile, TEXT):
        #self.createTempFile()
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            n = 128
            b = bytearray(n)
            assert bz2f.readinto(b) == n
            assert b == TEXT[:n]
            n = len(TEXT) - n
            b = bytearray(len(TEXT))
            assert bz2f.readinto(b) == n
            assert b[:n] == TEXT[-n:]

    @pytest.mark.compressed_data
    def test_read_line(self, defaultTempFile, TEXT_LINES):
        #self.createTempFile()
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            pytest.raises(TypeError, bz2f.readline, None)
            for line in TEXT_LINES:
                assert bz2f.readline() == line


    @pytest.mark.compressed_data
    def test_read_line_multi_stream(self, fiveStreamTempFile, TEXT_LINES):
        #self.createTempFile(streams=5)
        filename = fiveStreamTempFile
        with LacFile(filename) as bz2f:
            pytest.raises(TypeError, bz2f.readline, None)
            for line in TEXT_LINES * 5:
                assert bz2f.readline() == line


    @pytest.mark.compressed_data
    def test_read_lines(self, defaultTempFile, TEXT_LINES):
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            pytest.raises(TypeError, bz2f.readlines, None)
            assert bz2f.readlines() == TEXT_LINES

    @pytest.mark.compressed_data
    def test_read_lines_multi_stream(self, fiveStreamTempFile, TEXT_LINES):
        filename = fiveStreamTempFile
        with LacFile(filename) as bz2f:
            pytest.raises(TypeError, bz2f.readlines, None)
            assert bz2f.readlines() == TEXT_LINES * 5

    @pytest.mark.compressed_data
    def test_iterator(self, defaultTempFile, TEXT_LINES):
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            assert list(iter(bz2f)) == TEXT_LINES

    @pytest.mark.compressed_data
    def test_iterator_multi_stream(self, fiveStreamTempFile, TEXT_LINES):
        filename = fiveStreamTempFile
        with LacFile(filename) as bz2f:
            assert list(iter(bz2f)) == TEXT_LINES * 5


    def test_closed_iterator_deadlock(self, defaultTempFile):
        # Issue #3309: Iteration on a closed LacFile should release the lock.
        filename = defaultTempFile
        bz2f = LacFile(filename)
        bz2f.close()
        pytest.raises(ValueError, next, bz2f)
        # This call will deadlock if the above call failed to release the lock.
        pytest.raises(ValueError, bz2f.readlines)

    def test_write(self, filename, TEXT):
        with LacFile(filename, "w") as bz2f:
            pytest.raises(TypeError, bz2f.write)
            bz2f.write(TEXT)
        with open(filename, 'rb') as f:
            assert ext_decompress(f.read()) == TEXT


    def test_write_chunks10(self, filename, TEXT):
        with LacFile(filename, "w") as bz2f:
            n = 0
            while True:
                str = TEXT[n*10:(n+1)*10]
                if not str:
                    break
                bz2f.write(str)
                n += 1
        with open(filename, 'rb') as f:
            assert ext_decompress(f.read()) == TEXT


    def test_write_non_default_compress_level(self, filename, TEXT):
        expected = lac.compress(TEXT, compresslevel=5)
        with LacFile(filename, "w", compresslevel=5) as bz2f:
            bz2f.write(TEXT)
        with open(filename, "rb") as f:
            assert f.read() == expected

    def test_write_lines(self, filename, TEXT_LINES, TEXT):
        with LacFile(filename, "w") as bz2f:
            pytest.raises(TypeError, bz2f.writelines)
            bz2f.writelines(TEXT_LINES)
        # Issue #1535500: Calling writelines() on a closed LacFile
        # should raise an exception.
        pytest.raises(ValueError, bz2f.writelines, ["a"])
        with open(filename, 'rb') as f:
            assert ext_decompress(f.read()) == TEXT

    def test_write_methods_on_read_only_file(self, filename, TEXT):
        with LacFile(filename, "w") as bz2f:
            bz2f.write(b"abc")

        with LacFile(filename, "r") as bz2f:
            pytest.raises(OSError, bz2f.write, b"a")
            pytest.raises(OSError, bz2f.writelines, [b"a"])

    def test_append(self, filename, TEXT):
        with LacFile(filename, "w") as bz2f:
            pytest.raises(TypeError, bz2f.write)
            bz2f.write(TEXT)
        with LacFile(filename, "a") as bz2f:
            pytest.raises(TypeError, bz2f.write)
            bz2f.write(TEXT)
        with open(filename, 'rb') as f:
            assert ext_decompress(f.read()) == TEXT * 2


    @pytest.mark.compressed_data
    def test_seek_forward(self, defaultTempFile, TEXT):
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            pytest.raises(TypeError, bz2f.seek)
            bz2f.seek(150)
            assert bz2f.read() == TEXT[150:]

    @pytest.mark.compressed_data
    def test_seek_forward_across_streams(self, twoStreamTempFile, TEXT):
        #self.createTempFile(streams=2)
        filename = twoStreamTempFile
        with LacFile(filename) as bz2f:
            pytest.raises(TypeError, bz2f.seek)
            bz2f.seek(len(TEXT) + 150)
            assert bz2f.read() == TEXT[150:]

    @pytest.mark.compressed_data
    def test_seek_backwards(self, defaultTempFile, TEXT):
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            bz2f.read(500)
            bz2f.seek(-150, 1)
            assert bz2f.read() == TEXT[500-150:]

    @pytest.mark.compressed_data
    def test_seek_backwards_across_streams(self, twoStreamTempFile, TEXT):
        #self.createTempFile(streams=2)
        filename = twoStreamTempFile
        with LacFile(filename) as bz2f:
            readto = len(TEXT) + 100
            while readto > 0:
                readto -= len(bz2f.read(readto))
            bz2f.seek(-150, 1)
            assert bz2f.read() == TEXT[100-150:] + TEXT

    @pytest.mark.compressed_data
    def test_seek_backwards_from_end(self, defaultTempFile, TEXT):
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            bz2f.seek(-150, 2)
            assert bz2f.read() == TEXT[len(TEXT)-150:]

    @pytest.mark.skip(reason="FIXME")
    @pytest.mark.compressed_data
    def test_seek_backwards_from_end_across_streams(self, twoStreamTempFile, TEXT):
        #self.createTempFile(streams=2)
        filename = twoStreamTempFile
        with LacFile(filename) as bz2f:
            bz2f.seek(-1000, 2)
            read_result = bz2f.read()
        assert read_result == (TEXT * 2)[-1000:]

    @pytest.mark.compressed_data
    def test_seek_post_end(self, defaultTempFile, TEXT):
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            bz2f.seek(150000)
            assert bz2f.tell() == len(TEXT)
            assert bz2f.read() == b""

    @pytest.mark.compressed_data
    def test_seek_post_end_multi_stream(self, fiveStreamTempFile, TEXT):
        filename = fiveStreamTempFile
        with LacFile(filename) as bz2f:
            bz2f.seek(150000)
            assert bz2f.tell() == len(TEXT) * 5
            assert bz2f.read() == b""

    @pytest.mark.compressed_data
    def test_seek_post_end_twice(self, defaultTempFile, TEXT):
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            bz2f.seek(150000)
            bz2f.seek(150000)
            assert bz2f.tell() == len(TEXT)
            assert bz2f.read() == b""

    @pytest.mark.compressed_data
    def test_seek_post_end_twice_multi_stream(self, fiveStreamTempFile, TEXT):
        filename = fiveStreamTempFile
        with LacFile(filename) as bz2f:
            bz2f.seek(150000)
            bz2f.seek(150000)
            assert bz2f.tell() == len(TEXT) * 5
            assert bz2f.read() == b""

    @pytest.mark.compressed_data
    def test_seek_pre_start(self, defaultTempFile, TEXT):
        filename = defaultTempFile
        with LacFile(filename) as bz2f:
            bz2f.seek(-150)
            assert bz2f.tell() == 0
            assert bz2f.read() == TEXT


    @pytest.mark.compressed_data
    def test_seek_pre_start_multi_stream(self, twoStreamTempFile, TEXT):
        #self.createTempFile(streams=2)
        filename = twoStreamTempFile
        with LacFile(filename) as bz2f:
            bz2f.seek(-150)
            assert bz2f.tell() == 0
            assert bz2f.read() == TEXT * 2

    def test_fileno(self, defaultTempFile):
        filename = defaultTempFile
        with open(filename, 'rb') as rawf:
            bz2f = LacFile(rawf)
            try:
                assert bz2f.fileno() == rawf.fileno()
            finally:
                bz2f.close()
        pytest.raises(ValueError, bz2f.fileno)

    @pytest.mark.compressed_data
    def test_seekable(self, DATA):
        bz2f = LacFile(BytesIO(DATA))
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

        src = BytesIO(DATA)
        src.seekable = lambda: False
        bz2f = LacFile(src)
        try:
            assert not bz2f.seekable()
        finally:
            bz2f.close()
        pytest.raises(ValueError, bz2f.seekable)


    @pytest.mark.compressed_data
    def test_readable(self, DATA):
        bz2f = LacFile(BytesIO(DATA))
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

    @pytest.mark.compressed_data
    def test_writable(self, DATA):
        bz2f = LacFile(BytesIO(DATA))
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

    @pytest.mark.skip(reason="FIXME")
    @pytest.mark.compressed_data
    def test_open_del(self, filename):
        filename = defaultTempFile
        for i in range(10000):
            o = LacFile(filename)
            del o

    def test_open_nonexistent(self):
        pytest.raises(OSError, LacFile, "/non/existent")


####    @pytest.mark.skip(reason="FIXME")
    @pytest.mark.compressed_data
    def test_readlines_no_newline(self, filename):
        # Issue #1191043: readlines() fails on a file containing no newline.
        # data = b'BZh91AY&SY\xd9b\x89]\x00\x00\x00\x03\x80\x04\x00\x02\x00\x0c\x00 \x00!\x9ah3M\x13<]\xc9\x14\xe1BCe\x8a%t'
        #data = _HEADER + b'Test' + _EOS
        #data = b"\xfe\xfeI]\xc3\x0bThat's all, folks!"
        data = b"\xfe\xfe\xe6\xe7\xa3\x80That's all, folks!"
        with open(filename, "wb") as f:
            f.write(data)
        with LacFile(filename) as bz2f:
            lines = bz2f.readlines()
        assert lines == [b'Test']
        with LacFile(filename) as bz2f:
            xlines = list(bz2f.readlines())
        assert xlines == [b'Test']

    def test_context_protocol(self, filename):
        f = None
        with LacFile(filename, "wb") as f:
            f.write(b"xxx")
        f = LacFile(filename, "rb")
        f.close()
        try:
            with f:
                pass
        except ValueError:
            pass
        else:
            pytest.fail("__enter__ on a closed file didn't raise an exception")
        try:
            with LacFile(filename, "wb") as f:
                1/0
        except ZeroDivisionError:
            pass
        else:
            pytest.fail("1/0 didn't raise an exception")

    @pytest.mark.skip(reason="Too many 1's and we stack overflow in tiktoken.encode")
    def test_threading(self, filename):
        # Issue #7205: Using a LacFile from several threads shouldn't deadlock.
        data = b"1" * 2**20
        nthreads = 10
        with LacFile(filename, 'wb') as f:
            def comp(self):
                for i in range(5):
                    f.write(data)
            threads = [threading.Thread(target=comp) for i in range(nthreads)]
            with threading_helper.start_threads(threads):
                pass

    @pytest.mark.compressed_data
    def test_mixed_iteration_and_reads(self, defaultTempFile, TEXT_LINES, TEXT):
        filename = defaultTempFile
        linelen = len(TEXT_LINES[0])
        halflen = linelen // 2
        with LacFile(filename) as bz2f:
            bz2f.read(halflen)
            assert next(bz2f) == TEXT_LINES[0][halflen:]
            assert bz2f.read() == TEXT[linelen:]
        with LacFile(filename) as bz2f:
            bz2f.readline()
            assert next(bz2f) == TEXT_LINES[1]
            assert bz2f.readline() == TEXT_LINES[2]
        with LacFile(filename) as bz2f:
            bz2f.readlines()
            pytest.raises(StopIteration, next, bz2f)
            assert bz2f.readlines() == []

    def test_multi_stream_ordering(self, filename):
        # Test the ordering of streams when reading a multi-stream archive.
        data1 = b"foo" * 1000
        data2 = b"bar" * 1000
        with LacFile(filename, "w") as bz2f:
            bz2f.write(data1)
        with LacFile(filename, "a") as bz2f:
            bz2f.write(data2)
        with LacFile(filename) as bz2f:
            assert bz2f.read() == data1 + data2

    @pytest.mark.skip(reason="Binary data that doesn't utf-8 decode")
    def test_open_bytes_filename(self, filename, DATA):
        str_filename = filename
        try:
            bytes_filename = str_filename.encode("ascii")
        except UnicodeEncodeError:
            pytest.skip("Temporary file name needs to be ASCII")
        with LacFile(bytes_filename, "wb") as f:
            f.write(DATA)
        with LacFile(bytes_filename, "rb") as f:
            assert f.read() == DATA
        # Sanity check that we are actually operating on the right file.
        with LacFile(str_filename, "rb") as f:
            assert f.read() == DATA

    @pytest.mark.skip(reason="Binary data that doesn't utf-8 decode")
    def test_open_path_like_filename(self, filename, DATA):
        filename = pathlib.Path(filename)
        with LacFile(filename, "wb") as f:
            f.write(DATA)
        with LacFile(filename, "rb") as f:
            assert f.read() == DATA


    @pytest.mark.skip(reason="Blows up tiktoken.encode")
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

    @pytest.mark.compressed_data
    #@pytest.mark.skip(reason="Binary data with our flags in it")
    def test_read_bytes_io(self, DATA, TEXT):
        with BytesIO(DATA) as bio:
            with LacFile(bio) as bz2f:
                pytest.raises(TypeError, bz2f.read, float())
                assert bz2f.read() == TEXT
            assert not bio.closed

    @pytest.mark.compressed_data
    def test_peek_bytes_io(self, DATA, TEXT):
        with BytesIO(DATA) as bio:
            with LacFile(bio) as bz2f:
                pdata = bz2f.peek()
                assert len(pdata) != 0
                assert TEXT.startswith(pdata)
                assert bz2f.read() == TEXT

    def test_write_bytes_io(self, TEXT):
        with BytesIO() as bio:
            with LacFile(bio, "w") as bz2f:
                pytest.raises(TypeError, bz2f.write)
                bz2f.write(TEXT)
            assert ext_decompress(bio.getvalue()) == TEXT
            assert not bio.closed

    @pytest.mark.compressed_data
    def test_seek_forward_bytes_io(self, DATA, TEXT):
        with BytesIO(DATA) as bio:
            with LacFile(bio) as bz2f:
                pytest.raises(TypeError, bz2f.seek)
                bz2f.seek(150)
                assert bz2f.read() == TEXT[150:]

    @pytest.mark.compressed_data
    def test_seek_backwards_bytes_io(self, DATA, TEXT):
        with BytesIO(DATA) as bio:
            with LacFile(bio) as bz2f:
                bz2f.read(500)
                bz2f.seek(-150, 1)
                assert bz2f.read() == TEXT[500-150:]

    @pytest.mark.skip(reason="bz2 specific")
    def test_read_truncated(self, DATA, TEXT):
        # Drop the eos_magic field (6 bytes) and CRC (4 bytes).
        truncated = DATA[:-10]
        with LacFile(BytesIO(truncated)) as f:
            pytest.raises(EOFError, f.read)
        with LacFile(BytesIO(truncated)) as f:
            assert f.read(len(TEXT)) == TEXT
            pytest.raises(EOFError, f.read, 1)
        # Incomplete 4-byte file header, and block header of at least 146 bits.
        for i in range(22):
            with LacFile(BytesIO(truncated[:i])) as f:
                pytest.raises(EOFError, f.read, 1)

    @pytest.mark.skip(reason="bz2 specific")
    def test_issue44439(self):
        q = array.array('Q', [1, 2, 3, 4, 5])
        LENGTH = len(q) * q.itemsize

        with LacFile(BytesIO(), 'w') as f:
            assert f.write(q) == LENGTH
            assert f.tell() == LENGTH


class TestLacCompressor:
    
    def test_compress(self, TEXT):
        bz2c = LacCompressor()
        pytest.raises(TypeError, bz2c.compress)
        data = bz2c.compress(TEXT)
        data += bz2c.flush()
        assert ext_decompress(data) == TEXT

    @pytest.mark.compressed_data
    def test_compress_empty_string(self, EMPTY_DATA):
        bz2c = LacCompressor()
        data = bz2c.compress(b'')
        data += bz2c.flush()
        assert data == EMPTY_DATA

    def test_compress_chunks10(self, TEXT):
        bz2c = LacCompressor()
        n = 0
        data = b''
        while True:
            str = TEXT[n*10:(n+1)*10]
            if not str:
                break
            data += bz2c.compress(str)
            n += 1
        data += bz2c.flush()
        assert ext_decompress(data) == TEXT

    @pytest.mark.skip(reason="FIXME: don't understand")
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


class TestLacDecompressor:

    def test_constructor(self):
        pytest.raises(TypeError, LacDecompressor, 42)

    @pytest.mark.compressed_data
    def test_decompress(self, DATA, TEXT):
        bz2d = LacDecompressor()
        pytest.raises(TypeError, bz2d.decompress)
        text = bz2d.decompress(DATA)
        assert text == TEXT

    @pytest.mark.compressed_data
    def test_decompress_chunks10(self, DATA, TEXT):
        bz2d = LacDecompressor()
        text = b''
        n = 0
        while True:
            str = DATA[n*10:(n+1)*10]
            if not str:
                break
            text += bz2d.decompress(str)
            n += 1
        assert text == TEXT

    @pytest.mark.compressed_data
    def test_decompress_unused_data(self, DATA, TEXT):
        bz2d = LacDecompressor()
        unused_data = b"this is unused data"
        text = bz2d.decompress(DATA+unused_data)
        assert text == TEXT
        assert bz2d.unused_data == unused_data

    @pytest.mark.compressed_data
    def test_eoferror(self, DATA):
        bz2d = LacDecompressor()
        text = bz2d.decompress(DATA)
        pytest.raises(Exception, bz2d.decompress, b"anything")
        pytest.raises(Exception, bz2d.decompress, b"")

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

    @pytest.mark.compressed_data
    def test_decompressor_inputbuf_1(self, DATA, TEXT):
        # Test reusing input buffer after moving existing
        # contents to beginning
        bzd = LacDecompressor()
        out = []

        # Create input buffer and fill it
        assert bzd.decompress(DATA[:100],
                                        max_length=0) == b''

        # Retrieve some results, freeing capacity at beginning
        # of input buffer
        out.append(bzd.decompress(b'', 2))

        # Add more data that fits into input buffer after
        # moving existing data to beginning
        out.append(bzd.decompress(DATA[100:105], 15))

        # Decompress rest of data
        out.append(bzd.decompress(DATA[105:]))
        assert b''.join(out) == TEXT

    @pytest.mark.compressed_data
    def test_decompressor_inputbuf_2(self, DATA, TEXT):
        # Test reusing input buffer by appending data at the
        # end right away
        bzd = LacDecompressor()
        out = []

        # Create input buffer and empty it
        assert bzd.decompress(DATA[:200],
                                        max_length=0) == b''
        out.append(bzd.decompress(b''))

        # Fill buffer with new data
        out.append(bzd.decompress(DATA[200:280], 2))

        # Append some more data, not enough to require resize
        out.append(bzd.decompress(DATA[280:300], 2))

        # Decompress rest of data
        out.append(bzd.decompress(DATA[300:]))
        assert b''.join(out) == TEXT

    @pytest.mark.compressed_data
    def test_decompressor_inputbuf_3(self, DATA, TEXT):
        # Test reusing input buffer after extending it

        bzd = LacDecompressor()
        out = []

        # Create almost full input buffer
        out.append(bzd.decompress(DATA[:200], 5))

        # Add even more data to it, requiring resize
        out.append(bzd.decompress(DATA[200:300], 5))

        # Decompress rest of data
        out.append(bzd.decompress(DATA[300:]))
        assert b''.join(out) == TEXT

    def test_failure(self, BAD_DATA):
        bzd = LacDecompressor()
        pytest.raises(Exception, bzd.decompress, BAD_DATA * 30)
        # Previously, a second call could crash due to internal inconsistency
        pytest.raises(Exception, bzd.decompress, BAD_DATA * 30)

    @support.refcount_test
    def test_refleaks_in___init__(self):
        gettotalrefcount = support.get_attribute(sys, 'gettotalrefcount')
        bzd = LacDecompressor()
        refs_before = gettotalrefcount()
        for i in range(100):
            bzd.__init__()
        assert gettotalrefcount() - refs_before == pytest.approx(0, abs=10)


class TestCompressDecompress:

    def test_compress(self, TEXT):
        data = lac.compress(TEXT)
        assert ext_decompress(data) == TEXT

    @pytest.mark.compressed_data
    def test_compress_empty_string(self, EMPTY_DATA):
        text = lac.compress(b'')
        assert text == EMPTY_DATA

    @pytest.mark.compressed_data
    def test_decompress(self, DATA, TEXT):
        text = lac.decompress(DATA)
        assert text == TEXT

    def test_decompress_empty(self):
        text = lac.decompress(b"")
        assert text == b""

    @pytest.mark.compressed_data
    def test_decompress_to_empty_string(self, EMPTY_DATA):
        text = lac.decompress(EMPTY_DATA)
        assert text == b''

    @pytest.mark.compressed_data
    def test_decompress_incomplete(self, DATA):
        pytest.raises(ValueError, lac.decompress, DATA[:-10])

    def test_decompress_bad_data(self, BAD_DATA):
        pytest.raises(OSError, lac.decompress, BAD_DATA)

    @pytest.mark.compressed_data
    def test_decompress_multi_stream(self, DATA, TEXT):
        text = lac.decompress(DATA * 5)
        assert text == TEXT * 5

    @pytest.mark.compressed_data
    def test_decompress_trailing_junk(self, DATA, BAD_DATA, TEXT):
        text = lac.decompress(DATA + BAD_DATA)
        assert text == TEXT

    @pytest.mark.compressed_data
    def test_decompress_multi_stream_trailing_junk(self, DATA, BAD_DATA, TEXT):
        text = lac.decompress(DATA * 5 + BAD_DATA)
        assert text == TEXT * 5


class TestOpen:
    "Test the open function."

    # def open(self, *args, **kwargs):
    #     return lac.open(*args, **kwargs)

    def test_binary_modes(self, filename, TEXT):
        for mode in ("wb", "xb"):
            if mode == "xb":
                unlink(filename)
            with lac.open(filename, mode) as f:
                f.write(TEXT)
            with open(filename, "rb") as f:
                file_data = ext_decompress(f.read())
                assert file_data == TEXT
            with lac.open(filename, "rb") as f:
                assert f.read() == TEXT
            with lac.open(filename, "ab") as f:
                f.write(TEXT)
            with open(filename, "rb") as f:
                file_data = ext_decompress(f.read())
                assert file_data == TEXT * 2

    def test_implicit_binary_modes(self, filename, TEXT):
        # Test implicit binary modes (no "b" or "t" in mode string).
        for mode in ("w", "x"):
            if mode == "x":
                unlink(filename)
            with lac.open(filename, mode) as f:
                f.write(TEXT)
            with open(filename, "rb") as f:
                file_data = ext_decompress(f.read())
                assert file_data == TEXT
            with lac.open(filename, "r") as f:
                assert f.read() == TEXT
            with lac.open(filename, "a") as f:
                f.write(TEXT)
            with open(filename, "rb") as f:
                file_data = ext_decompress(f.read())
                assert file_data == TEXT * 2

    def test_text_modes(self, TEXT, filename):
        text = TEXT.decode("ascii")
        text_native_eol = text.replace("\n", os.linesep)
        for mode in ("wt", "xt"):
            if mode == "xt":
                unlink(filename)
            with lac.open(filename, mode, encoding="ascii") as f:
                f.write(text)
            with open(filename, "rb") as f:
                file_data = ext_decompress(f.read()).decode("ascii")
                assert file_data == text_native_eol
            with lac.open(filename, "rt", encoding="ascii") as f:
                assert f.read() == text
            with lac.open(filename, "at", encoding="ascii") as f:
                f.write(text)
            with open(filename, "rb") as f:
                file_data = ext_decompress(f.read()).decode("ascii")
                assert file_data == text_native_eol * 2

    def test_x_mode(self, filename):
        for mode in ("x", "xb", "xt"):
            unlink(filename)
            encoding = "utf-8" if "t" in mode else None
            with lac.open(filename, mode, encoding=encoding) as f:
                pass
            with pytest.raises(FileExistsError):
                with lac.open(filename, mode) as f:
                    pass

    @pytest.mark.compressed_data
    def test_fileobj(self, DATA, TEXT):
        with lac.open(BytesIO(DATA), "r") as f:
            assert f.read() == TEXT
        with lac.open(BytesIO(DATA), "rb") as f:
            assert f.read() == TEXT
        text = TEXT.decode("ascii")
        with lac.open(BytesIO(DATA), "rt", encoding="utf-8") as f:
            assert f.read() == text

    def test_bad_params(self):
        # Test invalid parameter combinations.
        pytest.raises(ValueError,
                          lac.open, filename, "wbt")
        pytest.raises(ValueError,
                          lac.open, filename, "xbt")
        pytest.raises(ValueError,
                          lac.open, filename, "rb", encoding="utf-8")
        pytest.raises(ValueError,
                          lac.open, filename, "rb", errors="ignore")
        pytest.raises(ValueError,
                          lac.open, filename, "rb", newline="\n")

    def test_encoding(self, TEXT, filename):
        # Test non-default encoding.
        text = TEXT.decode("ascii")
        text_native_eol = text.replace("\n", os.linesep)
        with lac.open(filename, "wt", encoding="utf-16-le") as f:
            f.write(text)
        with open(filename, "rb") as f:
            file_data = ext_decompress(f.read()).decode("utf-16-le")
            assert file_data == text_native_eol
        with lac.open(filename, "rt", encoding="utf-16-le") as f:
            assert f.read() == text

    @pytest.mark.skip(reason="bz2 specific")
    def test_encoding_error_handler(self, filename):
        # Test with non-default encoding error handler.
        with lac.open(filename, "wb") as f:
            f.write(b"foo\xffbar")
        with lac.open(filename, "rt", encoding="ascii", errors="ignore") \
                as f:
            assert f.read() == "foobar"

    def test_newline(self, TEXT, filename):
        # Test with explicit newline (universal newline mode disabled).
        text = TEXT.decode("ascii")
        with lac.open(filename, "wt", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with lac.open(filename, "rt", encoding="utf-8", newline="\r") as f:
            assert f.readlines() == [text]




@pytest.mark.slow
def test_decompressor_chunks_maxsize(BIG_DATA, BIG_TEXT):
    bzd = LacDecompressor()
    max_length = 100
    out = []

    # Feed some input
    len_ = len(BIG_DATA) - 64
    out.append(bzd.decompress(BIG_DATA[:len_],
                              max_length=max_length))
    assert not bzd.needs_input
    assert len(out[-1]) == max_length

    # Retrieve more data without providing more input
    out.append(bzd.decompress(b'', max_length=max_length))
    assert not bzd.needs_input
    assert len(out[-1]) == max_length

    # Retrieve more data while providing more input
    out.append(bzd.decompress(BIG_DATA[len_:],
                              max_length=max_length))
    assert len(out[-1]) <= max_length

    # Retrieve remaining uncompressed data
    while not bzd.eof:
        out.append(bzd.decompress(b'', max_length=max_length))
        assert len(out[-1]) <= max_length

    out = b"".join(out)
    assert out == BIG_TEXT
    assert bzd.unused_data == b""




def tearDownModule():
    support.reap_children()




if __name__ == '__main__':
    unittest.main()

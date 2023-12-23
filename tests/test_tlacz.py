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
    with LacFile(filename) as f:
        d = f.read()
    assert d == data1 * 2
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
    with LacFile(filename) as f:
        d = f.read()
    assert d == data1 * 50
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

@pytest.mark.compressed_data
def test_read():
    text = b"Hello world!"
    #data = b'\xfe\xfe\x88<\xe3\x03\x00\x00\xff\xff'
    #data = b"\xfe\xfeN\xef\x14$\xbb\x92\xa1\xbfThat's all, folks!"
    #data = b"\xfe\xfeN\xef\x14$\xb2\xf4T\xeeThat's all, folks!"
    data = b"\xfe\xfeN\xef\x14$\xb2\xa1T\xeaThat's all, folks!"
    with mock_file(data) as f:
        with LacFile(f) as lacf:
            assert lacf.read() == text
    

def test_multi_stream_ordering_without_actions(tmp_path):
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

def test_multi_stream_ordering(tmp_path):
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
    # TEXT = b"Hello world!"
    #DATA = b'\xfe\xfe\x93=\x19\x00W\x00\x19\x00\x0f\x00\x19\x00\x0f\x00\x19\x00\x93=\xff6\x93=\xff6`"\x0e\x00\x1d\xa3\xc6\x00`"\x19\x00W\x00\x19\x00\x10\x00\x19\x00\x10\x00\x19\x00`"\xff6`"\x19\x00\xc6\x00\x9e\x1a\x1e\x1f\x19\x00W\x00\x19\x00\x11\x00\x19\x00\x11\x00\x19\x00\x9e\x1a\x1e\x1f\xff6R\x00`"\x19\x00\xc6\x00D\x01L\x00\x19\x00W\x00\x19\x00\x12\x00\x19\x00\x13\x00\x19\x00D\x01L\x00\xff6i\x1e\x0e\x00D\x01L\x00\x19\x00\xc6\x00\x7f\x86\x19\x00W\x00\x19\x00\x13\x00\x19\x00\x16\x00\x19\x00\x7f\x86\xff6i\x1e\x0e\x00\xd9\x0a\xca\x03\x0e\x00K\x00\x8eu\x19\x00\xc6\x00}j\x19\x00W\x00\x19\x00\x14\x00\x19\x00\x0f\x00\x19\x00}j\xff6R\x00`"\xff6`"\x0e\x00}j\xc6\x00\xd9\xc1V\x0b\x19\x00W\x00\x19\x00\x15\x00\x19\x00\x0f\x00\x19\x00\xd9\xc1V\x0b\xff6R\x00`"\xff6R\x00`"\x0e\x00\xd9\xc1V\x0b\xc6\x00G\x00\xc5\x09\x19\x00W\x00\x19\x00\x16\x00\x19\x00\x0f\x00\x19\x00G\x00\xc5\x09\xff6R\x00`"\xff6R\x00`"\x0e\x00G\x00\xc5\x09\xc6\x00\xb1\x11\x19\x00W\x00\x19\x00\x17\x00\x19\x00)\x04\x19\x00\xb1\x11\xff6i\x1e\x0e\x00\xd9\x0a\xca\x03\x0e\x00\xb1\x11\x19\x00\xc6\x00K*\x19\x00W\x00\x19\x00\x18\x00\x19\x00\xcd\x05\x19\x00K*\xff6i\x1e\x0e\x00\xd9\x0a\xca\x03\x0e\x00K*\x19\x00\xc6\x00T\x00\xcd\x04O\x00\x19\x00W\x00\x19\x00\xac\x03\x19\x00\x87\x05\x19\x00T\x00\xcd\x04O\x00\xff6i\x1e\x0e\x00\xd9\x0a\xca\x03\x0e\x00T\x00\xcd\x04O\x00\x19\x00\xc6\x00\x18\xb6\x19\x00W\x00\x19\x00\x85\x04\x19\x00\x0f\x00\x19\x00\x18\xb6\xff6\x93=\x19\x00\xc6\x00\xfeM\x19\x00W\x00\x19\x00)\x04\x19\x00\xf8\x0b\x19\x00\xfeM\xff6%9\x0e\x00\xfeM\x19\x00\xc6\x00F\x00c"\x19\x00W\x00\x19\x00\xcd\x05\x19\x00\xf6\x04\x19\x00F\x00c"\xff6%9\x0e\x00S\x1f\x0e\x00F\x00c"\x0c\x00\xd2\x1e\x19\x00\xc6\x00\xbd\x02O\x00\x19\x00W\x00\x19\x00\x87\x05\x19\x00`\x04\x19\x00%\x00R\x1c\x0b.\xff6i\x1e\x0e\x00\xbd\x02O\x00\xff6`"\x0e\x00\x1d\xa3\xc6\x00\x88\x88^\x04\x19\x00W\x00\x19\x00L\x8cz\x0a\x19\x00L\x8cz\x0a\x19\x00\xdb`\xff6\xbb+\x19\x00\xc6\x00\xb9\x1c\xf92\x19\x00W\x00\x19\x00\xf8\x0b\x19\x00P"\x19\x00\xb9\x1c\xf92\xff6i\x1e\x0e\x00\xd9\x0a\xca\x03\x0e\x00\xb9\x1c\xf92\x19\x00\xc6\x00\x0d!p\x01\x8f\x0b\x07\x01\x19\x00W\x00\x19\x00\xdb\x0f\x19\x00\xdb\x0f\x80\x0e\x0e\x00\xbb+\x0e\x00\x0d!p\x01\x8f\x0b\x07\x01\xff6`"\x0e\x00\x1d\xa3\xc6\x00\xb9\x1c\xbf\x85\x19\x00W\x00\x19\x00P"\x19\x00\x11<\x19\x00\xa3\x18U\xba\xb4%\xff6i\x1e\x0e\x00S\x1f\x0e\x00\x88\x17Bc\xff6`"\x0e\x00\x1d\xa3\xc6\x00Hp\x986\x19\x00W\x00\x19\x00\x11<\x19\x00];\x19\x00R\x0e\xc5E\x1e\x11\xff6i\x1e\x0e\x00S\x1f\x0e\x00Hp\x986\xff6`"\x0e\x00\x1d\xa3\xc6\x00\xc7\x09\x19\x00W\x00\x19\x00];\x19\x00\x984\x80\x0e\x0e\x00i\x1e\x0e\x00\xc7\x09\xff6`"\x0e\x00Z%\xc6\x00\xff\xff'
    # DATA = b'\xfe\xfe\x88<\xe3\x03\x00\x00\xff\xff'
    #DATA = b'\xfe\xfePK9\xe4\xf1V\xa1\xab\xb3JVd\x0f\x952\xb8\x1a~\xba\x1f\xce\xd7\x11N\xa4\xbe\x07\xee\xb9F\x1d\x1f$\xe1\xab\xdf\xe1m\x8f>z\x92M\x81\x06\x9eO\xb1kV\x868>\xc2\x01n\x90\xc0Ao\xdb\xe7A\x0c\xe2l\xda\x97=\x1e(\x08&\xe6\xa58\xc2\xbf"~\xe4\x0e\xee\x1a\x95^\x15\xe8\x8d7F(#fM\rc\xb5k\xc0\xd0\x0e\xa6\xa3\x16\x8fZ|i\xa2\xd2\xcex\x9a\x92\x83"\xeb\x85\x8aq\x97.\x02\xf1al\xda\xed\xa2\xf4"S\x8a\x0b\xb2\xb8\xca\xeb\xeb9%Ee\xfa\x7f\xf6\x8f\x80\x14\x94\x10\x07}4\x17{\xa3\xbb;\xa7A\xa8VpuG\x936\xee\x96Hh\xfe\x17\xc7\x9eS\x02\xa1\x18\x97Nrr\xd7 \x12,z\x81\xdf\x13~}\xa3\xc8 RY!\xd2\x11d\xb1\xd5\xab1m\xac\x9f\xa4k\x89\xe5\x1f\xfa\x95\xde\xadJ&\x9c\xd4\xbc\xc1\xd3\xff\xa8\n\x06\x1e<\x93\xd5\xb8\x9a\xf2?\xd5\xa0\xc9~\x1cy\rj\xf1\xb3\xf5\xd0\x16eUR\x12\xfc\x94\xe9i\xa9h\x94\xd5\xca-\xd2R\xe9\xf4B[dGM\xaa\xfd\xf4\xf2[\x05\x8a\x12>\xc7\x86p\x9dy&\x88O\xe4\x99\x0f5\xd0\x95\xb4P<\xdf\x8d\x19\x9bf\xc3\xb1\x10\xb9\x8b\x94\x0f\xc6\xe1I\xf0\x994\x18\xa4F\x08\xf2WYa\x19@\x06\x18|8\x0b\xee\'\xa9\xe2\xe4\x87\xd7\xb04\t<0\x86\xd3P"\xa4\xd7\xdf\x1d\x7f\x8d\x8e\x1e\xf5\x90u\xfa\xb7o\xd9\xed\\\xdf\xc6r}\xfa\xdc\xc8s-1z"\x86*\x11\xa2R\xe2\xda\x88\x9b[\xc3\x1f:7\xb2\x01}\x1f\xb6\xb9\x1cp\xb4_}N\xd4\x1ec\x93\x91_ZJ\xd7\xbcKq\rGtN\x0b\xe1\xc9\xabm\x8a\x92\x19[V;\xc5\x12\x7f\xa3kW\xc9\x17\xb5\xdc\x04\xc3\xa3\xdc\xa4\x91cB\xacX\xf0|:\x06\x18\xe5#>)\x7f\xd3\xc6\xa8*$r\xb1\x81\x1c\xca\x03\xfd\x96k\x06\xe8\x132\x12\xb6s\\\xf4\xea\xea\\LP\xd4\xe1\x9a\x1f%\xba\xa5y\x9ft\x15\x1e\x97\x8d\x16\xa3\x97\x87h\x1e\xe7\tG\x81\xb9\x18L\xe6Q\\\xcbg\xc0\x01,\x08\xd8*PP\xbdq\xb5/\x1a^\x16\xa9h!@\xa7\xb9\xb4\x80\x9f#\xd6\x18j\x84\x9f\x17\xa67B\xca\xbc\x0f6\x05=\xcb\xd9\x93\xf8w\x98R\x80\x12\x02yx\x8a\x0f\x05\x9f\xe2&]x\xa83p\xf6\xaa\x00,\xbd!M\x92v\xf1\x1c\xea3|\xc1\x8aDo\x05b\xd2\xe9\x94P7\xc5\xda\xb6\xff\\\x95 \x8c\xa4\x0cy\x16\xa9\xf6\xf9{#\xe4S&\x7f\x8a\x99\xc9\xa5\x11\x16?(\x0c\x95/[\x8eR\xd5tC\x91\xfe9!\x8d\xc7\xb5s;_x\x9e\xb8\x12h\xad]\x8fU\x9b\xa8+\xed\xac\x8e\x8e\x0bZ\x03C\x07\x0f\xeel!\xb5{\x8e\x9c\xd1|8\xa6\xe1\x08\xbdI\x1dq\x98\xf1\x8e\xa7Q\xd0\xf7[b\xe1UV\xd7\xa2\xc2\x14\xb9\x9fy\x97\xb2\x1aE\xc5\xfbq\x98t\x1a\x9c\x7f$A\x8a\xf7\x17\xfa\xa61\xcc\xc9\xe6\xd2\x95_\xe2\xe6\xdbq\xbfZ\x94\xb7U\xd3\x90L\x817+\xb8o\xa6\xff\xe8E\xe94#j\xc3\x1fG\xe4\xd0\x91\x7f\xd6\xf0\xa7\x82\x08That\'s all, folks!'
    #DATA = b'\xfe\xfePK9\xe4\xf1!g\x13\xe9WS\xae\xfa[\xa0\xb5oL\x1d\x00\x1f\x9bK\xa3\xb83\x1b\x86\x81\x96&C\xc9\x85\xf1\x81\xde\x01\x83&\x88\xd5\xf3\xc1^\xa0\xf7\x0f\x91\xfd!;\x01t&\x86\xb6\x7f#\xf4\x86\xe2=b\xc3\x04*:\xf1\xe3\xcf!\xd8xrN\xa3Pv\x17j\x04?\xc6q\x95"\xfa}\xf4\x04-\xfb\xad\xa9=\xe3Y\xf4\xa6\xe8\x8f\x93Y\xfb9/\x1c\x94u~G\xc3\xaft\xfe\xdc[\xedy\xf3rm\x0f\x1e\xff\xc6\xc6\x8cd@#\xe7\x17b\xe4{\x0c\xfc\x81\x1d\x95\\\xcap\xbb\x9eDk\xc5\xb6\xfe\xf0\x17\xa6\xe5\x959\x1b\x82\xf9\x19\xc9\xf0\x1d\xe0\xe4\xe8\\[\xd3\x0e{5s\x90A\xd3b\xf8\xdd\xabF\xbb\xc3\x1a \xdaz\xde\xf2 \x88\xb4\xdd\\\x02\xf7Y\x9dZ~M\xc8\xfa\xc4\xf3N\x82L-3pC\x84\xc6L,@\x1e\xa0wW\xf4\xbd\xb5&\x97\xde\x08\x8cWl\xc8\x98N\x97t\x16K8\xb83x\xb6Z\xa0N\xfc\xfb_\x94\xf4\xe5\n,\x12\x85\xe0\x8b/BVZ? \xb0\xde\x1d\r]m\xd5\x90u.\x85\x86O0t\xe0\xb4\xc7\xa22\xa8\xeb\xe8N)U\x92g\xd4\x82\xdc%r\x9e?#\x92\xcf%a\x00\xd2*\xb6#\x9d\xf4G\xf0\x91g\xc7\x0e\x94W\xb9\x91\xe5K\x1eB\xc4\xd1z\xdfv:\t\x00\xdf*\x07e?\x1f\x88S\x1fl\xe5qQ\xe5\xff\xb8\x1dwZ\xaf\xa7\x95`#\xed\x9c\xb4n\xbb\xac\xce5\xc4\xd6\'C\xa6\x9f\xb4 \xbd\x94"\xe9\xd5]\xbd\x05\xd7\x00TS\x14\x05\xf8\xcb\x9b\xec\xe8k\x87\nV[\xc5\xb6\xae\x87r\xa4\x15lE\xdd\xa9\xa8=9\x88i Y4h\x0c\xfa\x02/\x16s\x9cERKb\xf75\xcb\x06\x8c\x89b\x0eC\xc6w\x9a\xf22\x89\xd0X\xf58\xd3\xe1LUfBF\xb9S\x9b/U\xae#6\x98\xbc\x9a\xae\x99\xce-\xdfS\xea:\xd4[\xfb\xe8\xe6\x81(\xba<&h\xd7NXn&\x10\xcdI-\xd7\x83\xd73B\xbc\xc8\x89b\x87\x80\x17\xab\x92|\xd3)!{t\xb1\x08\x8bs\xfc\x8e\xd1R\xc6\x1dT\xbb\x8d\xcbl\x1e\xe3#\x14\xec\xd3\x80\xd5p\xdc\xd0\xd9\xff\xca\x1ds\x18&\x19\xf6/B\xc9O\xbb\xa5m\x8c\xccY\xa6\xb4\x8a\xc7\xd6\x94\xe0Sa"{\x97P\x89&\xd9\xf3\xdcyHb|g\xf1q\x12\xb3\xa2:\'=U\xdd\x1eZ\x16~\x08\xdbnf"\x06\xe4\x82\xd4\xe9`\xde\xdd\xf7>\x80\x8cw\xa4\x93\'2H\x9ew\xb3\xa5\xdeW31\xf8\xfd$1u!\x03?\x03#\xf6\xe1\x83\x05\xfb\xcb\xff\x82\x80*\x11M\x1a\x00That\'s all, folks!'
    DATA = b'\xfe\xfePK9\xe4\xf0\xccg\x13\xe8sR\xea\xf9s\x9e\x8dk\x0c\x05\xe7\xa9\\\xc9\x1f\xef5J\xa7\x81\x8c\xa6\xabI\x85qf\xdd\xdd\x80F\x84U\xda\xc1:\x9f\xd7\nY\xba\xe4\x05c\xb4\x0e}\xf4R\xbe\xe1\x16\xda\xfc\x16\xc1\xa8\x1bz\xdd\xe2\xeb\x1ej\xd4\xf8\x1d\xd5\x98af\x8b\xe3\xdc\x95\xb0u\xa2\xbb}\xcay-\xea\xac\x9d;\xdf5\xf4d\xa7\x7fi;\x01\xca\x05\x1c\x00P\xa6\x1dcvN\xe1\x08\x9b\xcbm\xf3Z*\xff\x17\xaf\x04\xc5y\xfd\xe6aX\xeb`\xde.\xcabA\x0f\x93\\c\xec\xba\xd5\xa2K5\xb6\x9a\xc3\x17B\xda\x15\x1f\x19\xd2\xe6\xf8p\xf5\x1d"]K\x0e\xf9\x03\r\xa0\x832=]Jq\x90\xdaA\x9a\xb7\x12|\x1f\x8cb\\\xd4\xa2jlu\xb3\xd8\xef\xdcr>*(0\xf8\x07\xd9\x19\x969\\\xe3i\xb7\x04\xb7\xc4<*\x1b@o\xd7T\xb61\x85\xc4,\x0b\x89\xbb1\xa5\x9a?\x95\x88\x15\x9b\x03\xaa\xf7s\x12\xb1\xe0?s;8\x8d\xb4\xe4\xcb\x8c\x12\x85\xb0\x8bWEVdy\xb0\xb0\x9e\x18R\x15i\xb0\xc3\xbdnt\xfa\xcf\x02k \xb4\xc7\xe27\xa6\xc3L`-U\xf2L\x19\xff\xdcg\r\x1es\xe6t\xe1NS\x05\xb6,\xeb\xd7\xa5w\xdb\xf2\xd4\xe9\xc7\xe1Bew\x96\xed\xce\xc7HEh\xa4mK;\t\xd4w@\x13=O6\xe1\xfe\xdf\xb5\xa6\xb2z\xf9\x80\x01\'x\x84|\x94G\x82\xd4 \xeef\xe6\xbc\xd5\xf2\xbb\x87\xfe,1\xb0\xa2/\x88(Dh)\xe2\x0f1\x0fGX\xe6\xe9\xc7\x86\x1d\xf5\\\xed\\ln(V\xa6\rM\xa5`\xbf\xbc\xd5{\xb7\x1d\xab\xeas\x00\xd8l*\x94t|\x11\x90\x811\x05v\xb7\xf1h\x08\x93\x0b\x87\xbb\x0b\x80\xaa\x8f\xf6A\xbeu\xed`2\x9a8X\xf57\xb4\x05L\x00\x10BN\xe6\xd3\x9b\xafU\xae\xa3W.\xbc\xb2\xae\x97\x89<OU\xedY\xb2\xab\x82h\xbd*O\xe1.~fC\xc8@\xa6%\xf7\xc4i!\xcbG2\xe5\xdf.h{\xac\xc5"\x17j\xf4Y\x8a\'\xf6\xe1!\x80\x83%k[)1\x08\x99\xdaG\x89\x8b\xf8\x0b\xb1G"{\x82SYQ\xc7]\\\xb77\xd9\x8f\xbf\x18\xe3\x16R L\x90\x8aT\xa4\n~\xba\xb0\xc6\x9d\x807\xcc\xf7\r$\xb5\xff\x08\xc7\x83\x88#\xf1\xf8\xdb\x8f(\x1fQz\xeeY\x82\x10\xd5u\xc1\x19OO~7\x81\x1b\xe2\xd3lf\xe6\x06\x1a@J\xa3\xb2\x0f\x03\x812\xc0\x17\xe5d$\xb5\xc9\xf8\xc3\xf8\x08\xe4IFc\xa4\xb6|\xab\xb3l\xe7k5\xf3\xcbV\x7f\x817U\xc9\x12J\x85&-H\x99@That\'s all, folks!'

    EMPTY_DATA = b"\xfe\xfe\xff\xffThat's all, folks!"
    BAD_DATA = b'this is not a valid bzip2 file'

    # Some tests need more than one block of uncompressed data. Since one block
    # is at least 100,000 bytes [bzip2], we gather some data dynamically and compress it.
    # Note that this assumes that compression works correctly, so we cannot
    # simply use the bigger test data for all tests.
    test_size = 0
    # Too big for our slowness: BIG_TEXT = bytearray(128*1024)
    BIG_TEXT = bytearray(8*1024)
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

    @pytest.mark.compressed_data
    def test_read(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.read, float())
            assert bz2f.read() == self.TEXT

    def test_read_bad_file(self):
        self.createTempFile(streams=0, suffix=self.BAD_DATA)
        with LacFile(self.filename) as bz2f:
            pytest.raises(OSError, bz2f.read)

    @pytest.mark.compressed_data
    def test_read_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.read, float())
            assert bz2f.read() == self.TEXT * 5

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
    def test_read_trailing_junk(self):
        self.createTempFile(suffix=self.BAD_DATA)
        with LacFile(self.filename) as bz2f:
            assert bz2f.read() == self.TEXT

    @pytest.mark.compressed_data
    def test_read_multi_stream_trailing_junk(self):
        self.createTempFile(streams=5, suffix=self.BAD_DATA)
        with LacFile(self.filename) as bz2f:
            assert bz2f.read() == self.TEXT * 5

    @pytest.mark.compressed_data
    def test_read0(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.read, float())
            assert bz2f.read(0) == b""

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
    def test_read100(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            assert bz2f.read(100) == self.TEXT[:100]

    @pytest.mark.compressed_data
    def test_peek(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pdata = bz2f.peek()
            assert len(pdata) != 0
            assert self.TEXT.startswith(pdata)
            assert bz2f.read() == self.TEXT

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
    def test_read_line(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.readline, None)
            for line in self.TEXT_LINES:
                assert bz2f.readline() == line

    @pytest.mark.compressed_data
    def test_read_line_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.readline, None)
            for line in self.TEXT_LINES * 5:
                assert bz2f.readline() == line

    @pytest.mark.compressed_data
    def test_read_lines(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.readlines, None)
            assert bz2f.readlines() == self.TEXT_LINES

    @pytest.mark.compressed_data
    def test_read_lines_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.readlines, None)
            assert bz2f.readlines() == self.TEXT_LINES * 5

    @pytest.mark.compressed_data
    def test_iterator(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            assert list(iter(bz2f)) == self.TEXT_LINES

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
    def test_seek_forward(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.seek)
            bz2f.seek(150)
            assert bz2f.read() == self.TEXT[150:]

    @pytest.mark.compressed_data
    def test_seek_forward_across_streams(self):
        self.createTempFile(streams=2)
        with LacFile(self.filename) as bz2f:
            pytest.raises(TypeError, bz2f.seek)
            bz2f.seek(len(self.TEXT) + 150)
            assert bz2f.read() == self.TEXT[150:]

    @pytest.mark.compressed_data
    def test_seek_backwards(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            bz2f.read(500)
            bz2f.seek(-150, 1)
            assert bz2f.read() == self.TEXT[500-150:]

    @pytest.mark.compressed_data
    def test_seek_backwards_across_streams(self):
        self.createTempFile(streams=2)
        with LacFile(self.filename) as bz2f:
            readto = len(self.TEXT) + 100
            while readto > 0:
                readto -= len(bz2f.read(readto))
            bz2f.seek(-150, 1)
            assert bz2f.read() == self.TEXT[100-150:] + self.TEXT

    @pytest.mark.compressed_data
    def test_seek_backwards_from_end(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            bz2f.seek(-150, 2)
            assert bz2f.read() == self.TEXT[len(self.TEXT)-150:]

    @pytest.mark.compressed_data
    def test_seek_backwards_from_end_across_streams(self):
        self.createTempFile(streams=2)
        with LacFile(self.filename) as bz2f:
            bz2f.seek(-1000, 2)
            read_result = bz2f.read()
        assert read_result == (self.TEXT * 2)[-1000:]

    @pytest.mark.compressed_data
    def test_seek_post_end(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            bz2f.seek(150000)
            assert bz2f.tell() == len(self.TEXT)
            assert bz2f.read() == b""

    @pytest.mark.compressed_data
    def test_seek_post_end_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            bz2f.seek(150000)
            assert bz2f.tell() == len(self.TEXT) * 5
            assert bz2f.read() == b""

    @pytest.mark.compressed_data
    def test_seek_post_end_twice(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            bz2f.seek(150000)
            bz2f.seek(150000)
            assert bz2f.tell() == len(self.TEXT)
            assert bz2f.read() == b""

    @pytest.mark.compressed_data
    def test_seek_post_end_twice_multi_stream(self):
        self.createTempFile(streams=5)
        with LacFile(self.filename) as bz2f:
            bz2f.seek(150000)
            bz2f.seek(150000)
            assert bz2f.tell() == len(self.TEXT) * 5
            assert bz2f.read() == b""

    @pytest.mark.compressed_data
    def test_seek_pre_start(self):
        self.createTempFile()
        with LacFile(self.filename) as bz2f:
            bz2f.seek(-150)
            assert bz2f.tell() == 0
            assert bz2f.read() == self.TEXT

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
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
        #data = _HEADER + b'Test' + _EOS
        data = b"\xfe\xfeI]\xc3\x0bThat's all, folks!"
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

    @pytest.mark.skip(reason="Too many 1's and we stack overflow in tiktoken.encode")
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

    @pytest.mark.compressed_data
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

    @pytest.mark.skip(reason="Binary data that doesn't utf-8 decode")
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

    @pytest.mark.skip(reason="Binary data that doesn't utf-8 decode")
    def test_open_path_like_filename(self):
        filename = pathlib.Path(self.filename)
        with LacFile(filename, "wb") as f:
            f.write(self.DATA)
        with LacFile(filename, "rb") as f:
            assert f.read() == self.DATA

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
    def test_read_bytes_io(self):
        with BytesIO(self.DATA) as bio:
            with LacFile(bio) as bz2f:
                pytest.raises(TypeError, bz2f.read, float())
                assert bz2f.read() == self.TEXT
            assert not bio.closed

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
    def test_seek_forward_bytes_io(self):
        with BytesIO(self.DATA) as bio:
            with LacFile(bio) as bz2f:
                pytest.raises(TypeError, bz2f.seek)
                bz2f.seek(150)
                assert bz2f.read() == self.TEXT[150:]

    @pytest.mark.compressed_data
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

    @pytest.mark.skip(reason="bz2 specific")
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

    @pytest.mark.compressed_data
    def test_decompress(self):
        bz2d = LacDecompressor()
        pytest.raises(TypeError, bz2d.decompress)
        text = bz2d.decompress(self.DATA)
        assert text == self.TEXT

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
    def test_decompress_unused_data(self):
        bz2d = LacDecompressor()
        unused_data = b"this is unused data"
        text = bz2d.decompress(self.DATA+unused_data)
        assert text == self.TEXT
        assert bz2d.unused_data == unused_data

    @pytest.mark.compressed_data
    def test_eoferror(self):
        bz2d = LacDecompressor()
        text = bz2d.decompress(self.DATA)
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

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
    def test_decompress(self):
        text = lac.decompress(self.DATA)
        assert text == self.TEXT

    def test_decompress_empty(self):
        text = lac.decompress(b"")
        assert text == b""

    def test_decompress_to_empty_string(self):
        text = lac.decompress(self.EMPTY_DATA)
        assert text == b''

    @pytest.mark.compressed_data
    def test_decompress_incomplete(self):
        pytest.raises(ValueError, lac.decompress, self.DATA[:-10])

    def test_decompress_bad_data(self):
        pytest.raises(OSError, lac.decompress, self.BAD_DATA)

    @pytest.mark.compressed_data
    def test_decompress_multi_stream(self):
        text = lac.decompress(self.DATA * 5)
        assert text == self.TEXT * 5

    @pytest.mark.compressed_data
    def test_decompress_trailing_junk(self):
        text = lac.decompress(self.DATA + self.BAD_DATA)
        assert text == self.TEXT

    @pytest.mark.compressed_data
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

    @pytest.mark.compressed_data
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

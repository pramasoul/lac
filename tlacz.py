#!/usr/bin/env python
"""
Compress using a trained model as a predictor
The name ``lac'' is meant to suggest "LLM Arithmetic Coder"
"""

__author__ = "Tom Soulanille <soul@prama.com>"
__homage__author__ = "Nadeem Vawda <nadeem.vawda@gmail.com>"

# A shameless mill swap, with some chassis torch work, of python's
# Lib/bz2.py by Nadeem Vawda. The features are his, the bugs mine.

import lac

from builtins import open as _builtin_open
import io
import logging
import os
import psutil
import sys
import _compression

# # Configure logging
# logging.basicConfig(
#     #level=logging.DEBUG,
#     level=logging.WARNING,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler()],
# )  # StreamHandler logs to console


from lactok_compressor import LACTokCompressor as LacCompressor
from lactok_compressor import LACTokDecompressor as LacDecompressor

_MODE_CLOSED = 0
_MODE_READ = 1
# Value 2 no longer used
_MODE_WRITE = 3


class LacFile(_compression.BaseStream):

    """A file object providing transparent bzip2 (de)compression.

    A BZ2File can act as a wrapper for an existing file object, or refer
    directly to a named file on disk.

    Note that BZ2File provides a *binary* file interface - data read is
    returned as bytes, and data to be written should be given as bytes.
    """

    def __init__(self, filename, mode="r", **kwargs):
        """Open a bzip2-compressed file.

        If filename is a str, bytes, or PathLike object, it gives the
        name of the file to be opened. Otherwise, it should be a file
        object, which will be used to read or write the compressed data.

        mode can be 'r' for reading (default), 'w' for (over)writing,
        'x' for creating exclusively, or 'a' for appending. These can
        equivalently be given as 'rb', 'wb', 'xb', and 'ab'.

        If mode is 'w', 'x' or 'a', compresslevel can be a number between 1
        and 9 specifying the level of compression: 1 produces the least
        compression, and 9 (default) produces the most compression.

        If mode is 'r', the input file may be the concatenation of
        multiple compressed streams.
        """
        logging.debug(f"LacFile({filename=}, {mode=}, {kwargs=})")
        self.kwargs = kwargs
        self.name = str(filename)
        self._fp = None
        self._closefp = False
        self._mode = _MODE_CLOSED

        if mode in ("", "r", "rb"):
            mode = "rb"
            mode_code = _MODE_READ
        elif mode in ("w", "wb"):
            mode = "wb"
            mode_code = _MODE_WRITE
            self._compressor = LacCompressor(**kwargs)
        elif mode in ("x", "xb"):
            mode = "xb"
            mode_code = _MODE_WRITE
            self._compressor = LacCompressor(**kwargs)
        elif mode in ("a", "ab"):
            mode = "ab"
            mode_code = _MODE_WRITE
            self._compressor = LacCompressor(**kwargs)
        else:
            raise ValueError("Invalid mode: %r" % (mode,))

        if isinstance(filename, (str, bytes, os.PathLike)):
            self._fp = _builtin_open(filename, mode)
            self._closefp = True
            self._mode = mode_code
        elif hasattr(filename, "read") or hasattr(filename, "write"):
            self._fp = filename
            self._mode = mode_code
        else:
            raise TypeError("filename must be a str, bytes, file or PathLike object")

        if self._mode == _MODE_READ:
            raw = _compression.DecompressReader(
                self._fp, LacDecompressor, trailing_error=OSError, **kwargs
            )
            self._buffer = io.BufferedReader(raw)
        else:
            self._pos = 0

    def close(self):
        """Flush and close the file.

        May be called more than once without error. Once the file is
        closed, any other operation on it will raise a ValueError.
        """
        if self._mode == _MODE_CLOSED:
            return
        try:
            if self._mode == _MODE_READ:
                self._buffer.close()
            elif self._mode == _MODE_WRITE:
                self._fp.write(self._compressor.flush())
                self._compressor = None
        finally:
            try:
                if self._closefp:
                    self._fp.close()
            finally:
                self._fp = None
                self._closefp = False
                self._mode = _MODE_CLOSED
                self._buffer = None

    @property
    def closed(self):
        """True if this file is closed."""
        return self._mode == _MODE_CLOSED

    def fileno(self):
        """Return the file descriptor for the underlying file."""
        self._check_not_closed()
        return self._fp.fileno()

    def seekable(self):
        """Return whether the file supports seeking."""
        return self.readable() and self._buffer.seekable()

    def readable(self):
        """Return whether the file was opened for reading."""
        self._check_not_closed()
        return self._mode == _MODE_READ

    def writable(self):
        """Return whether the file was opened for writing."""
        self._check_not_closed()
        return self._mode == _MODE_WRITE

    def peek(self, n=0):
        """Return buffered data without advancing the file position.

        Always returns at least one byte of data, unless at EOF.
        The exact number of bytes returned is unspecified.
        """
        self._check_can_read()
        # Relies on the undocumented fact that BufferedReader.peek()
        # always returns at least one byte (except at EOF), independent
        # of the value of n
        return self._buffer.peek(n)

    def read(self, size=-1):
        """Read up to size uncompressed bytes from the file.

        If size is negative or omitted, read until EOF is reached.
        Returns b'' if the file is already at EOF.
        """
        self._check_can_read()
        return self._buffer.read(size)

    def read1(self, size=-1):
        """Read up to size uncompressed bytes, while trying to avoid
        making multiple reads from the underlying stream. Reads up to a
        buffer's worth of data if size is negative.

        Returns b'' if the file is at EOF.
        """
        self._check_can_read()
        if size < 0:
            size = io.DEFAULT_BUFFER_SIZE
        return self._buffer.read1(size)

    def readinto(self, b):
        """Read bytes into b.

        Returns the number of bytes read (0 for EOF).
        """
        self._check_can_read()
        return self._buffer.readinto(b)

    def readline(self, size=-1):
        """Read a line of uncompressed bytes from the file.

        The terminating newline (if present) is retained. If size is
        non-negative, no more than size bytes will be read (in which
        case the line may be incomplete). Returns b'' if already at EOF.
        """
        if not isinstance(size, int):
            if not hasattr(size, "__index__"):
                raise TypeError("Integer argument expected")
            size = size.__index__()
        self._check_can_read()
        return self._buffer.readline(size)

    def readlines(self, size=-1):
        """Read a list of lines of uncompressed bytes from the file.

        size can be specified to control the number of lines read: no
        further lines will be read once the total size of the lines read
        so far equals or exceeds size.
        """
        if not isinstance(size, int):
            if not hasattr(size, "__index__"):
                raise TypeError("Integer argument expected")
            size = size.__index__()
        self._check_can_read()
        return self._buffer.readlines(size)

    def write(self, data):
        """Write a byte string to the file.

        Returns the number of uncompressed bytes written, which is
        always the length of data in bytes. Note that due to buffering,
        the file on disk may not reflect the data written until close()
        is called.
        """
        self._check_can_write()
        if isinstance(data, (bytes, bytearray)):
            length = len(data)
        else:
            # accept any data that supports the buffer protocol
            data = memoryview(data)
            length = data.nbytes

        compressed = self._compressor.compress(data)
        self._fp.write(compressed)
        self._pos += length
        return length

    def writelines(self, seq):
        """Write a sequence of byte strings to the file.

        Returns the number of uncompressed bytes written.
        seq can be any iterable yielding byte strings.

        Line separators are not added between the written byte strings.
        """
        return _compression.BaseStream.writelines(self, seq)

    def seek(self, offset, whence=io.SEEK_SET):
        """Change the file position.

        The new position is specified by offset, relative to the
        position indicated by whence. Values for whence are:

            0: start of stream (default); offset must not be negative
            1: current stream position
            2: end of stream; offset must not be positive

        Returns the new file position.

        Note that seeking is emulated, so depending on the parameters,
        this operation may be extremely slow.
        """
        self._check_can_seek()
        return self._buffer.seek(offset, whence)

    def tell(self):
        """Return the current file position."""
        self._check_not_closed()
        if self._mode == _MODE_READ:
            return self._buffer.tell()
        return self._pos


def open(
        filename, mode="rb", encoding=None, errors=None, newline=None, **kwargs
):
    """Open a bzip2-compressed file in binary or text mode.

    The filename argument can be an actual filename (a str, bytes, or
    PathLike object), or an existing file object to read from or write
    to.

    The mode argument can be "r", "rb", "w", "wb", "x", "xb", "a" or
    "ab" for binary mode, or "rt", "wt", "xt" or "at" for text mode.
    The default mode is "rb", and the default compresslevel is 9.

    For binary mode, this function is equivalent to the BZ2File
    constructor: BZ2File(filename, mode, compresslevel). In this case,
    the encoding, errors and newline arguments must not be provided.

    For text mode, a BZ2File object is created, and wrapped in an
    io.TextIOWrapper instance with the specified encoding, error
    handling behavior, and line ending(s).

    """
    logging.debug(f"open({filename=}, {mode=}, ... {kwargs=})")
    assert "device" in kwargs
    
    if "t" in mode:
        if "b" in mode:
            raise ValueError("Invalid mode: %r" % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if errors is not None:
            raise ValueError("Argument 'errors' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")

    bz_mode = mode.replace("t", "")
    binary_file = LacFile(filename, bz_mode, **kwargs)

    if "t" in mode:
        encoding = io.text_encoding(encoding)
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file


def compress(data, **kwargs):
    """Compress a block of data.

    For incremental compression, use a LacCompressor object instead.
    """
    comp = LacCompressor(**kwargs)
    return comp.compress(data) + comp.flush()


def decompress(data, **kwargs):
    """Decompress a block of data.

    For incremental decompression, use a LacDecompressor object instead.
    """
    results = []
    while data:
        decomp = LacDecompressor(**kwargs)
        try:
            res = decomp.decompress(data)
        except OSError:
            if results:
                break  # Leftover data is not a valid bzip2 stream; ignore it.
            else:
                raise  # Error on the first iteration; bail out.
        results.append(res)
        if not decomp.eof:
            raise ValueError(
                "Compressed data ended before the " "end-of-stream marker was reached"
            )
        data = decomp.unused_data
    return b"".join(results)


# Following taken from Lib/gzip.py and adapted
# gzip.py says at the top:
# based on Andrew Kuchling's minigzip.py distributed with the zlib module


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="A simple command line interface for the lac module: act like gzip, "
        "but do not delete the input file.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true", help="compress faster")
    group.add_argument("--best", action="store_true", help="compress better")
    group.add_argument("-d", "--decompress", action="store_true", help="decompress")

    parser.add_argument("args", nargs="*", default=["-"], metavar="file")
    parser.add_argument("-c", "--stdout", "--to-stdout", action="store_true",
                        help="Write output on standard output; keep input files unchanged")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run the model, e.g 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        # default=4,
        default=psutil.cpu_count(logical=False),
        help="number of threads to use if device is cpu",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="internal",
        help="model to use for prediction",
    )
    parser.add_argument(
        "-T", "--temperature", type=float, default=1.0, help="model's logits scaling"
    )
    parser.add_argument('--log', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity about internals"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="work quietly")

    args = parser.parse_args()
    #sys.stderr.write(f"{args=}\n")

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), None),
        #format="%(asctime)s - %(levelname)s - %(message)s",
        #format="[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        format="%(pathname)s:%(lineno)d %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )  # StreamHandler logs to console

    #sys.stderr.write(f"{logging.root.level=}\n")
    logging.debug(f"{args=}")

    lacfile_args = { 'model_name': args.model,
                      'device': args.device,
                      'threads': args.threads,
    }
                      
    chunk_size = io.DEFAULT_BUFFER_SIZE
    # buggy stdout_chunk_size = 32      # FIXME: hangs if 20 or below (threshold undetermined)
    # At 32, failed ./tlacz.py -c ~/tmp/F.txt --device cuda:2 | ./tlacz.py -d - --device cuda:3
    stdout_chunk_size = 64
    #stdout_chunk_size = chunk_size # DEBUG: hack it to unchanging

    if args.stdout:
        chunk_size = stdout_chunk_size

    for arg in args.args:
        if args.decompress:
            if arg == "-":
                f = LacFile(sys.stdin.buffer,
                            **lacfile_args,
                )
                g = sys.stdout.buffer
                chunk_size = stdout_chunk_size
            else:
                if arg[-5:] != ".lacz":
                    sys.exit(f"filename doesn't end in .lacz: {arg!r}")
                f = open(arg, "rb", **lacfile_args)
                if args.stdout:
                    g = sys.stdout.buffer
                    chunk_size = stdout_chunk_size
                else:
                    g = _builtin_open(arg[:-5], "wb")
        else:
            if arg == "-":
                f = sys.stdin.buffer
                g = LacFile(
                    sys.stdout.buffer,
                    mode="wb",
                    **lacfile_args,
                )
            else:
                f = _builtin_open(arg, "rb")
                if args.stdout:
                    g = LacFile(
                        sys.stdout.buffer,
                        mode="wb",
                        **lacfile_args,
                    )
                else:
                    g = open(arg + ".lacz", "wb", **lacfile_args)
        #sys.stderr.write(f"{chunk_size=}\n")
        _compression.BUFFER_SIZE = chunk_size
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            g.write(chunk)
            g.flush()
        if g is not sys.stdout.buffer:
            g.close()
        if f is not sys.stdin.buffer:
            f.close()


if __name__ == "__main__":
    main()

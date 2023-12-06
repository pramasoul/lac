"""A fake compressor, to assist in validating code structure"""

# Look like zlib.decompressobj or _bz2.BZ2Decompressor

# ChatGPT4 used AskTheCode: I have retrieved the content of the
# Lib/_compression.py file from the Python 3.10 branch. Based on the
# source code, here is an overview of the decompressor API as
# implemented in the DecompressReader class:

# __init__(self, fp, decomp_factory, trailing_error=(),
# **decomp_args): Initializes the DecompressReader. It takes a
# file-like object fp, a decompressor factory decomp_factory, an
# optional trailing_error tuple, and additional decompressor arguments
# **decomp_args.

# readable(self): Returns True, indicating that the DecompressReader
# supports reading.

# close(self): Closes the DecompressReader. It sets the decompressor
# to None and calls the close method of the superclass.

# seekable(self): Returns whether the underlying file-like object is
# seekable.

# readinto(self, b): Reads data into a pre-allocated bytes-like object
# b.

# read(self, size=-1): Reads and returns up to size bytes. If size is
# negative or omitted, reads until EOF.

# readall(self): Reads and returns all data until EOF.

# _rewind(self): Rewinds the file to the beginning of the data stream.

# seek(self, offset, whence=io.SEEK_SET): Seeks to a given offset in
# the file according to the whence parameter.

# tell(self): Returns the current file position.

import io
import logging

BUFFER_SIZE = io.DEFAULT_BUFFER_SIZE  # Compressed data read chunk size
_HEADER = b'\x00'
_EOF_BYTE = b'\xff'

# def make_fake_decompressor():
#     """A factory for fake decompressors"""
#     rv = FakeDecompressor()
#     logging.debug(f"make_fake_decompressor()={rv}")
#     return rv

class FakeDecompressor:
    """A fake decompressor"""

    def __init__(self, compression_level=9):
        # This is bz2-specific:
        if compression_level < 0 or compression_level > 9:
            raise TypeError
        logging.debug(f"FakeDecompressor.__init__()")
        self.header_seen = False
        self.unused_data = b""
        self._eos = False

    @property
    def needs_input(self):
        return len(self.unused_data) == 0
        #return not self._eos

    @property
    def eof(self):
        # return self._eos and len(self.unused_data) == 0
        return self._eos

    # @property
    # def unconsumed_tail(self):
    #     # FIXME: is this vestigial to gzip/zlib?
    #     # return self.unused_data
    #     return b""

    def decompress(self, rawblock, size=None, max_length=None):
        logging.debug(f"FakeDecompressor.decompress({len(rawblock)=}, {size=}) {self.header_seen=} {len(self.unused_data)=}")
        rawblock = self.unused_data + rawblock
        self.unused_data = None # tripwire if we don't update
        if size is None:
            size = len(rawblock)
        if not self.header_seen and size > 0:
            # Would need to accumulate a header if it were more than one byte
            if rawblock.startswith(_HEADER):
                rawblock = rawblock[len(_HEADER):]
                self.header_seen = True
            else:
                raise OSError
        ep = rawblock.find(_EOF_BYTE)
        if ep == -1 or ep >= size:
            rv = rawblock[:size]
            self.unused_data = rawblock[size:]
        else:
            rv = rawblock[:ep]
            self.unused_data = rawblock[ep+1:] # skip the end-of-stream flag byte
            self._eos = True
        logging.debug(f"FakeDecompressor.decompress {len(rv)=} {self.header_seen=} {len(self.unused_data)=}")
        return rv
    
    def __getstate__(self, *args, **kwargs):
        raise TypeError
    
class FakeCompressor:
    """A fake compressor"""
    def __init__(self, compression_level=9):
        # This is bz2-specific:
        if compression_level < 0 or compression_level > 9:
            raise TypeError
        self.header_sent = False

    def compress(self, data, *args, **kwargs):
        logging.debug(f"FakeCompressor.compress({len(data)=})")
        if not self.header_sent:
            logging.debug(f"FakeCompressor.compress including header")
            rv = _HEADER + data
            self.header_sent = True
        else:
            rv = data
        return rv

    def flush(self, *args):
        logging.debug(f"FakeCompressor.flush({args=})")
        return _EOF_BYTE
    
    def __getstate__(self, *args, **kwargs):
        raise TypeError

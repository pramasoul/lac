"""A tokenizing compressor with arithmetic coding, to assist in validating code structure"""

# The objective is to look like _bz2.BZ2Decompressor
# https://www.cs.cmu.edu/afs/cs/project/pscico-guyb/realworld/99/code/bzip2-0.9.5c/manual_3.html

import hashlib
import io
import json
import logging
import numpy as np
import re
import struct
import subprocess
import sys
import tiktoken
import torch
import zlib

import numpy as np

from typing import List, Tuple

from lac_llm import provide_model, provide_prediction_service
from lac_llm import PredictionService, FlatPredictionService, CountingPredictionService
from ac2_for_z import PDFPredictor, A_to_bin, A_from_bin

from config import SingletonConfig
config = SingletonConfig()
#from dataclasses import dataclass

#@dataclass
class Thing:
    pass

BUFFER_SIZE = io.DEFAULT_BUFFER_SIZE  # Compressed data read chunk size
_EOS = b"That's all, folks!"    # DEBUG
PRECISION = 48                        # The arithmetic coder's arithmetic precision to use

__version_bytes__ = bytes([0, 1])
__version__ = f"{'.'.join(str(int(b)) for b in __version_bytes__)}"


def magic_number_bytes():
    bs = b"LACZ"
    # bit-reverse the bytes
    rv = bytes([int(bin(i | 256)[3:][::-1], 2) for i in bs])
    try:
        rv.decode()
    except UnicodeDecodeError:
        # This is where we want to be
        pass
    else:
        raise ValueError(
            "Our magic number is UTF decodable, which risks collision with real text files"
        )
    finally:
        return rv


def get_nvcc_version_info() -> dict[str]:
    # FIXME: is nvcc necessarily present on any machine that can run our code? NO, e.g. cpu
    try:
        output = subprocess.check_output("nvcc --version", shell=True).decode()
    except subprocess.CalledProcessError:
        release_info = build_info = None
    else:
        # Regular expression to match the release and build information
        release_pattern = r"release (\d+\.\d+), V(\d+\.\d+\.\d+)"
        build_pattern = r"Build (cuda_\d+\.\d+\.r\d+\.\d+/compiler\.\d+_\d+)"

        # Search for matches in the output
        release_match = re.search(release_pattern, output)
        build_match = re.search(build_pattern, output)

        # Extract the matched groups if found
        release_info = release_match.group(2) if release_match else None
        build_info = build_match.group(1) if build_match else None

    return release_info, build_info

def get_python_version_number_string():
    return sys.version.split()[0]


def get_versions() -> dict[str]:
    sys_cuda_release, sys_cuda_build = get_nvcc_version_info()
    rv = {
        "lacz": __version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "sys_cuda_build": sys_cuda_build,
        "sys_cuda_release": sys_cuda_release,
        "python": get_python_version_number_string(),
        "np": np.__version__,
    }
    return rv


# FIXME: decide on a header dictionary. Changes are breaking and will require
# versioning and keeping all the prior ones available to interpret old headers.
# header_zdict = b'{"lacz": "0.1", "torch": "2.0.1", "cuda": "11.8", "cudnn": 8700,'
# ' "sys_cuda_build": "cuda_12.2.r12.2/compiler.33191640_0",'
# ' "sys_cuda_release": "12.2.140", "python": "3.10.11", "np": "1.24.3"}'

# Use an empty dict for now
header_zdict = b""

def lacz_header(compressor) -> bytes:
    rv = []
    rv.append(magic_number_bytes())
    rv.append(__version_bytes__)
    d = { "versions": get_versions(),
          "encoding_name": compressor.encoding_name,
          "model_name": compressor.model_name,
          "device": compressor.device,
          "threads": compressor.threads,
          "temperature": compressor.temperature,
    }
    zc = zlib.compressobj(zdict=header_zdict)
    vjz = zc.compress(json.dumps(d).encode("utf-8")) + zc.flush()
    rv.append(struct.pack("!H", len(vjz)))  # prepend the length
    rv.append(vjz)
    rv = b"".join(rv)
    return rv

lacz_cleartext_hasher = hashlib.sha256
# class myhash:
#     def __init__(self):
#         self.h = hashlib.sha256()
#         self.n_seen = self.n_updates = 0

#     def update(self, v):
#         self.n_updates += 1
#         self.n_seen += len(v)
#         logging.info(f"{self.n_updates} hashing {len(v)}: {v[:16]}...\n")
#         self.h.update(v)

#     def digest(self):
#         return self.h.digest()

#     def hexdigest(self):
#         return self.h.hexdigest()

# lacz_cleartext_hasher = myhash

lacz_cleartext_digest_len = len(lacz_cleartext_hasher().digest())
lacz_footer_len = len(_EOS) + lacz_cleartext_digest_len


class TokPredictor(PDFPredictor):
    def __init__(self, max_token, prediction_service: PredictionService, precision=PRECISION):
        logging.debug(f"TokPredictor({max_token=}, {precision=})")
        assert 1 < precision < 64 #uint64 limit
        super().__init__(None, precision)
        self.max_token = max_token
        self.prediction_service = prediction_service
        self.restart()

    def __repr__(self):
        # max_tok_i = np.argmax(self.pdf)
        # max_tok_p = self.pdf[max_tok_i]
        # return "\n".join([f"TokPredictor: {self.max_token}, {self.precision} {self.accepts=} pdf {max_tok_i}: {max_tok_p}",
        #                  f"{sum(self.pdf)=}, {sum(self.pdf)-len(self.pdf)-self.accepts=}"])
        return f"TokPredictor: {self.max_token}, {self.precision} {self.accepts=}"
    
    def restart(self):
        self.prediction_service.restart()
        self.set_cdf_from_pdf(self.prediction_service.probabilities.cpu())
        self.accepts = 0

    def accept(self, tok):
        assert 0 <= tok <= self.max_token
        self.prediction_service.accept(tok)
        self.set_cdf_from_pdf(self.prediction_service.probabilities.cpu())
        self.accepts += 1


class LACTokCompressor:
    """Arithmetic Coding tok compressor"""

    def __init__(self,
                 encoding_name="gpt2",
                 model_name="internal",
                 #tok_mode="line-by-line",
                 tok_mode="all but last partial line",
                 device="cpu",
                 threads=1,
                 temperature=1.0,
                 save_toks=False):
        self.encoding_name = encoding_name
        self.model_name = model_name
        self.device = device
        self.threads = threads
        self.temperature = temperature

        self.tok_enc = tiktoken.get_encoding(encoding_name)
        self.eot_token = self.tok_enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        self.tok_mode = tok_mode #DEBUG
        if self.tok_mode == "buffer minimum for correct":
            self.tok_max = max(len(self.tok_enc.decode([i])) for i in range(self.tok_enc.n_vocab))
        logging.debug(f"LAXTokCompressor calling provide_prediction_service({model_name=}, {device=}, {threads=}, {temperature=})\n")
        if "mem" in config.debug:
            import pdb
            pdb.set_trace()
        self.predictor = TokPredictor(self.eot_token, provide_prediction_service(model_name=model_name,
                                                                                 device=device,
                                                                                 threads=threads,
                                                                                 temperature=temperature,
        ))
        logging.debug(f"LACTokCompressor: {encoding_name} <|endoftext|> is {self.eot_token}")
        self.a2b = A_to_bin(self.predictor, PRECISION)
        self.input_accumulator = ""
        self.cleartext_hasher = lacz_cleartext_hasher()
        self.bits_accumulator = []
        self.header_sent = False

        self.debug_save_toks = save_toks
        if save_toks:
            self.toks = []

    def __repr__(self):
        return f"comp({self.encoding_name} {self.predictor} {' H'[self.header_sent]},{len(self.input_accumulator)}({self.input_accumulator[:4]}) -> {len(self.bits_accumulator)}({''.join(str(v) for v in self.bits_accumulator[:8])})"

    def compress(self, data, *args, **kwargs):
        logging.debug(f"LACTokCompressor.compress({len(data)=}) {self=}")

        # Ensure data is in text format
        if isinstance(data, (bytes, bytearray)):
            try:
                data = data.decode('utf-8')  # Decode using UTF-8 or another appropriate encoding
            except UnicodeDecodeError:
                raise ValueError("Byte data provided is not in valid UTF-8 format")
        elif not isinstance(data, str):
            raise TypeError("Data must be either a string, bytes, or bytearray")

        logging.debug(f"Compress hash {data.encode('utf-8')}:")
        self.cleartext_hasher.update(data.encode('utf-8'))
        self.input_accumulator += data
        if not self.header_sent:
            logging.debug(f"LACTokCompressor.compress including header in rv")
            rv = self._header()
            self.header_sent = True
        else:
            rv = b''

        # Without reading the entire file before tokenizing, we cannot
        # trust the tokenization at the margins of each buffer of text
        # we tokenize.
        # FIXME: for now we trust that there will be line breaks often
        # enough, and that a line break is a guarantee of token break.

        toks = []
        # DEBUG
        #tok_mode = "hold all until flush"
        if self.tok_mode == "line-by-line":
            # FIXME: is below still true?
            # old way, buggy, but don't have good explanation.
            # Fails on test_atok_compressor.py::test_like_tlacz_write_read_with_pathlike_file
            nl_split = self.input_accumulator.split('\n')
            for line in nl_split[:-1]:
                toks.extend(self.tok_enc.encode(line + '\n'))
            self.input_accumulator = nl_split[-1]
        elif self.tok_mode == "all but last partial line":
            nl_split =  self.input_accumulator.rsplit('\n', maxsplit=1)
            if len(nl_split) == 2:
                if nl_split[1]:
                    toks.extend(self.tok_enc.encode(nl_split[0] + '\n'))
                    self.input_accumulator = nl_split[1]
            # else no text past last newline (if any) yet
        elif self.tok_mode == "hold all until flush":
            pass
        elif self.tok_mode == "buffer minimum for correct":
            if len(self.input_accumulator) >= self.tok_max:
                for t in self.tok_enc.encode(self.input_accumulator):
                    toks.append(t)
                    self.input_accumulator = self.input_accumulator[len(self.tok_enc.decode([t])):]
                    if len(self.input_accumulator) < self.tok_max:
                        break


        try:
            eot_ix = toks.locate(self.eot_token)
        except:
            eot_ix = None
        logging.debug(f"{len(toks)=}, {toks[-10:]=}")
        if self.debug_save_toks:
            self.toks.append(toks)
        self.bits_accumulator.extend(list(self.a2b.bits(toks, stop=False)))
        logging.debug(f"LACTokCompressor.compress {self=}")
        v, self.bits_accumulator = bits_to_bytes(self.bits_accumulator)
        rv += v
        logging.debug(f"LACTokCompressor.compress {self=} {len(rv)=} {rv[:16]}[...]{rv[-16:]} {eot_ix=}")
        return rv


    def flush(self, *args):
        logging.debug(f"LACTokCompressor.flush({self=}, {args=})")
        toks = self.tok_enc.encode(self.input_accumulator)
        toks.append(self.eot_token)

        # DEBUG:
        try:
            eot_ix = toks.index(self.eot_token)
        except ValueError:
            eot_ix = None
        logging.debug(f"LACTokCompressor.flush {self=} {toks=} {eot_ix=}")

        if self.debug_save_toks:
            self.toks.append(toks)
        self.bits_accumulator.extend(list(self.a2b.bits(toks, stop=True)))

        if not self.header_sent:
            rv = self._header()
            self.header_sent = True
        else:
            rv = b""

        v, self.bits_accumulator = bits_to_bytes(self.bits_accumulator + [0]*7) # FIXME? or ok?
        rv += v
        logging.debug(f"LACTokCompressor.flush {self=} {rv=} {self.bits_accumulator=}")
        assert all(v == 0 for v in self.bits_accumulator)
        self.bits_accumulator = [] # discard padding
        logging.debug(f"{rv=}")
        rv += self._footer()
        logging.debug(f"{rv=}")
        logging.debug(f"LACTokCompressor.flush {self=} {len(rv)=} {rv[:16]}[...]{rv[-16:]}")
        return rv

    def _header(self):
        return lacz_header(self)

    def _footer(self):
        # FIXME
        logging.debug(f"_footer: hasher digest 0x{self.cleartext_hasher.hexdigest()}")
        return _EOS + self.cleartext_hasher.digest()

    def __getstate__(self, *args, **kwargs):
        raise TypeError  # Can't pickle us


class LACTokDecompressor:
    """A tok decompressor"""

    def __init__(self,
                 encoding_name="gpt2",
                 model_name="internal",
                 device="cpu",
                 threads=1,
                 temperature=1.0,
                 save_toks=False):
        logging.debug(f"LACTokDecompressor.__init__({encoding_name=})")
        if not isinstance(encoding_name, (str, bytes)):
            raise TypeError(f"encoding name is a {type(encoding_name)} want string")
        self.encoding_name = encoding_name
        self.model_name = model_name
        self.device = device
        self.threads = threads  # FIXME: need this?
        self.temperature = temperature

        self.tok_enc = tiktoken.get_encoding(encoding_name)
        self.eot_token = self.tok_enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        logging.debug(f"LAXTokecmpressor calling provide_prediction_service({model_name=}, {device=}, {threads=}, {temperature=})\n")
        assert device is not None
        self.predictor = None   # Defer creation until we've parsed the header
        logging.debug(f"LACTokDecompressor: {encoding_name} <|endoftext|> is {self.eot_token}")
        self.dtype = np.dtype('<u2')
        self._restart()

        self.debug_save_toks = save_toks
        if self.debug_save_toks:
            self.toks = []

    def _restart(self):
        self.n_bytes_ingested = 0
        self.header_seen = False
        self.unused_data = b""
        self.token_buffer = []
        self.output_buffer = b""
        logging.debug(f"LacTokDecompressor._restart getting new cleartext_hasher")
        self.cleartext_hasher = lacz_cleartext_hasher() # FIXME: create it here? where?
        self.predictor and self._restart_AC()
        self._eos = False
        self.state = "Expecting header"

    def _restart_AC(self):
        self.predictor.restart()
        self.b2a = A_from_bin(self.predictor, PRECISION)        


    def __repr__(self):
        return f"decomp({self.state=} {self.needs_input=}, {self._eos=} {self.eof=}, {len(self.unused_data)=}({self.unused_data[:4]}),{len(self.token_buffer)=}({self.token_buffer[:4]}...{self.token_buffer[-4:]})) -> {len(self.output_buffer)=}({self.output_buffer[:4]}...{self.output_buffer[-4:]}),{self.n_bytes_ingested=})"
        
    @property
    def needs_input(self):
        return len(self.output_buffer) == 0 \
            and self.state not in ("Expecting footer", "Footer good")
            
    @property
    def eof(self):
        # return False
        if self.output_buffer == b"":
            return self._eos
        else:
            return False

    def decompress(self, rawblock, size=None, max_length=None):
        """ """

        logging.debug(
            f"LACTokDecompressor.decompress({self=}, {len(rawblock)=}, {size=}, {max_length=})"
        )

        # FIXME: what is size? how is it to be used?
        if size is not None:
            if max_length is not None:
                max_length = min(size, max_length)
            else:
                max_length = size

        # If restarting, clear unused data, etc
        if self.eof:
            self._restart()

        # Two phases:
        # 1) Ingest and process the rawblock of compressed bytes
        # 2) Deliver the requested amount of data, or whatever we can
        # Phase 1 begins here:

        self.unused_data += rawblock
        self.n_bytes_ingested += len(rawblock)
        
        # The state machine
        #NO: while self.unused_data: # adds complexity handled by higher levels already
        if self.state == "Footer good":
            self._restart_AC()
            self.state = "Expecting header"

        if self.state == "Expecting header":
            if self.check_for_header_and_process():
                self.state = "Expecting data or footer"
                if self.predictor is None: # We deferred creating the predictor until we had the header info
                    self.predictor = TokPredictor(self.eot_token,
                                                  provide_prediction_service(model_name=self.model_name,
                                                                             device=self.device,
                                                                             threads=self.threads,
                                                                             temperature=self.temperature,
                                                  ))
                    self._restart_AC()
        if self.state == "Expecting data or footer":
            new_toks, n_bits = tokens_to_stop_with_bits_consumed(self.b2a, self.unused_data, self.eot_token)
            if self.debug_save_toks:
                self.toks.append(new_toks)
            try:
                eot_ix = new_toks.index(self.eot_token)
            except ValueError:
                eot_ix = None
            logging.debug(f"LACTokDecompressor.decompress {self} {eot_ix=}")
            self.unused_data = self.unused_data[(n_bits+7)//8:] # If we used any bits from that last byte we skip
            if new_toks and new_toks[-1] == self.eot_token:
                logging.debug(f"LACTokDecompressor.decompress hit eot token")
                assert new_toks.pop() == self.eot_token
                self.state = "Expecting footer"
            logging.debug(f"LACTokDecompressor.decompress {self} {eot_ix=}")
            self.token_buffer.extend(new_toks)
            logging.debug(f"LACTokDecompressor.decompress {self} {eot_ix=}")

            # Now turn tokens into text in output buffer
            decoded_toks = self.tok_enc.decode_bytes(self.token_buffer)
            self.token_buffer = []
            self.output_buffer += decoded_toks
            logging.debug(f"LACTokDecompressor.decompress {self=}, {decoded_toks=}")

        if self.state == "Expecting footer":
            # We may have some output that hasn't been returned
            # and will need to get that out before checking the hash in the footer
            if not self.output_buffer or True:
                self.state = "Looking for footer"

        if self.state == "Looking for footer":
            check_result = self.check_for_footer_and_process()
            logging.debug(f"{check_result=}")
            if check_result == "good":
                self.state = "Footer good"
            elif check_result == "looking":
                pass            # Stay in "Looking for footer" state
            else:
                raise ValueError(f"bad footer: {check_result}")
            logging.debug(f"LACTokDecompressor.decompress {self=}")

        # if self.state == "Footer good":
        #     self._restart_AC()
        #     self.state = "Expecting header"
        #     logging.debug(f"LACTokDecompressor.decompress {self=}")

        # Phase 2: Provide output
        # Now provide output, respecting max_length
        if max_length is None:
            rv = self.output_buffer
            self.output_buffer = b""
        else:
            rv = self.output_buffer[:max_length]
            self.output_buffer = self.output_buffer[max_length:]
        logging.debug(f"LACTokDecompressor.decompress {self} {len(rv)=}({rv[:16]}...)")

        # Update the hash of the decompressed data
        if rv:
            logging.debug(f"Decompress hash {rv}: ")
            self.cleartext_hasher.update(rv)

            # If we have seen all the output, and the header, we can check the hashes
            if self.output_buffer == b"" and self.state == "Footer good":
                decompression_digest = self.cleartext_hasher.digest()
                if self.source_digest == decompression_digest:
                    logging.debug(f"Hashes agree")
                else:
                    raise OSError(
                        f"Original file had hash of 0x{self.source_digest.hex()}"
                        f" but this decompression is 0x{decompression_digest.hex()}"
                    )
        return rv


    def check_for_header_and_process(self):
        # lac header starts with four magic bytes (b'2\x82\xc2Z'),
        # two version bytes,
        # packed uint16 length of rest of header
        # Until we have that, we don't have the header
        if not hasattr(self, "header_magic"):
            self.header_magic_bytes = magic_number_bytes()
        if not self.unused_data.startswith(self.header_magic_bytes[:len(self.unused_data)]):
            raise OSError("Bad magic bytes in header")
        if len(self.unused_data) < 8:
            return False
        zjson_header_len = struct.unpack("!H", self.unused_data[6:8])[0]
        # if zjson_header_len > 1024: complain
        if len(self.unused_data) < 8 + zjson_header_len:
            return False
        # Here we have the expected number of bytes to decode the header
        self.header = Thing()
        self.header.version_bytes = self.unused_data[4:6]
        vjz = self.unused_data[8:8+zjson_header_len]
        zd = zlib.decompressobj(zdict=header_zdict)
        header_dict = json.loads(zd.decompress(vjz) + zd.flush())
        logging.debug(f"Header {self.header.version_bytes} {header_dict}")
        for k in ( "encoding_name",
                   "model_name",
                   #"device",
                   "threads",
                   "temperature",
        ):
            if k in header_dict:
                self.__dict__[k] = header_dict[k]

        # If compressed with some cuda device, any cuda will do (FIXME: verify)
        # If a cuda device is already specified, use that one
        if hd := header_dict.get("device"):
            if hd.startswith("cuda") and not self.device.startswith("cuda"):
                self.device = "cuda"
            else:
                self.device = hd

        self.unused_data = self.unused_data[8+zjson_header_len:]
        return True


    def check_for_footer_and_process(self):
        footer_start = self.unused_data.find(_EOS)
        logging.debug(f"check_for_footer_and_process: {footer_start=}")
        if footer_start > 0:
            logging.warning(f"Expected footer right here, but it's {footer_start} bytes further")
            return f"displaced to start at {footer_start}"
        if footer_start == -1:
            if len(self.unused_data) >= len(_EOS):
                return "not found"
            else:
                return "looking"
        if footer_start == 0:
            if len(self.unused_data) < len(_EOS) + lacz_cleartext_digest_len:
                return "looking"
            self.source_digest = self.unused_data[footer_start + len(_EOS) : footer_start + len(_EOS) + lacz_cleartext_digest_len]
            self.unused_data = self.unused_data[footer_start + lacz_footer_len:]
            self._eos = True    # FIXME?
            return "good"


    def __getstate__(self, *args, **kwargs):
        raise TypeError


# Chat4, tweaked (with my bugfix setting dtype):
def bits_to_bytes(bitstream: List[int]): #-> Tuple(bytearray, List[int]):
    # Separate out any tail of the bitstream that doesn't take a full byte
    # We'll pass it back
    cut_point = (len(bitstream)//8) * 8
    bitstream, tail = bitstream[:cut_point], bitstream[cut_point:]

    # Convert the list to a NumPy array
    bit_array = np.array(bitstream, dtype='uint8')
    assert np.all((bit_array >= 0) & (bit_array <= 1)), "Some bits in call to bits_to_bytes were improper"

    if len(bit_array) % 8 != 0:
        raise ValueError("Length of bitstream should be a multiple of 8")

    # Reshape the array into 8 columns (each row represents a byte)
    reshaped_array = bit_array.reshape(-1, 8)

    # Convert bits to byte values
    byte_values = reshaped_array.dot(1 << np.arange(8, dtype='uint8')[::-1])

    # Convert to bytes or bytearray
    byte_array = bytearray(byte_values)

    return byte_array, tail

# Chat4. (I bet there's a numpy way that's faster)
def bytes_to_bits(byte_data):
    bitstream = []

    for byte in byte_data:
        # Convert each byte to a binary string, then to a list of bits
        bits = [int(bit) for bit in format(byte, '08b')]
        bitstream.extend(bits)

    return bitstream

# Me:
def tokens_to_stop_with_bits_consumed(decoder, bs, stop_token=None):
    masks = [1<<p for p in range(7, -1, -1)]
    acc = []
    n = 0
    for byte in bs:
        for mask in masks:
            bit = bool(byte & mask)
            n += 1
            vt = decoder(bit)
            for v in vt:
                acc.append(v)
                if v == stop_token:
                    return acc, n # discarding any tokens after stop
    return acc, n

################################################################
# Debugging aids

def mem_used():
    import os
    import psutil

    rv = Thing()

    # Get the current process
    rv.pid = process = psutil.Process(os.getpid())

    # Virtual memory usage
    rv.vm = irtual_memory_usage = process.memory_info().vms  # in bytes

    # Resident Set Size (physical memory usage)
    rv.rss = resident_memory_usage = process.memory_info().rss  # in bytes

    total_virtual_memory = virtual_memory_usage
    total_resident_memory = resident_memory_usage

    # Include child processes
    for child in process.children(recursive=True):
        child_memory_info = child.memory_info()
        total_virtual_memory += child_memory_info.vms
        total_resident_memory += child_memory_info.rss

    rv.total_vm = total_virtual_memory
    rv.total_rss = total_resident_memory

    return rv

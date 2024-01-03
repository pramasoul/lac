"""A tokenizing compressor with arithmetic coding, to assist in validating code structure"""

# The objective is to look like _bz2.BZ2Decompressor
# https://www.cs.cmu.edu/afs/cs/project/pscico-guyb/realworld/99/code/bzip2-0.9.5c/manual_3.html

import io
import logging
import numpy as np
import struct
import tiktoken

from typing import List, Tuple

from lac_llm import provide_model, provide_prediction_service
from lac_llm import PredictionService, FlatPredictionService, CountingPredictionService
from ac2_for_z import PDFPredictor, A_to_bin, A_from_bin

from config import SingletonConfig


BUFFER_SIZE = io.DEFAULT_BUFFER_SIZE  # Compressed data read chunk size
_HEADER = b"\xfe\xfe"                 # Beyond token range
_EOS = b"That's all, folks!"    # DEBUG
PRECISION = 48                        # The arithmetic coder's arithmetic precision to use

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
        self.set_cdf_from_pdf(self.prediction_service.probabilities)
        self.accepts = 0

    def accept(self, tok):
        assert 0 <= tok <= self.max_token
        self.prediction_service.accept(tok)
        self.set_cdf_from_pdf(self.prediction_service.probabilities)
        self.accepts += 1


class LACTokCompressor:
    """Arithmetic Coding tok compressor"""

    def __init__(self,
                 encoding_name="gpt2",
                 model_name="internal",
                 tok_mode="line-by-line",
                 device="cpu",
                 threads=1,
                 save_toks=False):
        self.encoding_name = encoding_name
        self.tok_mode = tok_mode #DEBUG
        self.tok_enc = tiktoken.get_encoding(encoding_name)
        if self.tok_mode == "buffer minimum for correct":
            self.tok_max = max(len(self.tok_enc.decode([i])) for i in range(self.tok_enc.n_vocab))
        self.eot_token = self.tok_enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        logging.info(f"LAXTokCompressor calling provide_prediction_service({model_name=}, {device=}, {threads=})\n")
        self.predictor = TokPredictor(self.eot_token, provide_prediction_service(model_name=model_name,
                                                                                 device=device,
                                                                                 threads=threads))
        logging.debug(f"LACTokCompressor: {encoding_name} <|endoftext|> is {self.eot_token}")
        self.a2b = A_to_bin(self.predictor, PRECISION)
        self.input_accumulator = ""
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
        return _HEADER

    def _footer(self):
        # FIXME
        return _EOS
        #return b"That's all, folks!"

    def __getstate__(self, *args, **kwargs):
        raise TypeError  # Can't pickle us


class LACTokDecompressor:
    """A tok decompressor"""

    def __init__(self,
                 encoding_name="gpt2",
                 model_name="internal",
                 device="cpu",
                 threads=1,
                 save_toks=False):
        logging.debug(f"LACTokDecompressor.__init__({encoding_name=})")
        if not isinstance(encoding_name, (str, bytes)):
            raise TypeError(f"encoding name is a {type(encoding_name)} want string")
        self.encoding_name = encoding_name
        self.tok_enc = tiktoken.get_encoding(encoding_name)
        self.eot_token = self.tok_enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        logging.debug(f"LAXTokecmpressor calling provide_prediction_service({model_name=}, {device=}, {threads=})\n")
        assert device is not None
        self.predictor = TokPredictor(self.eot_token, provide_prediction_service(model_name=model_name,
                                                                                 device=device,
                                                                                 threads=threads))
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
        self._restart_AC()
        self._eos = False
        self.state = "Expecting header"

    def _restart_AC(self):
        self.predictor.restart()
        self.b2a = A_from_bin(self.predictor, PRECISION)        


    def __repr__(self):
        return f"decomp({self.state=} {self.needs_input=}, {self._eos=} {self.eof=},{len(self.unused_data)=}({self.unused_data[:4]}),{len(self.token_buffer)=}({self.token_buffer[:4]}...{self.token_buffer[-4:]})) -> {len(self.output_buffer)}({self.output_buffer[:4]}...{self.output_buffer[-4:]}),{self.n_bytes_ingested})"
        
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

        # If restarting, clear unused data
        if self.eof:
            self._restart()

        # Two phases:
        # 1) Ingest and process the rawblock of compressed bytes
        # 2) Deliver the requested amount of data, or whatever we can
        # Phase 1 begins here:

        self.unused_data += rawblock
        self.n_bytes_ingested += len(rawblock)
        
        # The state machine
        #while self.unused_data: # adds complexity handled by higher levels already
        if self.state == "Expecting header":
            if self.check_for_header_and_process():
                self.state = "Expecting data or footer"

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
            logging.debug(f"LACTokDecompressor.decompress {self=}")

            # Now turn tokens into text in output buffer
            decoded_toks = self.tok_enc.decode_bytes(self.token_buffer)
            self.token_buffer = []
            self.output_buffer += decoded_toks
            logging.debug(f"LACTokDecompressor.decompress {self=}, {decoded_toks=}")

        if self.state == "Expecting footer":
            check_result = self.check_for_footer_and_process()
            logging.debug(f"{check_result=}")
            if check_result == "good":
                self.state = "Footer good"
            elif check_result == "looking":
                pass            # Stay in "Expecting footer" state
            else:
                raise ValueError(f"bad footer: {check_result}")
            logging.debug(f"LACTokDecompressor.decompress {self=}")

        if self.state == "Footer good":
            self._restart_AC()
            self.state = "Expecting header"
            logging.debug(f"LACTokDecompressor.decompress {self=}")

        # Phase 2: Provide output
        # Now provide output, respecting max_length
        if max_length is None:
            rv = self.output_buffer
            self.output_buffer = b""
        else:
            rv = self.output_buffer[:max_length]
            self.output_buffer = self.output_buffer[max_length:]
        logging.debug(f"LACTokDecompressor.decompress {self} {len(rv)=}({rv[:16]}...)")
        return rv


    def check_for_header_and_process(self):
        if self.unused_data.startswith(_HEADER):
            self.unused_data = self.unused_data[len(_HEADER) :]
            return True
        elif not self.unused_data.startswith(_HEADER[:len(self.unused_data)]):
            raise OSError("Header expected first")
        else:
            return False

    def check_for_footer_and_process(self):
        footer_start = self.unused_data.find(_EOS)
        logging.debug(f"check_for_footer_and_process: {footer_start=}")
        if footer_start > 0:
            logging.warning(f"Expected footer right here, but it's {footer_start} bytes further")
            return f"displaced to start at {footer_start}"
        if footer_start == -1:
            if len(self.unused_data) >= len(_EOS):
                return("not found")
            else:
                return "looking"
        if footer_start == 0:
            footer_len = len(_EOS)
            self.unused_data = self.unused_data[footer_start + footer_len:]
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

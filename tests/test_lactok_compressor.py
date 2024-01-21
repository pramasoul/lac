"""Test script for lactok_compressor"""

#import array
#import functools
#import io
import copy
import logging
#import os
#import pathlib
import psutil
import random
import struct
import sys

import numpy as np


import pytest

from binascii import hexlify, unhexlify
from contextlib import contextmanager
from io import BytesIO, DEFAULT_BUFFER_SIZE
from typing import Callable, List, Tuple

from unittest.mock import mock_open

#import tok_compressor as tc
import lactok_compressor as latc
from lactok_compressor import LACTokCompressor, LACTokDecompressor

#from lac_llm import FlatPredictionService, CountingPredictionService


# The test configurations to choose from (source-configured)
# Thorough, and quite time-consuming:
configurations = []
for model_name in ["internal", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
    for device in ["cpu"]:
# for model_name, device in [("internal", "cuda:0"),
#                            ("gpt2", "cuda:1"),
#                            ("gpt2-medium", "cuda:2"),
#                            ("gpt2-xl", "cuda:3"),
#                            ("internal", "cpu"),
# ]:
#     if True: # cheap indent
        thread_list = [1]
        if device == "cpu":
            thread_list = [psutil.cpu_count(logical=False)]
        for threads in thread_list:
            configurations.append(
                { "model_name": model_name,
                  "device": device,
                  "threads": threads,
                }
            )


# Fastest configuration. Run this first, until you pass, then run thorough configuration
#configurations = [{"model_name": "internal", "device": "cuda:3", "threads": psutil.cpu_count(logical=False)}]

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


################################################################
# Tests

def test_bytes_to_bits():
    y2b = latc.bytes_to_bits
    assert y2b(b'') == []
    assert y2b(b'0') == [0,0,1,1, 0,0,0,0]
    assert y2b(unhexlify('cafe')) == [1,1,0,0, 1,0,1,0, 1,1,1,1, 1,1,1,0]

def test_bits_to_bytes():
    b2y = latc.bits_to_bytes
    y2b = latc.bytes_to_bits
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
    b2y = latc.bits_to_bytes
    y2b = latc.bytes_to_bits
    random.seed(42)
    for l in (0, 1, 10, 1024, 1<<20-1):
        b = random.randbytes(1024)
        assert b2y(y2b((b))) == (b, [])


@pytest.fixture
def lac_magic_bytes():
    return b'2\x82\xc2Z'

# LACT_args = { "model_name": "internal",
#               "device": "cuda:3",
#               "threads": psutil.cpu_count(logical=False),
# }

# This brings our command-line switches into arguments for LACTok
# Note "model_name" is set with --model <name>
@pytest.fixture
def lact_args(model_name, device, threads):
    return { "model_name": model_name,
             "device": device,
             "threads": threads,
    }

#@pytest.mark.skip(reason="Dependent on header format")
@pytest.mark.compressed_data
def test_compress_empty(lact_args, lac_magic_bytes):
    c = LACTokCompressor(**lact_args)
    compressed = c.compress("") + c.flush()
    #assert compressed == b"\xfe\xfe\xff\xc0That's all, folks!"
    #assert compressed.startswith(b"\xfe\xfe")
    assert compressed.startswith(lac_magic_bytes)
    #assert compressed.endswith(b"That's all, folks!")
    t = len(b"That's all, folks!")
    assert compressed[-(t + 32) : -32] == b"That's all, folks!"
    #assert compressed[2] == b"\xff"[0]
    #assert len(compressed) == 4 + len(b"That's all, folks!")

    
#@pytest.mark.compressed_data
def test_decompress_empty(lact_args):
    #compressed = b"\xfe\xfe\xff\xffThat's all, folks!"
    #compressed = b"\xfe\xfe\xff\xc0That's all, folks!"
    c = LACTokCompressor(**lact_args)
    compressed = c.compress("") + c.flush()
    d = LACTokDecompressor(**lact_args)
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

@pytest.fixture
def medium_swedish_text(): # Some useful two-byte UTF-8 in here
    return r"""Om jag ock bleve tusen år gammal tror jag aldrig på jorden jag skulle
kunna glömma den dagen jag kom till min plats som piga. Om jag bleve
tusen år gammal skall jag ändå aldrig kunna få ur mitt minne den
stunden, den resan, den ankomsten. Fyra timmar på tåg neråt landet, vid
stationen skulle jag hämtas av husbonden för att sedan åka en mil
landväg. Riggad var jag och blommor hade jag fått av skrattande vänner
och bekanta, som viftade av mig, och jag kände mig mest som en
Amerikaresande när jag sjönk ned på en tom bänk i kupén och började
övertänka situationen. Det gick som en lång rysning över mig, när jag
såg framåt mot den månad som väntade, och det var just inte med glada
tankar jag stirrade ut genom fönstret, som var smutsigt och regnigt och
grått -- en trist och beklämmande fyrkant där bilderna gledo förbi lika
snabbt som på en biograf. Liljeholmsbron, Älvsjö, Huddinge -- herre gud
om jag vore på hemväg i stället! frös jag till invärtes och började
beskåda mig själv så gott jag kunde. Allting var främmande och kändes så
underligt. En lång brun kappa, händerna i bruna bomullsvantar och genom
den glesa vävnaden blänkte en bred gullring fram som tecken på min nya
värdighet av förlovad. Vad var det nu han hette min fästman? funderade
jag. Svensson -- nej Andersson -- _nej_ Bernhard Karlsson. Det stod ju i
ringen: »Din egen Bernhard». Vad i all världen hade jag fått det namnet
ifrån -- Bernhard! Det var då det löjligaste jag kunnat hitta på, varför
inte Karl, August eller Ernst eller vad som helst utom det där fåniga
namnet. Jag satt och retade upp mig själv och skrattade litet överlägset
åt min egen tydliga och odisputabla nervositet."""



def compress_decompress_test(text, lact_args):
    c = LACTokCompressor(**lact_args)
    d = LACTokDecompressor(**lact_args)
    compressed = c.compress(text) + c.flush()
    decompressed = d.decompress(compressed)
    if type(text) is str: text = text.encode("utf8")
    assert text == decompressed
    
def test_cd_short(lact_args):
    compress_decompress_test(b"", lact_args)
    compress_decompress_test(b"Hi!", lact_args)
    compress_decompress_test(b"Hi!Hi!", lact_args)
    compress_decompress_test(b"The quick brown fox, et al.", lact_args)

def test_cd_short_nl(lact_args):
    c,d = LACTokCompressor(**lact_args), LACTokDecompressor(**lact_args)
    text = b"\n"
    zbody = c.compress(text)
    ztail = c.flush()
    reconstructed = d.decompress(zbody + ztail)
    assert text == reconstructed

def test_cd_short_2nl(lact_args):
    c,d = LACTokCompressor(**lact_args), LACTokDecompressor(**lact_args)
    text = b"\n\n"
    zbody = c.compress(text)
    ztail = c.flush()
    reconstructed = d.decompress(zbody + ztail)
    assert text == reconstructed

def test_cd_brief(lact_args):
    compress_decompress_test(b"The quick brown fox jumped over the lazy dogs.\n", lact_args)

def test_cd_medium(medium_text, lact_args):
    compress_decompress_test(medium_text, lact_args)

def test_cd_swedish_short(lact_args):
    compress_decompress_test("tusen år gammal skall jag ändå aldrig kunna få ur mitt minne den", lact_args)
    compress_decompress_test("tusen år gammal skall jag ändå aldrig kunna få ur mitt minne den".encode('utf-8'), lact_args)

def test_cd_swedish_medium(medium_swedish_text, lact_args):
    compress_decompress_test(medium_swedish_text, lact_args)

def cd_char_at_a_time_test(text, lact_args):
    comp = LACTokCompressor(**lact_args)
    logging.info(f"caat compressing {text}: ")

    if isinstance(text, str):
        compressed = b"".join(comp.compress(char) for char in text) + comp.flush()
    elif isinstance(text, (bytes, bytearray)):
        compressed = b"".join(comp.compress(bytes((m,))) for m in text) + comp.flush()
    else:
        assert 0, f"Bad input type {typeof(text)} to cd_char_at_a_time_test()"
        
    logging.info(f"caat decompressing all together: ")
    decompressed = LACTokDecompressor(**lact_args).decompress(compressed)
    if type(text) is str: text = text.encode("utf8")
    assert text == decompressed
    logging.info(f"caat decompressing byte at a time: ")
    decomp = LACTokDecompressor(**lact_args)
    decompressed = b"".join(decomp.decompress(bytes([b])) for b in compressed)
    assert text == decompressed

    
def test_cd_caat_brief(lact_args):
    cd_char_at_a_time_test("", lact_args)
    cd_char_at_a_time_test("Hi!", lact_args)
    cd_char_at_a_time_test("The quick brown fox jumped over the lazy dogs.\n", lact_args)

def test_cd_caat_medium(medium_text, lact_args):
    cd_char_at_a_time_test(medium_text, lact_args)

def test_cd_caat_swedish_short(lact_args): # character-at-a-time with some multibyte UTF-8
    cd_char_at_a_time_test("tusen år gammal skall jag ändå aldrig kunna få ur mitt minne den", lact_args)
    cd_char_at_a_time_test("tusen år gammal skall jag ändå aldrig kunna få ur mitt minne den".encode('utf-8'), lact_args)

def test_cd_caat_swedish_medium(medium_swedish_text, lact_args):
    cd_char_at_a_time_test(medium_swedish_text, lact_args)

import glob
import os

def like_from_tlacz_test(size, lact_args):
    #logging.debug(f"test_like_from_tlacz({size=})")
    test_size = 0
    text = bytearray(size)
    for fname in glob.glob(os.path.join(glob.escape(os.path.dirname(__file__)), '*.py')):
        with open(fname, 'rb') as fh:
            test_size += fh.readinto(memoryview(text)[test_size:])
        if test_size > 128*1024:
            break
    assert len(text) == size
    #BIG_DATA = lac.compress(BIG_TEXT, compresslevel=1)
    comp = LACTokCompressor(**lact_args)
    data = comp.compress(text, compresslevel=1)
    logging.debug(f"pre-flush {comp=}")
    data += comp.flush()
    logging.debug(f"post-flush {comp=}")

def test_like_from_tlacz_short(lact_args):
    like_from_tlacz_test(128, lact_args)

@pytest.mark.slow
@pytest.mark.skip(reason="Not useful as a recurring test")
@pytest.mark.parametrize("size", [1<<i for i in range(16)])
def test_like_from_tlacz_ramp(benchmark, size, lact_args):
    #like_from_tlacz_test(size)
    #v = benchmark(like_from_tlacz_test, size)
    benchmark.pedantic(like_from_tlacz_test, args=(size,lact_args), iterations=1, rounds=1)
    

data1 = b"""  int length=DEFAULTALLOC, err = Z_OK;
  PyObject *RetVal;
  int flushmode = Z_FINISH;
  unsigned long start_total_out;

"""

@pytest.mark.skip(reason="FIXME: Is this test valid with LLM prediction?")
def test_find_tok_difference_in_compressed(lact_args):
    for n in range(7):
        input_data = data1 * n
        comp_lbl = LTC(tok_mode = "line-by-line",save_toks=1, lact_args=lact_args)
        data_lbl = comp_lbl.compress(input_data, compresslevel=1) + comp_lbl.flush()
        comp_hold = LTC(tok_mode = "hold all until flush",save_toks=1, lact_args=lact_args)
        data_hold = comp_hold.compress(input_data, compresslevel=1) + comp_hold.flush()
        comp_minbuf = LTC(tok_mode = "buffer minimum for correct",save_toks=1, lact_args=lact_args)
        data_minbuf = comp_minbuf.compress(input_data, compresslevel=1) + comp_minbuf.flush()
        decomp = LACTokDecompressor(**lact_args)
        decompressed = decomp.decompress(data_hold)

        comp_lbl.predictor.restart()
        bits1 = [list(comp_lbl.a2b.bits(comp_lbl.toks[0],0))]
        a = (repr(comp_lbl.a2b),comp_lbl.a2b.certain)
        bits1 += [list(comp_lbl.a2b.bits(comp_lbl.toks[1]))]
        comp_lbl.predictor.restart()
        bits2 = list(comp_lbl.a2b.bits(comp_lbl.toks[0]+comp_lbl.toks[1]))
        comp_lbl.predictor.restart()
        bits3 = [list(comp_lbl.a2b.bits(comp_lbl.toks[0]+comp_lbl.toks[1],0))]
        b = repr(comp_lbl.a2b)
        bits3 += [list(comp_lbl.a2b.bits([]))]
        
        assert (bits1[0]+bits1[1]) == bits2
        
        assert decompressed == input_data
        assert data_hold == data_minbuf
        assert LACTokDecompressor(**lact_args).decompress(data_lbl) == input_data
        #assert data_lbl == data_hold == data_minbuf

        # logging.info(f"pre-flush {comp=} {len(data)=}")
        # data += comp.flush()
        # logging.info(f"post-flush {comp=} {len(data)=}")


@pytest.mark.slow
@pytest.mark.parametrize("n", [0,1,2,3,4,8,16,32,48,50])
def test_n_like_tlacz_write_read_with_pathlike_file(n, lact_args):
    like_tlacz_write_read_with_pathlike_file_test(data1 * n, lact_args)

@pytest.mark.slow
def test_like_tlacz_write_read_with_pathlike_file(tmp_path, lact_args):
    data50 = data1 * 50
    like_tlacz_write_read_with_pathlike_file_test(data50, lact_args)


def like_tlacz_write_read_with_pathlike_file_test(input_data, lact_args):
    comp = LACTokCompressor(**lact_args)
    data = comp.compress(input_data, compresslevel=1)
    logging.info(f"pre-flush {comp=} {len(data)=}")
    data += comp.flush()
    logging.info(f"post-flush {comp=} {len(data)=}")
    decomp = LACTokDecompressor(**lact_args)
    decompressed = decomp.decompress(data)
    if decompressed != input_data:
        logging.info(f"{input_data[-50:]}")
        logging.info(f"{decompressed[len(input_data)-50:]}")
        logging.info(f"{comp.predictor=}")
        logging.info(f"{comp.a2b.accept_list[-50:]=}")
        logging.info(f"{decomp.predictor=}")
        logging.info(f"{decomp.b2a.accept_list[-50:]=}")
    assert decompressed == input_data


@pytest.mark.skip(reason="atok_compressor only does a single stream")
def test_like_tlacz_multi_stream_ordering(lact_args):
    # Test the ordering of streams when reading a multi-stream archive.
    data1 = b"foo" * 1000
    data2 = b"bar" * 1000
    # with LacFile(filename, "w") as bz2f:
    #     bz2f.write(data1)
    # with LacFile(filename, "a") as bz2f:
    #     bz2f.write(data2)
    # with LacFile(filename) as bz2f:
    #     assert bz2f.read() == data1 + data2
    comp = LACTokCompressor(**lact_args)
    data = comp.compress(data1, compresslevel=1) + comp.flush()
    comp = LACTokCompressor(**lact_args)
    data += comp.compress(data2, compresslevel=1) + comp.flush()
    decomp = LACTokDecompressor(**lact_args)
    decompressed = decomp.decompress(data)
    assert decompressed == data1 + data2


def test_compress_decompress_temperatures(medium_text, lact_args):
    text = medium_text
    temperatures = (0.5, 1.0, 2.0)
    comps = []
    for T in temperatures:
        c = LACTokCompressor(**lact_args, temperature=T)
        comps.append(c.compress(text) + c.flush())
    assert all(len(v)>0 for v in comps)
    assert min(len(v) for v in comps) < max(len(v) for v in comps)
    for T, compressed in zip(temperatures, comps):
        d = LACTokDecompressor(**lact_args, temperature=T)
        decompressed = d.decompress(compressed)
        assert text == decompressed.decode('utf8')


def test_decompress_respecting_header(medium_text, lact_args):
    text = medium_text
    # compress differently than the lact args:
    d = copy.copy(lact_args)
    d["model_name"] = ["gpt2", "internal"][lact_args["model_name"] == "gpt2"]
    d["device"] = ["cuda", "cpu"][lact_args["device"] == "cuda"]
    ## if d["threads"] >=32:
    d["temperature"] = "temperature" in lact_args and lact_args["temperature"] + 1.0 or 0.5
    comp = LACTokCompressor(**d)
    data = comp.compress(text, compresslevel=1) + comp.flush()

    # Does decompression respect the header in the data?
    decomp = LACTokDecompressor(**lact_args)
    decompressed = decomp.decompress(data)
    assert decompressed.decode('utf8') == text

def test_decompress_respecting_header_generic_cuda(medium_text, lact_args):
    # This test presents difficulties with automating its
    # verification. The issue is automating observing what cuda device
    # on which the compressor runs. One can run this manually,
    # specifying a different cuda device than cuda:1, and observe the
    # compression happen on cuda:1 and the decompression on the
    # specified other cuda device, with e.g. nvtop.
    # E.g. pytest tests/test_lactok_compressor.py::test_decompress_respecting_header_generic_cuda --device cuda:3

    text = medium_text
    # compress with cuda:0
    ka = copy.copy(lact_args)
    ka["device"] = "cuda:1"
    comp = LACTokCompressor(**ka)
    data = comp.compress(text, compresslevel=1) + comp.flush()

    # Does decompression respect the header in the data?
    decomp = LACTokDecompressor(**lact_args)
    decompressed = decomp.decompress(data)
    assert decompressed.decode('utf8') == text


def test_decompress_hash(lact_args):
    text = "Hi!"
    c = LACTokCompressor(**lact_args)
    compressed = c.compress(text) + c.flush()
    d = LACTokDecompressor(**lact_args)
    decompressed = d.decompress(compressed)
    assert text == decompressed.decode('utf8')
    messed = bytearray(compressed)
    messed[-1] ^= 1             # flip a bit in the hash at the end
    d = LACTokDecompressor(**lact_args)
    with pytest.raises(OSError):
        decompressed = d.decompress(messed)

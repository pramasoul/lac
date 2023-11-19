# Test arithmetic_coding.py

import pytest
import torch
import ctypes
import json
import logging

from binascii import hexlify, unhexlify
from typing import Callable, List

import numpy as np

from arithmetic_coding import ACSampler, packbits, unpackbits

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])  # StreamHandler logs to console

# Now you can use logging in your tests
def test_example():
    logging.debug("This is a debug message.")


CONTENT = "content"


def test_create_file(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text(CONTENT, encoding="utf-8")
    assert p.read_text(encoding="utf-8") == CONTENT
    assert len(list(tmp_path.iterdir())) == 1


@pytest.fixture
def acs() -> ACSampler:
    return ACSampler()
    
def test_create(acs: ACSampler):
    assert type(acs) == ACSampler


# Convenience function
def compression_done(sampler: ACSampler) -> None:
    sampler.on_compress_done = None
    sampler.flush_compress()
    sampler.compress_output.flush()
    sampler.bits_per_token = None
    sampler.compress_output = None

# Convenience function
def decompression_done(sampler: ACSampler) -> None:
    sampler.on_decompress_done = None
    sampler.decompress_output = None
    sampler.bits_per_token = None

#def compress_base_ten(digits:str, pdf: [int] = [0.1]*10) -> bytearray:
def compress_base_ten(digits:str,
                      pdfun: Callable[[int], List[int]] = lambda i: [0.1]*10) -> bytearray:
    output = bytearray(0)
    sampler = ACSampler()
    def tokgen():
        for d in digits:
            yield int(d)
    sampler.compress_tokens = tokgen()
    sampler.compress_output = packbits(output.append)
    sampler.on_compress_done = lambda: compression_done(sampler)
    i = 0
    while not sampler.compress_done:
        logging.debug(f"sampler {sampler}")
        token = sampler.sample(pdfun(i))
        i += 1
    logging.debug(f"compress_base_ten {sampler}")
    return output

def decompress_base_ten(data: bytearray,
                        pdfun: Callable[[int], List[int]] = lambda i: [0.1]*10) -> str:
    output = ""
    sampler = ACSampler()
    sampler.decompress_bits = unpackbits(data)
    sampler.decompress_output = lambda tok: None
    sampler.on_decompress_done = lambda: decompression_done(sampler)
    i = 0
    while not sampler.decompress_done:
        logging.debug(f"sampler {sampler}")
        token = sampler.sample(pdfun(i))
        output += str(token)
        i += 1
    logging.debug(f"decompress_base_ten {sampler}")
    return output

def test_compress_base_ten():
    digit_str = '3' * 14
    z = compress_base_ten(digit_str)
    assert isinstance(z, bytearray)
    assert hexlify(z) == b'555555555554'

@pytest.mark.skip(reason='need stop token mechanism')
def test_decompress_base_ten():
    z = unhexlify('555555555554')
    assert decompress_base_ten(z) == '3'*14

@pytest.mark.skip(reason='need stop token mechanism')
def test_compress_decompress_base_ten():
    digit_str = "314159265358979323846"
    z = compress_base_ten(digit_str)
    assert decompress_base_ten(z) == digit_str

def pdf_leveling_tilt(i: int) -> List[int]:
    return (np.arange(10) + i).tolist()

@pytest.mark.skip(reason='need stop token mechanism')
def test_compress_decompress_base_ten_varying_pdf():
    digit_str = "314159265358979323846"
    z = compress_base_ten(digit_str, pdfun=pdf_leveling_tilt)
    assert decompress_base_ten(z, pdfun=pdf_leveling_tilt) == digit_str

def test_compress_base_ten_varying_pdf():
    digit_str = '3' * 14
    z = compress_base_ten(digit_str, pdfun=pdf_leveling_tilt)
    assert isinstance(z, bytearray)
    #FIXME:assert hexlify(z) == b'555555555554'

def make_pdf_digit_oracle(s: str) -> Callable[[int], List[int]]:
    def pdfun(i: int) -> List[int]:
        v = [0] * 10
        try:
            v[int(s[i])] = 1
        except IndexError: # Ran off end of hint string
            v = [0.1] * 10
        return v
    return pdfun

def test_make_pdf_digit_oracle():
    digit_str = "314159265358979323846"
    pdfun = make_pdf_digit_oracle(digit_str)
    assert pdfun(0) == [0,0,0,1,0,0,0,0,0,0]
    assert pdfun(1) == [0,1,0,0,0,0,0,0,0,0]
    assert pdfun(5) == [0,0,0,0,0,0,0,0,0,1]
    assert pdfun(100) == [0.1] * 10

def test_compress_base_ten_digit_oracle():
    digit_str = '3' * 14
    pdfun = make_pdf_digit_oracle(digit_str)
    z = compress_base_ten(digit_str, pdfun=pdfun)
    assert isinstance(z, bytearray)
    #FIXME:assert hexlify(z) == b'555555555554'

def test_compress_decompress_base_ten_digit_oracle():
    digit_str = "314159265358979323846"
    pdfun = make_pdf_digit_oracle(digit_str)
    z = compress_base_ten(digit_str, pdfun=pdfun)
    assert decompress_base_ten(z, pdfun=pdfun) == digit_str

def test_compress_decompress_base_ten_digit_weak_oracle():
    digit_str = "314159265358979323846"
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o+b for o,b in zip(orafun(0), [0.1]*10)]
    z = compress_base_ten(digit_str, pdfun=pdfun)
    assert decompress_base_ten(z, pdfun=pdfun) == digit_str

def test_compress_decompress_base_ten_digit_pretty_good_oracle():
    digit_str = "314159265358979323846"
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o+b for o,b in zip(orafun(0), [0.01]*10)]
    z = compress_base_ten(digit_str, pdfun=pdfun)
    assert decompress_base_ten(z, pdfun=pdfun) == digit_str

def test_compress_decompress_base_ten_digit_impressive_oracle():
    digit_str = "314159265358979323846"
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o+b for o,b in zip(orafun(0), [0.0001]*10)]
    z = compress_base_ten(digit_str, pdfun=pdfun)
    assert decompress_base_ten(z, pdfun=pdfun) == digit_str

def test_compress_decompress_base_ten_digit_quite_impressive_oracle():
    digit_str = "314159265358979323846"
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o+b for o,b in zip(orafun(0), [0.0000001]*10)]
    z = compress_base_ten(digit_str, pdfun=pdfun)
    assert decompress_base_ten(z, pdfun=pdfun) == digit_str


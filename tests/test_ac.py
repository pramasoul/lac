# Test ac_for_z.py

import pytest
import torch
import ctypes
import json
import logging

from binascii import hexlify, unhexlify
from typing import Callable, List

import numpy as np

from ac_for_z import ACSampler, packbits, unpackbits

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)  # StreamHandler logs to console


# Now you can use logging in your tests
def test_example():
    logging.debug("This is a debug message.")


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


# from acs_adapter import ACSamplerAdapted
def compress_base_ten(
    digits: str, pdfun: Callable[[int], List[int]] = lambda i: [0.1] * 11
) -> bytearray:
    stop_token = 10
    output = bytearray(0)
    sampler = ACSampler(end_of_text_token=10)

    def tokgen():
        for d in digits:
            yield int(d)
        yield stop_token
        yield stop_token

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


def decompress_base_ten(
    data: bytearray, pdfun: Callable[[int], List[int]] = lambda i: [0.1] * 11
) -> str:
    stop_token = 10
    # sampler = ACSamplerAdapted()
    sampler = ACSampler(end_of_text_token=10)
    sampler.decompress_bits = unpackbits(data)
    output_tokens = []

    def append_to_output(s):
        output_tokens.append(s)

    sampler.decompress_output = append_to_output
    sampler.on_decompress_done = lambda: decompression_done(sampler)
    i = 0
    while not sampler.decompress_done:
        logging.debug(f"sampler {sampler}")
        token = sampler.sample(pdfun(i))
        logging.debug(f"token {token}")
        if token == stop_token:
            # FIXME: what is right here? compression_done(sampler)
            sampler.decompress_done = True
            break
        # output += str(token)
        i += 1
    output = "".join(str(tok) for tok in output_tokens)
    logging.debug(f"decompress_base_ten {sampler} output {output}")
    return output


@pytest.fixture(scope="session")
def long_pi():
    import mpmath

    mpmath.mp.dps = 10010
    pi_str = mpmath.nstr(mpmath.pi, 10001)
    return pi_str[0] + pi_str[2:]


def test_long_pi(long_pi):
    assert long_pi.startswith("314159265358979323846")


@pytest.mark.skip(reason="Output format in flux")
def test_compress_base_ten():
    digit_str = "3" * 14
    z = compress_base_ten(digit_str)
    assert isinstance(z, bytearray)
    assert hexlify(z) == b"555555555554"


@pytest.mark.skip(reason="Output format in flux")
def test_decompress_base_ten():
    z = unhexlify("555555555554")
    assert decompress_base_ten(z) == "3" * 14


def test_compress_decompress_base_ten(long_pi):
    digit_str = long_pi[:2]
    z = compress_base_ten(digit_str)
    assert decompress_base_ten(z) == digit_str
    digit_str = long_pi[:100]
    z = compress_base_ten(digit_str)
    assert decompress_base_ten(z) == digit_str
    digit_str = long_pi[:1000]
    z = compress_base_ten(digit_str)
    assert decompress_base_ten(z) == digit_str


@pytest.mark.slow
def test_compress_decompress_base_ten_long(long_pi):
    digit_str = long_pi
    z = compress_base_ten(digit_str)
    assert decompress_base_ten(z) == digit_str


def pdf_leveling_tilt(i: int) -> List[int]:
    return (np.arange(11) + i).tolist()


def test_compress_decompress_base_ten_varying_pdf():
    digit_str = "314159265358979323846"
    z = compress_base_ten(digit_str, pdfun=pdf_leveling_tilt)
    assert decompress_base_ten(z, pdfun=pdf_leveling_tilt) == digit_str


def test_compress_base_ten_varying_pdf():
    digit_str = "3" * 14
    z = compress_base_ten(digit_str, pdfun=pdf_leveling_tilt)
    assert isinstance(z, bytearray)
    # FIXME:assert hexlify(z) == b'555555555554'


def make_pdf_digit_oracle(s: str) -> Callable[[int], List[int]]:
    def pdfun(i: int) -> List[int]:
        v = [0] * 10 + [0.1]
        try:
            v[int(s[i])] = 1
        except IndexError:  # Ran off end of hint string
            v = [0.1] * 11
        return v

    return pdfun


def test_make_pdf_digit_oracle():
    digit_str = "314159265358979323846"
    pdfun = make_pdf_digit_oracle(digit_str)
    assert pdfun(0) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.1]
    assert pdfun(1) == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.1]
    assert pdfun(5) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.1]
    assert pdfun(100) == [0.1] * 11


def test_compress_base_ten_digit_oracle():
    digit_str = "3" * 14
    pdfun = make_pdf_digit_oracle(digit_str)
    z = compress_base_ten(digit_str, pdfun=pdfun)
    assert isinstance(z, bytearray)
    # FIXME:assert hexlify(z) == b'555555555554'


# @pytest.mark.skip(reason="FIXME")
def test_compress_decompress_base_ten_digit_oracle():
    digit_str = "314159265358979323846"
    pdfun = make_pdf_digit_oracle(digit_str)
    z = compress_base_ten(digit_str, pdfun=pdfun)
    dd_str = decompress_base_ten(z, pdfun=pdfun)
    # assert dd_str.startswith(digit_str)
    assert dd_str == digit_str


def test_compress_decompress_base_ten_digit_weak_oracle():
    digit_str = "314159265358979323846"
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o + b for o, b in zip(orafun(0), [0.1] * 11)]
    z = compress_base_ten(digit_str, pdfun=pdfun)
    dd_str = decompress_base_ten(z, pdfun=pdfun)
    # assert dd_str.startswith(digit_str)
    assert dd_str == digit_str


def test_compress_decompress_base_ten_digit_pretty_good_oracle():
    digit_str = "314159265358979323846"
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o + b for o, b in zip(orafun(0), [0.01] * 11)]
    z = compress_base_ten(digit_str, pdfun=pdfun)
    dd_str = decompress_base_ten(z, pdfun=pdfun)
    # assert dd_str.startswith(digit_str)
    assert dd_str == digit_str


def test_compress_decompress_base_ten_digit_impressive_oracle():
    digit_str = "314159265358979323846"
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o + b for o, b in zip(orafun(0), [0.0001] * 11)]
    z = compress_base_ten(digit_str, pdfun=pdfun)
    dd_str = decompress_base_ten(z, pdfun=pdfun)
    # assert dd_str.startswith(digit_str)
    assert dd_str == digit_str


def test_compress_decompress_base_ten_digit_quite_impressive_oracle():
    digit_str = "314159265358979323846"
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o + b for o, b in zip(orafun(0), [0.0000001] * 11)]
    z = compress_base_ten(digit_str, pdfun=pdfun)
    dd_str = decompress_base_ten(z, pdfun=pdfun)
    # assert dd_str.startswith(digit_str)
    assert dd_str == digit_str


@pytest.mark.slow
def test_compress_decompress_base_ten_oracular_long_1(long_pi):
    digit_str = long_pi
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o + b for o, b in zip(orafun(0), [1 / 100, 000] * 11)]  # sic
    z = compress_base_ten(digit_str, pdfun=pdfun)
    dd_str = decompress_base_ten(z, pdfun=pdfun)
    assert dd_str == digit_str


@pytest.mark.slow
def test_compress_decompress_base_ten_oracular_long_2(long_pi):
    digit_str = long_pi
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o + b for o, b in zip(orafun(0), [1 / 100_000] * 11)]  # sic
    z = compress_base_ten(digit_str, pdfun=pdfun)
    dd_str = decompress_base_ten(z, pdfun=pdfun)
    assert dd_str == digit_str


@pytest.mark.slow
def test_compress_decompress_base_ten_oracular_long_i(long_pi):
    digit_str = long_pi
    orafun = make_pdf_digit_oracle(digit_str)
    pdfun = lambda i: [o + b for o, b in zip(orafun(i), [1 / 100_000] * 11)]
    z = compress_base_ten(digit_str, pdfun=pdfun)
    dd_str = decompress_base_ten(z, pdfun=pdfun)
    assert dd_str == digit_str


def compress_raw(
    toks, pdfun: Callable[[int], List[int]] = lambda i: [1] * 2, prec: int = 48
):
    output = []
    sampler = ACSampler(precision=prec)
    sampler.compress_tokens = toks
    sampler.compress_output = output.append

    def comp_done():
        sampler.flush_compress()
        sampler.compress_output = None

    sampler.on_compress_done = comp_done
    i = 0
    while not sampler.compress_done:
        logging.debug(f"sampler {sampler}")
        token = sampler.sample(pdfun(i))
        yield from output
        output.clear()
        i += 1
    logging.debug(f"compress_raw {sampler}")
    yield from output


def decompress_raw(
    bits, pdfun: Callable[[int], List[int]] = lambda i: [1] * 2, prec: int = 48
):
    sampler = ACSampler(precision=prec)
    sampler.decompress_bits = bits
    i = 0
    while not sampler.decompress_done:
        logging.debug(f"sampler {sampler}")
        token = sampler.sample(pdfun(i))
        yield token
        logging.debug(f"token {token}")
        i += 1
    logging.debug(f"decompress_raw {sampler}")


def test_unbalanced_ternary():
    inp = [1, 0, 2]
    pdfunc = lambda i: [1, 0, 1]
    prec = 4
    bits = list(compress_raw(inp, pdfunc, prec=prec))
    print(bits)
    toks = list(decompress_raw(bits, pdfunc, prec=prec))
    assert toks[: len(inp)] == inp
    #
    #       0 1 2 3 4 5 6 7 8 9 a b c d e f|0 1 2 3 4 5 6 7 8
    # tok:  0 0 0 0 0 0 0 1 1 2 2 2 2 2 2 2|
    # "1"->               [ . )            |                  ->  0
    #                                   [ .|. . )             ->  1
    #                               [ . . .|. . . . )         ->  1
    #                       [ . . . . . . .|. . . . . . . . )
    #                       [ . . . . . . )|                  ->  1
    #       [ . . . . . . . . . . . . . )  |
    #       0 1 2 3 4 5 6 8 9 a b c d e 10 |
    #       0 0 0 0 0 0 0 1 2 2 2 2 2 2    |
    # "0"-> [ . . . . . . )                |                  ->  0
    #       [                           )  |
    # "2"->                 [           )  |                  ->  1
    #       [                       )      |
    # flush  0 0 0 0 1 1 1 1 2 2 2 2        |
    #               [       )              |  -> 0
    #                       [              |) -> 1
    #       [                              |)
    #    01110101

    # what we got: (the error was using bisect_left instead of bisect_right)
    #       0 1 2 3 4 5 6 7 8 9 a b c d e f|0 1 2 3 4 5 6 7 8 9 a b c d e f
    # "1"->                                |                               -> 0
    #                                   [  |    )                          -> 1
    #                               [      |        )                      -> 1
    #                       [              |                )
    # "0"->                                |                               -> 1
    #       [                           )  |
    #       0 0 0 0 0 0 1 2 2 2 2 2 2 2
    # "2"->               [             )  |                               -> 0
    #
    # flush -> 2
    # 011110

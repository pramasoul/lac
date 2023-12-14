# Test ac2_for_z.py

import pytest
import torch
import ctypes
import json
import logging

from binascii import hexlify, unhexlify
from typing import Callable, List

import numpy as np

#import ac2_for_z as ac2
from ac2_for_z import AC, Predictor, CDFPredictor, ProbPredictor
from ac2_for_z import region_overlap, group_bits, ungroup_bits

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)  # StreamHandler logs to console


# Now you can use logging in your tests
def test_example():
    logging.debug("This is a debug message.")


@pytest.fixture(scope="session")
def long_pi():
    import mpmath

    mpmath.mp.dps = 10010
    pi_str = mpmath.nstr(mpmath.pi, 10001)
    return pi_str[0] + pi_str[2:]


def test_long_pi(long_pi):
    assert long_pi.startswith("314159265358979323846")


def test_region_overlap():
    # [a,b] with [c,d]
    assert region_overlap(0,0,0,0) == 1
    assert region_overlap(123, 124, 124, 999) == 1
    assert region_overlap(0, 1, 2, 3) == 0
    big = (1<<63) - 1
    assert region_overlap(big, big, big, big) == 1
    assert region_overlap(big-3, big-2, big-1, big) == 0
    assert region_overlap(0, -1, -2, -3) == 0
    assert region_overlap(0, big, 0, big) == big+1
    assert region_overlap(1, 0, 0, 0) == 0
    assert region_overlap(1, 0, -1, 0) == 0
    assert region_overlap(0, 1, -1, 0) == 1

@pytest.mark.skip(reason="less important")
def test_group_bits():
    pass


@pytest.fixture
def ac() -> AC:
    return AC()

def test_create(ac: AC):
    assert type(ac) == AC


def test_predictor():
    p = Predictor(3)
    assert p.val_to_symbol(0, 3) == 0
    assert p.val_to_symbol(1, 3) == 1
    assert p.val_to_symbol(2, 3) == 2
    assert p.val_to_symbol(3, 3) == 3 # out-of-domain
    assert p.val_to_symbol(4, 3) == 4 # out-of-domain
    assert p.val_to_symbol(0, 9) == 0
    assert p.val_to_symbol(1, 9) == 0
    assert p.val_to_symbol(1, 9) == 0
    assert p.val_to_symbol(2, 9) == 0
    assert p.val_to_symbol(3, 9) == 1
    assert p.val_to_symbol(3, 4) == 2




def test_AC_ternary_1():
    a = AC(Predictor(3), 16)
    enc = a.to_bin
    dec = a.from_bin
    in_data = [0, 1, 2, 1, 0]
    out_bits = list(enc.bits(in_data))
    assert out_bits == [0,0,1,1,0,0,1,1,0]
    recovered = list(dec.run(out_bits))
    assert recovered[:len(in_data)] == in_data

def test_AC_binary_with_stop_1():
    predictor = CDFPredictor([49,98,100])
    a = AC(predictor, 16)
    enc = a.to_bin
    dec = a.from_bin
    #in_data = [0, 1, 2, 1, 0]
    in_data = [1,1,1,1,2]
    out_bits = list(enc.bits(in_data))
    assert out_bits == [1, 1, 1, 1, 0, 1, 1, 0, 0, 1]
    recovered = list(dec.run(out_bits))
    assert recovered[:len(in_data)] == in_data


def make_pdf_digit_oracle(s: str, lift=0.01) -> Callable[[int], List[int]]:
    def pdfun(i: int) -> List[int]:
        v = [lift] * 10 + [0.1 + lift]
        try:
            v[int(s[i])] += 1
        except IndexError:  # Ran off end of hint string
            v = [0.1] * 11
        return v

    return pdfun

def test_make_pdf_digit_oracle():
    digit_str = "314159265358979323846"
    pdfun = make_pdf_digit_oracle(digit_str)
    lift = 0.01
    assert pdfun(0) == [lift + v for v in [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.1]]
    assert pdfun(1) == [lift + v for v in [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.1]]
    assert pdfun(5) == [lift + v for v in [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.1]]
    assert pdfun(100) == [0.1] * 11


class OraclePredictor(ProbPredictor):
    def __init__(self, digit_str):
        super().__init__(11)
        self.digit_str = digit_str
        self.i = 0
    def prob(self,symbol):
        return int(10000*make_pdf_digit_oracle(self.digit_str)(self.i)[symbol])
    def accept(self,symbol):
        self.i += 1
        super().accept(symbol)
    def copy(self):
        r = OraclePredictor(self.digit_str)
        r.i = self.i
        return r

def test_AC_decimal_oracle():
    digit_str = "314159265358979323846"
    #pdfun = make_pdf_digit_oracle(digit_str)
    predictor = OraclePredictor(digit_str)
    a = AC(predictor, 16)
    enc = a.to_bin
    dec = a.from_bin
    in_data = [int(c) for c in digit_str] + [10] # Stop
    out_bits = list(enc.bits(in_data))
    assert out_bits == [0, 0, 1, 1, 0, 1, 0, 1, 1] # from a run
    recovered = list(dec.run(out_bits))
    assert recovered[:len(in_data)] == in_data
    assert len(out_bits) < len(digit_str)
    

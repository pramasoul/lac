# Test ac2_for_z.py

import pytest
import torch
import ctypes
import json
import logging
import random

from binascii import hexlify, unhexlify
from typing import Callable, List

import numpy as np

#import ac2_for_z as ac2
from ac2_for_z import AC, Predictor, CDFPredictor, PDFPredictor, ProbPredictor
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
def long_text():
    return r"""You will rejoice to hear that no disaster has accompanied the
commencement of an enterprise which you have regarded with such evil
forebodings. I arrived here yesterday, and my first task is to assure
my dear sister of my welfare and increasing confidence in the success
of my undertaking.

I am already far north of London, and as I walk in the streets of
Petersburgh, I feel a cold northern breeze play upon my cheeks, which
braces my nerves and fills me with delight. Do you understand this
feeling? This breeze, which has travelled from the regions towards
which I am advancing, gives me a foretaste of those icy climes.
Inspirited by this wind of promise, my daydreams become more fervent
and vivid. I try in vain to be persuaded that the pole is the seat of
frost and desolation; it ever presents itself to my imagination as the
region of beauty and delight. There, Margaret, the sun is for ever
visible, its broad disk just skirting the horizon and diffusing a
perpetual splendour. There—for with your leave, my sister, I will put
some trust in preceding navigators—there snow and frost are banished;
and, sailing over a calm sea, we may be wafted to a land surpassing in
wonders and in beauty every region hitherto discovered on the habitable
globe. Its productions and features may be without example, as the
phenomena of the heavenly bodies undoubtedly are in those undiscovered
solitudes. What may not be expected in a country of eternal light? I
may there discover the wondrous power which attracts the needle and may
regulate a thousand celestial observations that require only this
voyage to render their seeming eccentricities consistent for ever. I
shall satiate my ardent curiosity with the sight of a part of the world
never before visited, and may tread a land never before imprinted by
the foot of man. These are my enticements, and they are sufficient to
conquer all fear of danger or death and to induce me to commence this
laborious voyage with the joy a child feels when he embarks in a little
boat, with his holiday mates, on an expedition of discovery up his
native river. But supposing all these conjectures to be false, you
cannot contest the inestimable benefit which I shall confer on all
mankind, to the last generation, by discovering a passage near the pole
to those countries, to reach which at present so many months are
requisite; or by ascertaining the secret of the magnet, which, if at
all possible, can only be effected by an undertaking such as mine.

These reflections have dispelled the agitation with which I began my
letter, and I feel my heart glow with an enthusiasm which elevates me
to heaven, for nothing contributes so much to tranquillise the mind as
a steady purpose—a point on which the soul may fix its intellectual
eye. This expedition has been the favourite dream of my early years. I
have read with ardour the accounts of the various voyages which have
been made in the prospect of arriving at the North Pacific Ocean
through the seas which surround the pole. You may remember that a
history of all the voyages made for purposes of discovery composed the
whole of our good Uncle Thomas’ library. My education was neglected,
yet I was passionately fond of reading. These volumes were my study
day and night, and my familiarity with them increased that regret which
I had felt, as a child, on learning that my father’s dying injunction
had forbidden my uncle to allow me to embark in a seafaring life.

These visions faded when I perused, for the first time, those poets
whose effusions entranced my soul and lifted it to heaven. I also
became a poet and for one year lived in a paradise of my own creation;
I imagined that I also might obtain a niche in the temple where the
names of Homer and Shakespeare are consecrated. You are well
acquainted with my failure and how heavily I bore the disappointment.
But just at that time I inherited the fortune of my cousin, and my
thoughts were turned into the channel of their earlier bent.

Six years have passed since I resolved on my present undertaking. I
can, even now, remember the hour from which I dedicated myself to this
great enterprise. I commenced by inuring my body to hardship. I
accompanied the whale-fishers on several expeditions to the North Sea;
I voluntarily endured cold, famine, thirst, and want of sleep; I often
worked harder than the common sailors during the day and devoted my
nights to the study of mathematics, the theory of medicine, and those
branches of physical science from which a naval adventurer might derive
the greatest practical advantage. Twice I actually hired myself as an
under-mate in a Greenland whaler, and acquitted myself to admiration. I
must own I felt a little proud when my captain offered me the second
dignity in the vessel and entreated me to remain with the greatest
earnestness, so valuable did he consider my services."""


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


class OraclePredictor(PDFPredictor):
    def __init__(self, digit_str, precision=16):
        logging.debug(f"OraclePredictor(...,{precision=})")
        assert 1 < precision < 64 #uint64 limit
        super().__init__(None, precision)
        self.digit_str = digit_str
        self.digits = [int(c) for c in digit_str]
        self.i = 0
        self.pdfun = make_pdf_digit_oracle(self.digit_str)
        self.set_cdf_from_pdf(self.pdfun(0))
        
    def prob(self,symbol):
        # if self.i < len(self.digits) and symbol == self.digits[self.i]:
        #     rv = 1000
        # else:
        #     rv = 100
        if self.i < len(self.digits) and symbol == self.digits[self.i]:
            rv = 10100
        else:
            rv = 100
        if symbol == 10:
            #rv = 1100
            rv = 100
        return rv

    def accept(self,symbol):
        self.i += 1
        #super().accept(symbol)
        self.set_cdf_from_pdf(self.pdfun(self.i))

    def copy(self):
        r = type(self)(self.digit_str,self.precision)
        r.i = self.i
        return r

def AC_decimal_oracle_test(digit_str, precision=16, bpd=1):
    logging.debug(f"AC_decimal_oracle_test({len(digit_str)=}, {precision=})")
    predictor = OraclePredictor(digit_str, precision)
    a = AC(predictor, precision)
    enc = a.to_bin
    dec = a.from_bin
    in_data = [int(c) for c in digit_str] + [10] # Stop
    out_bits = list(enc.bits(in_data))
    recovered = list(dec.run(out_bits))
    assert recovered[:len(in_data)] == in_data
    r = ''.join(str(v) for v in recovered[:recovered.index(10)])
    assert r == digit_str
    assert len(out_bits) <= len(digit_str) * bpd


def test_AC_decimal_oracle(long_pi):
    AC_decimal_oracle_test(long_pi[:20], 16)
    for prec in range(16, 5, -1): # val_to_symbol range assertion fail at precision=5
        AC_decimal_oracle_test(long_pi[:20], prec)
    for prec in range(16, 64): # //I assume python bigint is coming into play here.
        # //We'll have to revisit once it's numpy'd
        # did this
        AC_decimal_oracle_test(long_pi[:20], prec)


@pytest.mark.slow
def test_AC_decimal_oracle_long(long_pi):
    for prec in range(6, 64): # //I assume python bigint is coming into play here.
        # //We'll have to revisit once it's numpy'd
        AC_decimal_oracle_test(long_pi, 16, 0.27)


r""" Predictor tests
symbol_to_range on ascending symbols as integers in-order returns tuple r
assert r[1] - r[0] > 0
for all symbols val_to_symbol(x) r[0] <= x < r[1] returns symbol
2^^(precision-1) <= denom <=? 2^^precision 
"""

def test_decimal_oracle(long_pi):
    digit_str = long_pi
    for s in range(11):
        for denom in (100, 12321, 17, 16): # 15 and lower fails first assert
            p = OraclePredictor(digit_str, denom.bit_length())
            r = p.symbol_to_range(s, denom)
            assert r[1] > r[0]
            assert p.val_to_symbol(r[0], denom) == s
            assert p.val_to_symbol(r[1]-1, denom) == s



# Can it handle many symbols?
# Fixed: @pytest.mark.skip(reason="FAILING")
def test_AC_many_symbols():
    fib = [1, 2]
    for i in range(10):
        fib.append(fib[-1] + fib[-2])
    for n_symbols in fib:
        p = Predictor(n_symbols)
        for denom in (n_symbols, n_symbols*2 - 1):
            for s in range(n_symbols):
                r = p.symbol_to_range(s, denom)
                assert r[1] > r[0]
                assert p.val_to_symbol(r[0], denom) == s
                assert p.val_to_symbol(r[1]-1, denom) == s
            

# Widely-varing probabilities?
class ShuffledFibPDF:
    def __init__(self, not_to_exceed=1<<20, seed=42):
        self.seed = seed
        fib = [1, 1]
        while (nf := fib[-1] + fib[-2]) < not_to_exceed:
            fib.append(nf)
        self.fib = fib
        self.rng = random.Random()
        self.reset()

    def reset(self):
        self.fib.sort()
        self.rng.seed(self.seed)

    def __call__(self):
        self.rng.shuffle(self.fib)
        return self.fib[:]      # Return a copy


def test_ShuffledFibPDF():
    f1 = ShuffledFibPDF()
    assert all(v <= 1<<20 for v in f1())
    f2 = ShuffledFibPDF(1<<60)
    assert all(v <= 1<<60 for v in f2())
    assert all(f1v in f2() for f1v in f1())
    assert f2() != f2()
    f1.reset()
    sfs = [f1() for i in range(10)]
    f1.reset()
    for sf in sfs:
        assert sf == f1()
    f3 = ShuffledFibPDF()
    f1.reset()
    for i in range(10):
        f1() == f3()


class CodeBook:
    def __init__(self, symbols):
        if isinstance(symbols, str):
            self.symbols = list(symbols)
        elif isinstance(symbols, (list, tuple)):
            self.symbols = symbols
        else:
            raise TypeError("I need a string or list or tuple please")

    def encode(self, s):
        return [self.symbols.index(c) for c in s]

    def decode(self, v):
        return "".join(self.symbols[i] for i in v)

def test_Codebook(long_pi, long_text):
    b = CodeBook("0123456789")
    e = b.encode(long_pi)
    assert b.decode(e) == long_pi
    assert b.decode(e[:-1]) != long_pi
    b = CodeBook(''.join(c for c in set(long_text)))
    e = b.encode(long_text)
    assert b.decode(e) == long_text


class ShufflingPredictor(ProbPredictor):
    def __init__(self, probs, seed=42):
        super().__init__(len(probs))
        self.orig_probs = probs
        self.seed = seed
        self.rng = random.Random()
        self.reset()

    def reset(self):
        #self.i = 0
        self.probs = self.orig_probs[:]
        self.rng.seed(self.seed)

    def prob(self, symbol):
        return self.probs[symbol]

    def accept(self,symbol):
        self.rng.shuffle(self.probs)
        #self.i += 1
        super().accept(symbol)

    def copy(self):
        r = type(self)(self.probs, self.seed)
        #r.i = self.i
        return r


def test_ShufflingPredictor():
    p = ShufflingPredictor(list(range(30)), seed=17)
    # Incomplete


def AC_wide_probabilities_test(text, precision=16):
    cb = CodeBook(''.join(c for c in set(text)))
    in_data = cb.encode(text)
    vmax = max(cb.encode(text))
    sfp = ShuffledFibPDF()
    probs = sfp()
    while len(probs) <= vmax:
        probs.extend(list(a+b for a, b in zip(sfp(), sfp())))
    probs = probs[:vmax+1]
    logging.debug(f"AC_wide_probabilties_test {vmax=} {probs=}")
    predictor = ShufflingPredictor(probs)
    a = AC(predictor, precision)
    enc = a.to_bin
    dec = a.from_bin
    out_bits = list(enc.bits(in_data))
    logging.debug(f"AC_wide_probabilties_test {len(text)=} {len(out_bits)=}")
    recovered = list(dec.run(out_bits))
    out_text = cb.decode(recovered)
    assert out_text[:len(in_data)] == text
    #assert recovered[:len(in_data)] == text
    #assert len(out_bits) <= len(digit_str) * bpd


def test_AC_wide_probabilities_medium(medium_text):
    AC_wide_probabilities_test(medium_text)

@pytest.mark.slow
def test_AC_wide_probabilities_various_precision_downward(medium_text):
    text = medium_text
    for prec in range(16, 7, -1): # val_to_symbol range assertion fail at precision=5
        AC_wide_probabilities_test(text, prec)

@pytest.mark.slow
def test_AC_wide_probabilities_various_precision_upward(medium_text):
    text = medium_text
    for prec in range(16, 66): # I assume python bigint is coming into play here.
        # We'll have to revisit once it's numpy'd
        AC_wide_probabilities_test(text, prec)

def test_AC_assert_need_enough_precision(medium_text):
    text = medium_text
    with pytest.raises(AssertionError) as xinfo:
        AC_wide_probabilities_test(text, 6)
    assert ", at least 7" in str(xinfo.value)
    AC_wide_probabilities_test(text, 7)



@pytest.mark.slow
def test_AC_wide_probabilities_long(long_text):
    AC_wide_probabilities_test(long_text)



def shufseq(n=30):
    l = list(range(n))
    random.shuffle(l)
    return l

class ShufSeq:
    def __init__(self, n=30):
        self.l = list(range(n))

    def __call__(self):
        random.shuffle(self.l)
        return self.l[:]


def test_shufseq():
    assert shufseq() != shufseq()
    os = ShufSeq()
    s1 = os()
    s2 = os()
    assert s1 != s2
    assert os() != os()
    


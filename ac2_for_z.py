# an attempt at a clear implementation

# goes between alphabet A and binary
# things of note:
#  due to loss of precision, A->B encode ≠ B<-A decode
#  general A to B is hard since you can get zoom in onto a decision boundary
#   so B has to be past-independent.
#  (hence, i'm choosing to set B to binary)
#
#  encoding: (A->B)
#   (using ternary (abc) as an example A)
#   |   |   |   | input alphabet
#   |     |     | output alphabet
#
#   example: bbc
#   |   |###|   | b ->
#   region is not fully within 1 or 0, so we can't output a certain bit
#   but, since region is smaller than 1/2, region is within 01 to 10,
#    so we can output a 0, expecting to later maybe output a 2
#   |     | |###I###|  -> 0
#   region resides within 1, so output 1
#   |   |#######I#######| -> 1
#   can't output more, next symbol
#   |     |  |##I##|  b ->
#   |     |#####I#####| -> 1
#   |     |     | |###| c ->
#   |  |########|       -> 2
#   flush: emit the fewest bits to have the correct last symbol
#                   -> 1
#  input: 0.bbc = 2+3+9 = 14/27 = .518... to .555...
#  output: 0.01121 = 10001 = 17/32 = .53125 to .5625
#                       slight missmatch is due to precision loss in the diagrams
#
# now decoding 100001
#   |   |   |   |
#   |     |#####|  <- 1   mark region but do not zoom in
#   |     |##|  |  <- 0
#   |     |#|   |  <- 0
#   now marked region is entirely within a symbol so we can emit that symbol
#   |   |_|#|   | b <-
#   perform the zoom in (fake emit bits) and recalculate the marked region all the way from the beginning
#      note: recalculation is unnecessary if the fixed point denominator is a power of B (2 for binary)
#          because no precision is lost stepping the marked region
#   |     |#####|###########|
#   |     |#####|##|
#   |     |###| |
#   take next bit
#   |   | |#|   |  <- 0
#   |     |#####| b<-
#   |   |   ||##|  <- 1
#   |   |#######| c <-
#
#
#  things to note to avoid being tricked
#   B is the language whose regions are being transformed to [0,1)
#   A is the language whose distribution is being mapped to [l,h)
#  i.e. since B is binary, the "zoom in" is always by a factor of 2
#

# Refs:
# - Howard & Vitter 94, Arithmetic Coding for Data Compression, CS-1994-09, Duke University https://web.archive.org/web/20131018025150/http://home.tiscali.nl/rajduim/Video%20Compression%20Docs/Arithmetic%20Coding%20For%20Data%20Compression%20(2).pdf

__author__ = "Paul Soulanille <paul@soulanille.net>"
# Hacked on by TAS

import bisect
import numpy as np

def region_overlap(a, b, c, d):
    # [a,b] with [c,d]
    return max(0, min(d, b) - max(a, c) + 1)


class Predictor:
    def __init__(self, n):
        """n even intervals, for this simplest sane case"""
        self.n = n

    def val_to_symbol(self, v, denom):
        # Lookup the symbol whose range v/denom falls within
        return (v * self.n) // denom

    def symbol_to_range(self, s, denom):
        # Gives an increasing tuple of fixed-point probabilities, scaled by denom
        return -((-s * denom) // self.n), -((-(s + 1) * denom) // self.n)

    def accept(self, symbol):
        # update internal model
        pass                    # nothing to do with this stationary example distribution

    def copy(self):
        return self


import itertools


class CDFPredictor(Predictor):
    def __init__(self, dist):
        """A predictor that works from a fixed distribution provided as a bisect-able thing"""
        self.dist = dist
        self.minp = min(filter(lambda v: v > 0, self.pdf_iter))

    @property
    def pdf_iter(self):
        return itertools.chain(
            [self.dist[0]],
            (self.dist[i + 1] - self.dist[i] for i in range(len(self.dist) - 1)),
        )

    def fudged_dist(self, denom):
        assert len(self.dist) <= denom, f"Can't represent {len(self.dist)} symbols with {denom=}.\n" + \
            f"Suggested fix: use a higher precision in constructing AC, at least {len(self.dist).bit_length()+1}"
        if self.dist[-1] <= denom * self.minp:
            return self.dist
        res = []
        p = 0
        for i in range(len(self.dist)):
            d = (self.dist[i] * denom) // self.dist[-1] - p
            d = max(1, min(denom - p - len(self.dist) + i + 1, d))
            p += d
            res.append(p)
        return res

    def val_to_symbol(self, v, denom):
        assert 0 <= v < denom, f"val_to_symbol({v=}, {denom=})"
        dist = self.fudged_dist(denom)
        return bisect.bisect_right(dist, (v * dist[-1]) // denom)

    def symbol_to_range(self, s, denom):
        dist = self.fudged_dist(denom)
        if s >= len(dist) or s < 0:
            raise AssertionError("unknown symbol", s)
        hd = dist[s]
        ld = dist[s - 1] if s > 0 else 0
        d = dist[-1]
        # min l such that (l*d)//denom >= ld
        # ld * denom
        l = -(-(ld * denom) // d)
        # min h such that h*d//denom >= hd
        h = -(-(hd * denom) // d)
        return (l, h)

class PDFPredictor(Predictor):
    # uses numpy
    """subclasses should implement accept(self,symbol) by using the set_cdf_from_pdf method or by setting self.dist directly
    """
    def __init__(self, dist, precision=48):
        """A predictor that works from a fixed distribution provided as a bisect-able thing"""
        self.dist = dist
        self.precision = precision

    def set_cdf_from_pdf(self,pdf):
        self.dist = self.cdf_from_pdf(pdf)

    def cdf_from_pdf(self, pdf):
        pdf = np.array(pdf, dtype=np.float64)
        bias = self.get_lop_bias(pdf)
        assert bias >= 0,f"too small precision, needs at least ceil(log2(num_tokens={len(pdf)}))={len(pdf).bit_length()}"
        pdf += bias
        #pdf *= self.region.one / np.sum(pdf)
        pdf *= (1<<self.precision) / np.sum(pdf)
        cdf = np.cumsum(pdf).astype(np.uint64)
        return cdf

    def get_lop_bias(self, pdf):
        # want min((pdf+bias)/sum(pdf+bias)) >= 2 ulp
        # (min(pdf)+bias) / (sum(pdf) + len(pdf)*bias) >= 2 ulp
        # looser bound:
        #  bias / (sum(pdf) + len(pdf)*bias) >= 2 ulp
        #  (sum(pdf) + len(pdf)*bias) / bias <= 1 / (2 ulp)
        #  sum(pdf) / bias + len(pdf) <= 1 / (2 ulp)
        #  sum(pdf) / bias <= 1 / (2 ulp) - len(pdf)
        #  bias >= sum(pdf) / (1 / (2 ulp) - len(pdf))
        return sum(pdf) / ((1<<self.precision) / 2 - len(pdf))


    def val_to_symbol(self, v, denom):
        assert 0 <= v < denom, f"val_to_symbol({v=}, {denom=})"
        assert denom >= (1<<self.precision>>1), f"{denom=} too small to ensure no catastrophic loss of {self.precision=}"
        v = (v<<self.precision)//denom
        return bisect.bisect_right(self.dist, v)#, key=lambda v:-((-int(v)*denom)>>self.precision))


    def symbol_to_range(self, s, denom):
        if s >= len(self.dist) or s < 0:
            raise AssertionError("unknown symbol", s)
        assert denom >= (1<<self.precision>>1), f"{denom=} too small to ensure no catastrophic loss of {self.precision=}"
        hd = int(self.dist[s])
        ld = int(self.dist[s - 1]) if s > 0 else 0
        d = int(self.dist[-1])
        # min l such that (l*d)//denom >= ld
        # ld * denom
        l = -(-(ld * denom) >> self.precision)
        # min h such that h*d//denom >= hd
        h = -(-(hd * denom) >> self.precision)
        return (l, h)
    def copy(self):
        return type(self)(self.dist,self.precision)
    
class ProbPredictor(CDFPredictor):
    """A Predictor that can handle changing probability distributions"""
    def __init__(self, n):
        self.n = n
        self.dcache: list | None = None

    def prob(self, symbol):
        """Need to implement in most use cases"""
        """ integers please (will be superceded) """
        return 1

    def calc_dist(self):
        p = 0
        self.dcache = []
        for s in range(self.n):
            p += self.prob(s)
            self.dcache.append(p)
        return self.dcache

    @property
    def dist(self):
        if self.dcache is None:
            return self.calc_dist()
        return self.dcache

    @property
    def minp(self):             # minimum non-zero
        return min(filter(lambda v: v > 0, self.pdf_iter))

    def accept(self, symbol):
        self.dcache = None      # flush for later recalc - extend

    def copy(self):
        return self


ternary = Predictor(3)


class AC:
    """ """
    def __init__(self, predictor=ternary, prec=16):
        """ A factory for A_to_bin and A_from_bin
 prec is the probability register precision in bits """
        self.predictor = predictor
        self.precision = prec

    def __repr__(self) -> str:
        return f"AC({repr(self.predictor)} at {self.precision} bits)"

    @property
    def to_bin(self):
        return A_to_bin(self.predictor.copy(), self.precision)

    @property
    def from_bin(self):
        return A_from_bin(self.predictor.copy(), self.precision)


class A_to_bin:
    """Alphabet to binary
    Callable with a symbol (an int), returning a generator of bits
    bits are improper; they are in [0,3]
"""

    def __init__(self, predictor=ternary, prec=16):
        self.predictor = predictor
        self.denom = 1 << prec  # fixed-point one (aka 1.0)
        self.decision = 1 << (prec - 1) # decision boundary for the bits
        self.l = 0                      # Low side of the interval (zero-extended)
        self.h = self.denom - 1         # High side (implicitly one-extended)
        self.emitted_bits = 0           # Counter of bits emitted by emit_bit
        self.debug_log = None           # Optional list for logging stuff into

    def __repr__(self):
        sl = bin(self.l + (self.denom << 1))[3:]
        sh = bin(self.h + (self.denom << 1))[3:]
        return f"A_to_bin([{sl[0]}.{sl[1:]},{sh[0]}.{sh[1:]}])"

    def receive_symbol(self, symbol):
        """Apply the range update (i.e. zoom in), and tell the predictor about the symbol"""
        if self.debug_log:
            self.debug_log.append((self.l, self.h, "recv", symbol))
        w = self.h - self.l + 1
        r = self.predictor.symbol_to_range(symbol, w)
        assert r[1] > r[0], f"{self}.receive_symbol({symbol}) {r=}"
        was_l, was_h = self.l, self.h
        self.h = self.l + r[1] - 1 # Remember ranges are [l, h) aka open ball on the high side
        self.l += r[0]
        assert self.l <= self.h, f"{self}.receive_symbol({symbol}) self.l {was_l}->{self.l}, self.h {was_h}->{self.h}"
        self.predictor.accept(symbol)

    def decide_bit(self):
        l = self.l // self.decision
        # h = self.h//self.decision
        if (self.h - self.l) < self.decision:
            return self.emit_bit(l)

    def emit_bit(self, b):
        if self.debug_log:
            self.debug_log.append((self.l, self.h, "emit", b))
        self.l = self.l * 2 - b * self.denom
        self.h = self.h * 2 + 1 - b * self.denom
        self.emitted_bits += 1
        return b

    def step(self, symbol):
        self.receive_symbol(symbol)
        r = self.decide_bit()
        while r is not None:
            yield r
            r = self.decide_bit()

    def flush(self):
        # emit the shortest bit string that is fully within [l,h]
        while self.l > 0 or self.h + 1 < self.denom:
            l = self.l // self.decision
            if region_overlap(
                self.l, self.h, l * self.decision, (l + 1) * self.decision
            ) < region_overlap(
                self.l, self.h, (l + 1) * self.decision, (l + 2) * self.decision
            ):
                l += 1
            yield self.emit_bit(l)
        self.l = 0
        self.h = self.denom - 1

    def __call__(self, symbol):
        if symbol is None:
            return tuple(self.flush())
        return tuple(self.step(symbol))

    def run(self, symbols, stop=1):
        for s in symbols:
            yield from self.step(s)
        if stop:
            yield from self.flush()

    def encode(self, symbols, stop=1):
        length = 0
        r = 0
        for v in self.run(symbols, stop):
            r <<= 1
            length += 1
            r += v
        return r, length

    @property
    def info(self):
        import math

        return -math.log2((self.h - self.l + 1) / self.denom)

    @property
    def total_encoded_entropy(self):
        return self.emitted_bits + self.info

    @property
    def certain(self):
        return 0 <= self.l and self.h < self.denom

    def bits(self, symbols, stop=1):
        it = self.run(symbols, stop)
        for v in it:
            if self.certain:
                yield v
            else:
                r = v
                l = 1
                for v in it:
                    r <<= 1
                    l += 1
                    r += v
                    if self.certain:
                        break
                while l:
                    l -= 1
                    yield (r >> l) & 1


class A_from_bin:
    def __init__(self, predictor=ternary, prec=16):
        self.predictor = predictor
        self.denom = 1 << prec
        self.decision = 1 << (prec - 1)
        self.l = 0
        self.h = self.denom - 1
        self.lb = 0
        self.hb = self.denom - 1

    def __repr__(self):
        sl = bin(self.l + (self.denom << 1))[3:]
        sh = bin(self.h + (self.denom << 1))[3:]
        slb = bin(self.lb + (self.denom << 1))[3:]
        shb = bin(self.hb + (self.denom << 1))[3:]
        slb = "".join(slb[i] for i in range(len(slb)) if slb[i] == shb[i])
        return f"A_from_bin([{sl[0]}.{sl[1:]},{sh[0]}.{sh[1:]}],{slb[0]}.{slb[1:]})"

    def receive_bit(self, bit):
        w = (self.hb - self.lb + 1) // 2
        self.lb += w * bit
        self.hb = self.lb + w - 1

    def decide_symbol(self):
        w = self.h - self.l + 1
        ls = self.predictor.val_to_symbol(self.lb - self.l, w)
        hs = self.predictor.val_to_symbol(self.hb - self.l, w)
        if ls == hs:
            return self.emit_symbol(ls)

    def emit_symbol(self, s, flushing = False):
        r = self.predictor.symbol_to_range(s, self.h - self.l + 1)
        # assert r[1]-r[0]
        if region_overlap(self.l + r[0], self.l + r[1] - 1, self.lb, self.hb) == 0:
            raise AssertionError(
                "predictor range does not correspond to val",
                self,
                s,
                r,
                (self.l + r[0], self.l + r[1] - 1),
            )
        #
        assert flushing or self.l+r[0] <= self.lb <= self.hb <= self.l+r[1]-1, "region escaped window"
        self.h = self.l + r[1] - 1
        self.l += r[0]
        self.predictor.accept(s)
        return s

    def emit_bit(self):
        l = self.l // self.decision
        if self.h - self.l < self.decision:
            self.l = self.l * 2 - l * self.denom
            self.h = self.h * 2 + 1 - l * self.denom
            self.lb = self.lb * 2 - l * self.denom
            self.hb = self.hb * 2 + 1 - l * self.denom
            return l

    def step(self, bit):
        self.receive_bit(bit)
        r = self.decide_symbol()
        while r is not None:
            while self.emit_bit() is not None:
                pass
            yield r
            r = self.decide_symbol()

    def flush(self):
        # emit the shortest A string that is fully within [lb,hb]
        # (shortest instead of least entropy because it's easier)
        # simple heuristic to not have to worry about 2nd order stuff
        # (replacing shortest with reasonably short)
        def k(s):
            r = self.predictor.symbol_to_range(s, self.h - self.l + 1)
            return region_overlap(
                self.lb - self.l, self.hb - self.l, r[0], r[1] - 1
            ) / (r[1] - r[0])

        while not (self.lb <= self.l and self.h <= self.hb):
            w = self.h - self.l + 1
            ls = self.predictor.val_to_symbol(max(0, self.lb - self.l), w)
            hs = self.predictor.val_to_symbol(min(w-1, self.hb - self.l), w)
            s = max((s for s in range(ls, hs + 1)), key=k)
            yield self.emit_symbol(s,True)
        self.l = 0
        self.h = self.denom - 1
        self.lb = 0
        self.hb = self.denom - 1

    def __call__(self, bit):
        if bit is None:
            return tuple(self.flush())
        return tuple(self.step(bit))

    def run(self, bits, stop=1):
        for b in bits:
            yield from self.step(b)
        if stop:
            yield from self.flush()

    def decode(self, bits, length, stop=1):
        def biter(bits, length):
            while length:
                length -= 1
                yield (bits >> length) & 1

        yield from self.run(biter(bits, length), stop)
        if stop:
            yield from self.flush()


def group_bits(bits, b=8):
    r = 1
    for v in bits:
        r <<= 1
        r |= v
        if r >> b:
            yield r ^ (1 << b)
            r >>= b
    if r > 1:
        while r >> b == 0:
            r <<= 1
        yield r ^ (1 << b)


def ungroup_bits(groups, b=8):
    for g in groups:
        for i in range(b):
            yield (g >> (b - i - 1)) & 1


def nth_order_stats(n, toks, start=-1):
    hist = dict()
    past = tuple(start for i in range(n))
    for t in toks:
        past = (*past[1:], t)
        if past not in hist:
            hist[past] = 0
        hist[past] += 1
    return hist


class History(ProbPredictor):
    def __init__(self, n, past=256, lfunc=lambda r, i, n, p: n * r**3 + 1):
        super().__init__(n)
        self.past = [-1] * past
        self.pasti = 0
        self.n = n
        self.lfunc = lfunc

    def runs(self):
        for i in range(len(self.past)):
            p = self.past[self.pasti - 1 - i]
            r = 0
            for j in range(1, len(self.past) - i):
                if self.past[self.pasti - 1 - i - j] != self.past[self.pasti - j]:
                    break
                r += 1
            if p > 0:
                yield (p, r, i)

    def calc_dist(self):
        self.dcache = [1] * self.n
        for s, r, i in self.runs():
            self.dcache[s] += self.lfunc(r, i, self.n, len(self.past))
        p = 0
        for i in range(len(self.dcache)):
            p += self.dcache[i]
            self.dcache[i] = p
        return self.dcache

    def accept(self, symbol):
        self.past[self.pasti] = symbol
        self.pasti = (self.pasti + 1) % len(self.past)
        return super().accept(symbol)

    def copy(self):
        h = History(self.n, 0, self.lfunc)
        h.past = list(self.past)
        h.pasti = self.pasti
        return h


def measure_compress(
    comp,
    inp,
    print_every_out=100,
    print_every_inp=100,
    save_bits=None,
    inp_cb=lambda t: "",
):
    stats = [0, 0, 0]
    if save_bits is None:
        save_bits = []

    def ini(s=stats):
        for v in inp:
            yield v
            s[2] = v
            s[0] += 1
            if (s[1] % print_every_inp) == 0:
                info = comp.total_encoded_entropy
                print(
                    s[0],
                    "->",
                    info,
                    "   ",
                    info / s[0],
                    " bits/tok ",
                    inp_cb(s[2]),
                    end="        \r",
                )

    def outi(i, s=stats):
        for b in i:
            save_bits.append(b)
            yield b
            s[1] += 1
            if (s[1] % print_every_out) == 0:
                info = comp.total_encoded_entropy
                print(
                    s[0],
                    "->",
                    info,
                    "   ",
                    info / s[0],
                    " bits/tok ",
                    inp_cb(s[2]),
                    end="        \r",
                )

    return bytes(group_bits(outi(comp.bits(ini()))))


class NFA(ProbPredictor):
    def __init__(self, state, transitions):
        self.t = transitions
        self.s = state

    def calc_dist(self):
        self.dcache = self.t[self.s][0]
        return self.dcache

    def accept(self, symbol):
        self.s = self.t[self.s][1][symbol]
        return super().accept(symbol)

    def copy(self):
        return NFA(self.s, self.t)


class PMarkov(ProbPredictor):
    # the idea here is to use a prng to probabilistically decide whether to track the observed
    # successor of an ngram.
    def __init__(self, prng):
        pass


class Markov_up_to_n(ProbPredictor):
    def __init__(
        self,
        n,
        order,
        lfunc=lambda c, o, n, m: c * n * o**3,
        past=[],
        table=None,
    ):
        super().__init__(n)
        self.order = order
        self.table = table if table is not None else dict()
        self.past = tuple(past)
        self.lfunc = lfunc

    def __getitem__(self, k):
        return self.table[k] if k in self.table else 0

    def __setitem__(self, k, v):
        self.table[k] = v

    def accept(self, symbol):
        past = self.past + (symbol,)
        for i in range(len(self.past)):
            self[past[-i - 1 :]] += 1
        self.past = past[-self.order :]
        return super().accept(symbol)

    def prob(self, symbol):
        past = self.past + (symbol,)
        return 1 + sum(
            self.lfunc(self[past[-i - 1 :]], i, self.n, self.order)
            for i in range(len(self.past))
        )

    def copy(self):
        return Markov_up_to_n(
            self.n,
            self.order,
            self.lfunc,
            self.past,
            {k: self.table[k] for k in self.table},
        )


class ModifiedMarkov(Predictor):  # incomplete
    def __init__(self, n, depth=3, dist=None, past=None):
        self.n = n
        self.depth = depth
        # dist structure
        # dist = [times,[dist x n]?]
        # where dist[1][a][1][b][1][c][0] is the number of occurrences of cba
        self.dist: list = [0] if dist is None else dist
        self.past = [] if past is None else past

    def est_prob(self, s, given=[]):
        d = self.dist
        p = d[0]
        for i in range(len(given) - 1):
            if len(d) == 1:
                break
            d = d[1][given[-i - 1]]

    def get_dist(self, denom):
        dist = []
        d = self.dist
        for i in range(len(self.past) - 1):
            if len(d) == 1:
                break
            d = d[1][self.past[-i - 1]]
        den = d[0]
        for s in range(self.n):
            # want p( a | cb ) = p(cba)/p(cb)
            d = self.dist
            for j in range(len(self.past)):
                if len(d) == 1:
                    break
                d = d[1][self.past[-i - 1]]
        return [1, 2, 3]

    def val_to_symbol(self, v, denom):
        import bisect

        return bisect.bisect_right(self.get_dist(denom), v)

    def symbol_to_range(self, s, denom):
        dist = self.get_dist(denom)
        h = dist[s]
        l = dist[s - 1] if s > 0 else 0
        return (l, h)

    def accept(self, symbol):
        self.past = (self.past + [symbol])[-self.depth :]
        d = self.dist
        d[0] += 1
        for i in range(len(self.past)):
            if len(d) == 1:
                d.append([[0] for i in range(self.n)])
            d = d[1][self.past[-i - 1]]
            d[0] += 1

    def copy(self):
        def copyd(v):
            if isinstance(v, list):
                return list(copyd(i) for i in v)
            return v

        return ModifiedMarkov(self.n, self.depth, copyd(self.dist), list(self.past))

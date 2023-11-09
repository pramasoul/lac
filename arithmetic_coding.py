import numpy as np
import math






class ACSampler:
    def __init__(self,precision=48):
        self.region = Region(precision)
        self.accumulator = CarryBuffer()
        self.compress_tokens = None
        self.compress_output = None
        self.decompress_bits = iter(())
        self.decompress_output = None
        self.bits_per_token = None
        self.on_decompress_done = None
        self.on_compress_done = None
    @property
    def decompress_bits(self):
        return self._decompress_bits
    @decompress_bits.setter
    def decompress_bits(self,bits):
        self._decompress_bits = iter(bits)
        self.decompress_done = False
    @property
    def compress_tokens(self):
        return self._compress_tokens
    @compress_tokens.setter
    def compress_tokens(self,toks):
        self._compress_tokens = iter(toks)
        self.compress_done = False
    def reset(self):
        self.region.reset()
        self.accumulator.reset()
        self.d_bits = 0
        self.d_bits_ulp = self.region.one
    def flush_compress(self):
        for bit in self.region.step(1,2,3):
            for b in self.accumulator.add(bit,self.region.definite):
                if self.compress_output: self.compress_output(b)
        for b in self.accumulator.flush():
            if self.compress_output: self.compress_output(b)                    
        self.region.reset()
    def sample(self,pdf):
        pdf = np.array(pdf,dtype=np.float64)
        pdf += self.get_lop_bias(pdf)
        pdf *= self.region.one/np.sum(pdf)
        cdf = np.cumsum(pdf).astype(np.uint64)
        return self.sample_scaled_cdf(cdf)
    def get_lop_bias(self,pdf):
        #want min((pdf+bias)/sum(pdf+bias)) >= 2 ulp
        # (min(pdf)+bias) / (sum(pdf) + len(pdf)*bias) >= 2 ulp
        # looser bound:
        #  bias / (sum(pdf) + len(pdf)*bias) >= 2 ulp
        #  (sum(pdf) + len(pdf)*bias) / bias <= 1 / (2 ulp)
        #  sum(pdf) / bias + len(pdf) <= 1 / (2 ulp)
        #  sum(pdf) / bias <= 1 / (2 ulp) - len(pdf)
        #  bias >= sum(pdf) / (1 / (2 ulp) - len(pdf))
        return sum(pdf) / (self.region.one / 2 - len(pdf))    
    def sample_scaled_cdf(self,cdf):
        assert minpdf:=np.min(mpdf:=np.diff(np.concatenate((np.array([0]),\
                            self.region.map(cdf.astype(float),float(cdf[-1]))-self.region.low)))) > 0,\
                f"cdf has unencodable token {np.argmin(mpdf)} (pdf = {minpdf})."\
                " Perhaps try using get_lop_bias or adding an arange to the cdf."
        if self.compress_tokens:
            try:
                tok = next(self.compress_tokens)
            except StopIteration:
                self.compress_done = True
                if self.on_compress_done: self.on_compress_done()
                tok = 0
            low = int(cdf[tok-1]) if tok else 0
            high = int(cdf[tok])
            denom = int(cdf[-1])
            if self.bits_per_token: self.bits_per_token(self.region.entropy_of(low,high,denom))
            for bit in self.region.step(low,high,denom):
                for b in self.accumulator.add(bit,self.region.definite):
                    if self.compress_output: self.compress_output(b)
            return tok
        else:
            denom = int(cdf[-1])
            lookup = lambda p:np.searchsorted(cdf,self.region.map(p) * denom // self.region.one,side='left')
            while lookup(self.d_bits) != lookup(self.d_bits+self.d_bits_ulp):
                try:
                    bit = next(self.decompress_bits)
                except StopIteration:
                    self.decompress_done = True
                    if self.on_decompress_done: self.on_decompress_done()
                    bit = 0
                self.d_bits_ulp >>= 1
                self.d_bits += bit*self.d_bits_ulp
            tok = lookup(self.d_bits)
            low = int(cdf[tok-1]) if tok else 0
            high = int(cdf[tok])
            if self.bits_per_token: self.bits_per_token(self.region.entropy_of(low,high,denom))
            mask = self.region.one-1
            for bit in self.region.step(low,high,denom):
                self.d_bits_ulp += self.d_bits_ulp + (self.d_bits_ulp == 0)
                self.d_bits = (self.d_bits << 1) & mask
            if self.decompress_output: self.decompress_output(tok)
            return tok
        


class Region:
    def __init__(self,precision = 32):
        self.precision = precision
        self.reset()
    def copy(self,o=None):
        if o is None:
            o = Region(self.precision)
            o.low = self.low
            o.high = self.high
            return o
        else:
            self.precision = o.precision
            self.low = o.low
            self.high = o.high
            return self
    def reset(self):
        self.low = 0
        self.high = self.one - 1
    @property
    def one(self):
        return 1 << self.precision
    @property
    def span(self):
        return self.high - self.low + 1
    @property
    def entropy(self):
        return self.precision - math.log2(self.span)
    def entropy_of(self,l,h,d=None):
        l,h = self.map(l,d),self.map(h,d)
        return math.log2(self.span) - math.log2(h-l)
    def map(self,v,d=None):
        d = d if d is not None else self.one
        return self.low + (self.span * v) // d
    def step(self,l,h,d=None):
        self.low,self.high = self.map(l,d),self.map(h,d)-1
        yield from self.emit()
    def emit(self):
        while self.span * 2 <= self.one:
            bit = self.low >> (self.precision - 1)
            yield bit
            self.low = (self.low << 1) - (bit << self.precision)
            self.high = ((self.high << 1) + 1) - (bit << self.precision)
    @property
    def definite(self):
        return (self.high >> self.precision - 1) == (self.low >> self.precision - 1)
            
class CarryBuffer:
    def __init__(self):
        self.reset()
    def copy(self,o=None):
        if o is None:
            o = CarryBuffer()
            o.buf = self.buf
            o.bits = self.bits
            return o
        else:
            self.buf = o.buf
            self.bits = o.bits
            return self
    def reset(self):
        self.buf = 0
        self.bits = 0
    def add(self,bit,definite=False):
        self.buf = (self.bits << 1) + bit
        self.bits += 1
        if definite:
            yield from self.flush()
    def flush(self):
        while self.bits > 0:
            b = self.buf >> self.bits
            yield b
            self.buf &= b-1
            self.bits -= 1

        
        

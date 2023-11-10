import numpy as np
import math
import bisect





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
        self.reset()
    def __repr__(self):
        l = self.d_bits_ulp.bit_length()
        in_bits_repr = (bin(self.d_bits+self.region.one)+"#")[3:-l]+"-"*l
        if self.compress_tokens:
            in_bits_repr = ""
            mode = "compressing"
        else:
            in_bits_repr = ",in_bits:"+in_bits_repr
            mode = "expanding"
        return f"ACSampler({mode},{self.region},{self.accumulator}{in_bits_repr})"
    @property
    def decompress_bits(self):
        return self._decompress_bits
    @decompress_bits.setter
    def decompress_bits(self,bits):
        self._decompress_bits = iter(bits) if bits is not None else iter(())
        self.decompress_done = False
    @property
    def compress_tokens(self):
        return self._compress_tokens
    @compress_tokens.setter
    def compress_tokens(self,toks):
        self._compress_tokens = iter(toks) if toks is not None else None
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
        assert (minpdf:=np.min(mpdf:=np.diff(np.concatenate((np.array([0]),\
                            self.region.map(cdf.astype(float),float(cdf[-1]))-self.region.low))))) > 0,\
                f"cdf has unencodable token {np.argmin(mpdf)} (pdf = {minpdf})."\
                " Perhaps try using get_lop_bias or adding an arange to the cdf."
        if self.compress_tokens:
            try:
                tok = next(self.compress_tokens)
            except StopIteration:
                self.compress_done = True
                if self.on_compress_done: self.on_compress_done()
                tok = 0
            #print("<-",tok)
            low = int(cdf[tok-1]) if tok else 0
            high = int(cdf[tok])
            denom = int(cdf[-1])
            if self.bits_per_token: self.bits_per_token(self.region.entropy_of(low,high,denom))
            for bit in self.region.step(low,high,denom):
                for b in self.accumulator.add(bit,self.region.definite):
                    if self.compress_output: self.compress_output(b)
                #print("->",bit)
                #print(self)
            return tok
        else:
            denom = int(cdf[-1])
            lookup = lambda p:bisect.bisect_left(cdf,p,key=self.region.map)
            while lookup(self.d_bits) != lookup(self.d_bits+self.d_bits_ulp-1):
                #print(self)
                #print(lookup(self.d_bits),lookup(self.d_bits+self.d_bits_ulp-1),(self.d_bits-self.region.low)/self.region.span,(self.d_bits+self.d_bits_ulp-1-self.region.low)/self.region.span)
                try:
                    bit = next(self.decompress_bits)
                except StopIteration:
                    self.decompress_done = True
                    if self.on_decompress_done: self.on_decompress_done()
                    bit = 0
                self.d_bits_ulp >>= 1
                self.d_bits += bit*self.d_bits_ulp
            #print(lookup(self.d_bits),lookup(self.d_bits+self.d_bits_ulp-1),(self.d_bits-self.region.low)/self.region.span,(self.d_bits+self.d_bits_ulp-1-self.region.low)/self.region.span)                
            tok = lookup(self.d_bits)
            #print(self)
            #print("->>",tok)
            low = int(cdf[tok-1]) if tok else 0
            high = int(cdf[tok])
            if self.bits_per_token: self.bits_per_token(self.region.entropy_of(low,high,denom))
            for bit in self.region.step(low,high,denom):
                #print(self)
                #print("->",bit)
                self.d_bits_ulp += self.d_bits_ulp + (self.d_bits_ulp == 0)
                #assert self.d_bits_ulp <= self.region.one
                self.d_bits = (self.d_bits << 1) - self.region.one*bit
            if self.decompress_output: self.decompress_output(tok)
            return tok
        


class Region:
    def __init__(self,precision = 48):
        self.precision = precision
        self.reset()
    def __repr__(self):
        return f"Region(prec={self.precision},[{self.low/self.one} {(self.high+1)/self.one}])"
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
    def unmap(self,v,d=None):
        d = d if d is not None else self.one
        return (v - self.low) * d // self.span
    def step(self,l,h,d=None):
        self.low,self.high = self.map(l,d),self.map(h,d)-1
        yield from self.emit()
    def emit(self):
        while self.span * 2 <= self.one:
            bit = self.low >> (self.precision - 1)
            self.low = (self.low << 1) - (bit << self.precision)
            self.high = ((self.high << 1) + 1) - (bit << self.precision)
            yield bit
    @property
    def definite(self):
        return self.high < self.one
        #return (self.high >> self.precision - 1) == (self.low >> self.precision - 1)
            
class CarryBuffer:
    def __init__(self):
        self.reset()
    def __repr__(self):
        return f"CarryBuffer({bin(self.buf|(1<<self.bits))[3:]})"
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
        self.buf = (self.buf << 1) + bit
        self.bits += 1
        if definite:
            yield from self.flush()
    def flush(self):
        while self.bits > 0:
            self.bits -= 1
            b = self.buf >> self.bits
            yield b
            self.buf &= (1<<self.bits)-1

        

class packbits:
    def __init__(self,byte_callback):
        self.state = 1
        self.byte_callback = byte_callback
    def __call__(self,bit):
        #msb first
        self.state <<= 1
        self.state |= bit
        if self.state>>8:
            self.byte_callback(self.state&255)
            self.state>>=8
    def flush(self):
        while self.state > 1:
            self(0)

def unpackbits(byte_generator):
    for byte in byte_generator:
        for b in range(8):
            yield (byte>>(7-b))&1


#example
def compress_base_ten(digits):
    import time
    output = bytearray(0)
    sampler = ACSampler()
    def tokgen():
        for d in digits:
            yield int(d)
    sampler.compress_tokens = tokgen()
    sampler.compress_output = packbits(output.append)
    def bpt(v,s=[0,0]):
        s[0] += 1
        s[1] += v
        print(f"\x1b[Kcompressed {s[0]} digits to {s[1]} bits, bpt={v}, avgbpt = {s[1]/s[0]}",end="\r")
    sampler.bits_per_token = bpt
    def comp_done():
        sampler.on_compress_done = None
        sampler.flush_compress()
        sampler.compress_output.flush()
        sampler.bits_per_token = None
        sampler.compress_output = None
        print("\ndone compressing")
    sampler.on_compress_done = comp_done
    
    history = []
    hist_len = 1
    while not sampler.compress_done:
        time.sleep(0.1) #simulate inference time
        logits = np.ones(10,dtype=float)
        pdf = np.exp(logits)
        token = sampler.sample(pdf)
        history = (history+[token])[-hist_len:]

    return output

def decompress_base_ten(data):
    import time
    output = ""
    sampler = ACSampler()
    sampler.decompress_bits = unpackbits(data)
    sampler.decompress_output = lambda tok: None
    def bpt(v,s=[0,0]):
        s[0] += 1
        s[1] += v
        print(f"\x1b[Kdecompressed {s[0]} digits from {s[1]} bits, bpt={v}, avgbpt = {s[1]/s[0]}",end="\r")

        
    sampler.bits_per_token = bpt
    def decomp_done():
        sampler.on_decompress_done = None
        sampler.decompress_output = None
        sampler.bits_per_token = None
        print("\ndone decompressing")

        
    sampler.on_decompress_done = decomp_done
    history = []
    hist_len = 1
    while not sampler.decompress_done:
        time.sleep(0.1) #simulate inference time
        logits = np.ones(10,dtype=float)
        pdf = np.exp(logits)
        token = sampler.sample(pdf)
        output += str(token)
        history = (history+[token])[-hist_len:]

    return output

        
        
    


def to_bin(num,base=10,prec=48):
    def toks():
        for v in num:
            yield "0123456789abcdefghijklmnopqrstuvwxyz".index(v.lower())
    sampler = ACSampler(prec)
    pdf = [1]*base
    sampler.compress_tokens = toks()
    out = []
    sampler.compress_output = out.append
    def done():
        sampler.flush_compress()
        sampler.compress_output = None
    sampler.on_compress_done = done
    while not sampler.compress_done:
        sampler.sample(pdf)
    return out
def from_bin(bits,base=10,prec=48):
    sampler = ACSampler(prec)
    sampler.decompress_bits = bits
    pdf = [1]*base
    out = []
    sampler.decompress_output = out.append
    def done():
        sampler.decompress_output = None
    sampler.on_decompress_done = done
    while not sampler.decompress_done:
        try:
            sampler.sample(pdf)
        except Exception as e:
            return sampler,e,out
    return "".join("0123456789abcdefghijklmnopqrstuvwxyz"[i] for i in out)

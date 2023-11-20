
#eg:
llm = None
def r(model_path="../../llama.cpp/models/llama-2-7b.ggmlv3.q5_1.bin",prec=48):
    import llama_cpp
    global llm
    if llm is None:
        Llama = llama_cpp.Llama
        llm = Llama(model_path=model_path, n_ctx=512)
    return AC(Llama_AC(llm),prec)

from arith_code import *
import numpy as np
class Llama_AC(ProbPredictor):
    def __init__(self,llm,maxtoks=2048):
        super().__init__(0)
        self.llm = llm
        self.overlap = 2
        self.reset()
    def reset(self):
        self.past = [1]
        self.llm.reset()
        self.llm.eval([1])
    def calc_dist(self):
        logits = self.llm._scores[-1]
        #pdf = (np.tanh(logits/2)+1)/2
        pdf = np.exp(logits)
        pdf /= np.sum(pdf)
        self.dcache = np.cumsum(np.clip((pdf*(1<<60)).astype(float),2,None)).astype(int)
        return self.dcache
    def accept(self, symbol):
        self.past.append(symbol)
        if len(self.past) == self.llm.n_ctx():
            self.past = self.past[self.llm.n_ctx()-self.llm.n_ctx()//self.overlap:]
            self.llm.reset()
            self.llm.eval(self.past)
        else:
            self.llm.eval([symbol])
        return super().accept(symbol)
    def copy(self):
        return Llama_AC(self.llm)

    @property
    def minp(self):
        return min(self.dist[0],np.min(np.diff(self.dist)))
    def val_to_symbol(self,v,denom):    
        dist = self.fudged_dist(denom)
        return int(np.searchsorted(dist,(v*int(dist[-1]))//denom,side="right"))
    def symbol_to_range(self, s, denom):
        dist = self.fudged_dist(denom)
        if s >= len(dist) or s < 0:
            raise AssertionError("unknown symbol",s)
        hd = int(dist[s])
        ld = int(dist[s-1]) if s > 0 else 0
        d = int(dist[-1])
        # min l such that (l*d)//denom >= ld
        # ld * denom
        l = -(-(ld*denom)//d)
        # min h such that h*d//denom >= hd
        h = -(-(hd*denom)//d)
        return (l,h)

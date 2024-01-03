"""copy of lac.py hacked up to support lactok_compressor.py"""
##!/usr/bin/env python
"""
Compress using a trained model as a predictor
The name ``lac'' is meant to suggest "LLM Arithmetic Coder"
"""
import argparse
import logging
import os
import pickle
import psutil
import re
import struct
import subprocess
import sys
import zlib

import numpy as np
import torch
from torch.nn import functional as F
import tiktoken

from binascii import hexlify
from contextlib import nullcontext
from gpt_model import GPTConfig, GPT
from pprint import pprint

from config import SingletonConfig


def provide_model(model_name="internal", device="cpu", threads=1):
    r"""
    HACKED by TAS
    After Karpathy
    """
    config = SingletonConfig()
    verbose = hasattr(config, "verbose") and config.verbose
    logging.debug(f"provide_model({model_name=}, {device=}, {threads=}, {verbose=})")
    # -----------------------------------------------------------------------------
    init_from = (
        #"resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        #"gpt2"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        # (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`)
    )
    init_from = (model_name, "resume")[model_name == "internal"]
    ckpt_path = "ckpt-0600.pt"  # HACK to allow setting by configurator.py
    start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    # temperature = (
    #     1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    # )
    seed = 1337
    #device = args.device  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32' or 'bfloat16' or 'float16'
    # dtype = 'float16' # DEBUG
    # FIXME:
    compile = False  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    # for k, v in vars(args).items():
    #     logging.debug(f"provide_model: {k} = {v}")
    #     exec(f"{k} = {v}")

    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast

    # Attempt determinism
    torch.use_deterministic_algorithms(True)
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8
    # (may limit overall performance) or :4096:8 (will increase library
    # footprint in GPU memory by approximately 24MiB).
    if device_type == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # model
    if init_from == "resume":
        verbose and sys.stderr.write(f"ckpt_path {ckpt_path}, device {device}\n")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        gptconf.verbose = verbose
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith("gpt2"):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0), verbose=verbose)

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if (
        init_from == "resume"
        and "config" in checkpoint
        and "dataset" in checkpoint["config"]
    ):  # older checkpoints might not have these...
        meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
        load_meta = os.path.exists(meta_path)
    if load_meta:
        sys.stderr.write(f"Loading meta from {meta_path}...\n")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        logging.info("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    return model, ctx, x


################################################################
# Prediction service


class PredictionService:
    def __init__(self):
        pass

    def restart(self):
        pass

    def accept(self, tok):
        pass

    @property
    def probabilities(self):
        return None


class FlatPredictionService(PredictionService):
    def __init__(self, n_symbols):
        self.n_symbols = n_symbols
        self.pdf = np.ones(n_symbols, dtype=np.float64)

    @property
    def probabilities(self):
        return self.pdf


class CountingPredictionService(PredictionService):
    def __init__(self, n_symbols):
        self.n_symbols = n_symbols
        self.restart()

    def restart(self):
        self.pdf = np.ones(self.n_symbols, dtype=np.float64) * 0.01

    def accept(self, tok):
        self.pdf[tok] += 1.0

    @property
    def probabilities(self):
        return self.pdf



class LLMPredictionService(PredictionService):
    def __init__(self, llm_predictor, idx=torch.tensor([[198]])):
        self.p = llm_predictor
        self.idx = idx.to(self.p.device)
        self._temp_idx = None

    def accept(self, tok):
        assert tok < self.p.model.config.vocab_size
        if self.idx.size(1) < self.p.model.config.block_size:
            self.idx = torch.cat((self.idx, torch.tensor([[tok]]).to(self.idx.device)), 1)
        else:
            if self._temp_idx is None:
                self._temp_idx = self.idx.clone()
            t = self._temp_idx
            # Shift all elements down, losing the first
            t[..., :-1] = self.idx[..., 1:]
            # Replace the last element with the new token
            t[..., -1] = tok
            self.idx.copy_(t)   # Should ping-pong between them instead?

    @property
    def probabilities(self):
        probs = self.p(self.idx)
        self.size = probs.size(0)
        return probs


class LLMPredictor:
    def __init__(self, model, ctx, temperature=1.0):
        self.model = model
        self.ctx = ctx
        self.temperature = temperature
        self.device = model.lm_head.weight.device

    def __call__(self, idx):
        idx = idx.to(self.device)
        assert idx.device == self.device
        with torch.no_grad():
            with self.ctx:
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = (
                    idx
                    if idx.size(1) <= self.model.config.block_size
                    else idx[:, -self.model.config.block_size :]
                )
                # forward the model to get the logits for the index in the sequence
                assert idx_cond.device == self.device
                logits, _ = self.model(idx_cond)
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / self.temperature
                # apply softmax to convert logits to (normalized) probabilities
                probabilities = F.softmax(logits, dim=-1)[-1]
        return probabilities.cpu()



class LLMPredictionServiceProvider:
    # FIXME: where should threads be passed, kept, used?
    def __init__(self, model_provider, threads=1):
        self.model_provider = model_provider
        self.threads = threads
        self._cache = {}

    def __call__(self, model_name, device, threads, temperature=1.0):
        device = device or "cpu"
        k = (model_name, device, temperature)
        if k in self._cache:
            lp, idx = self._cache[k]
        else:
            logging.info(f"LLMPredictionServiceProvider getting a model({device=}, {threads=})\n")
            model, ctx, idx = self.model_provider(model_name=model_name, device=device, threads=threads)
            lp = LLMPredictor(model, ctx, temperature=temperature)
            self._cache[k] = (lp, idx)
        return LLMPredictionService(lp, idx)
        

llm_psp = LLMPredictionServiceProvider(provide_model, 1)
def provide_prediction_service(model_name, device=None, threads=1, temperature=1.0):
    return llm_psp(model_name, device, threads, temperature=temperature)

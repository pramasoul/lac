"""copy of lac.py hacked up to support lactok_compressor.py"""
##!/usr/bin/env python
"""
Compress using a trained model as a predictor
The name ``lac'' is meant to suggest "LLM Arithmetic Coder"
"""
import argparse
import copy
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
from torch.cuda.amp import autocast
from torch.nn import functional as F
import tiktoken

from binascii import hexlify
from contextlib import contextmanager, nullcontext
from gpt_model import GPTConfig, GPT
from pprint import pprint

from config import SingletonConfig

config = SingletonConfig()

def make_torch_deterministic():
    # Attempt determinism
    torch.use_deterministic_algorithms(True)
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8
    # (may limit overall performance) or :4096:8 (will increase library
    # footprint in GPU memory by approximately 24MiB).
    #if device_type == "cuda":
    if True:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def provide_model(model_name="internal", device="cpu", threads=1):
    r"""
    HACKED by TAS
    After Karpathy
    """
    #config = SingletonConfig()
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

    device = torch.device(device)
    device_type = device.type

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    # ctx = (
    #     nullcontext()
    #     if device_type == "cpu"
    #     else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    # )

    # MOVED, and doing it here too causes trouble with cpu model
    # consistency: torch.set_num_threads(threads)
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
        logging.debug("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            start = f.read()
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    return model, autocast(), x


def provide_model_on_cpu(model_name):
    logging.debug(f"provide_model_on_cpu({model_name})")
    model, _, _ = provide_model(model_name, "cpu")
    # assert host_device_of(model) == torch.device(device), \
    #     f"provide_just_model expected provide_model to return a model on {device}, but it's on {host_device_of(model)}"
    # Set to evaluate only, no training, no autograd
    model.eval()
    model.requires_grad_(False)
    return model


def host_device_of(model):
    # For models not distributed across multiple devices
    return next(model.parameters()).device


class ModelOnDeviceCache:
    # A caching model-providing context manager
    # Initialized with a function that provides a model on the CPU given the model name
    # Callable as a context manager with the model_name and device
    # Creates, caches, moves models as necessary
    # FUTURE: manages gross memory use and cache evictions
    def __init__(self, model_on_cpu_provider):
        self.model_on_cpu_provider = model_on_cpu_provider
        self.cpu_model_cache = {}
        self.model_device_cache = {}

    @contextmanager
    def __call__(self, model_name, device="cpu"):
        # FIXME: manage out-of-memory with refcounting and lru eviction
        #log.debug(f"ModelOnDeviceCache instance called with {model_name=}, {device=}")
        device = torch.device(device)
        if (model_name, device) in self.model_device_cache:
            model_on_device = self.model_device_cache[(model_name, device)]
        else:
            if model_name in self.cpu_model_cache:
                model_on_cpu = self.cpu_model_cache[model_name]
            else:
                model_on_cpu = self.model_on_cpu_provider(model_name)
                self.cpu_model_cache[model_name] = model_on_cpu

            if device != torch.device("cpu"):
                model_on_device = copy.deepcopy(model_on_cpu).to(device)
            else:
                model_on_device = model_on_cpu
            self.model_device_cache[(model_name, device)] = model_on_device

            # assert host_device_of(model_on_device) == torch.device(device), \
            # f"ModelOnDeviceCache expected model on {device}, but it's on {host_device_of(model_on_device)}"
        yield model_on_device

    def evict(self, model_name, device):
        del self.model_device_cache[(model_name, torch.device(device))]
        if torch.device(device) == torch.device('cpu'):
            del self.cpu_model_cache[model_name]


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
    # Init with an LLMPredictor and a starting idx
    # .probabilities gets the (unnormalized) probability array for the possible token values
    # accept(tok) to update the history, before getting the new .probabilities given the new tok

    def __init__(self, llm_predictor, idx=torch.tensor([[198]])):
        self.p = llm_predictor
        self.idx = idx.to(self.p.device)
        self._temp_idx = None

    def accept(self, tok):
        assert tok < self.p.model_config.vocab_size
        if self.idx.size(1) < self.p.model_config.block_size:
            self.idx = torch.cat((self.idx, torch.tensor([[tok]]).to(self.idx.device)), 1)
        else:
            if self._temp_idx is None:
                self._temp_idx = self.idx.clone()
            t = self._temp_idx
            # Shift all elements down, losing the first
            t[..., :-1] = self.idx[..., 1:]
            # Replace the last element with the new token
            t[..., -1] = tok
            self.idx.copy_(t)   # FIXME: Should ping-pong between them instead?

    @property
    def probabilities(self):
        probs = self.p(self.idx)
        self.size = probs.size(0)
        return probs


class LLMPredictor:
    # Init with a ModelOnDeviceCache and model name, device and temperature on creation
    # Callable, returning probabilities using the MODC to transiently obtain the model
    def __init__(self, model_cache, model_name, device, temperature=1.0):
        self.mc = model_cache
        self.model_name = model_name
        self.device = torch.device(device)
        self.temperature = temperature
        self._model_config = None
        make_torch_deterministic() # FIXME: where should this really be?

    def __call__(self, idx):
        casting_cm = "autocast" in config.debug and autocast or nullcontext
        with torch.no_grad(), casting_cm(), self.mc(self.model_name, self.device) as model:
            idx = idx.to(self.device)

            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= model.config.block_size
                else idx[:, -model.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            # assert idx_cond.device == self.device
            logits, loss = model(idx_cond)
            if hasattr(config, "model_output_callback"):
                config.model_output_callback(idx_cond, logits)

        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / self.temperature
        if hasattr(config, "mangle_logits"):
            logits = config.mangle_logits(logits)

        # apply softmax to convert logits to (normalized) probabilities
        probabilities = F.softmax(logits, dim=-1)[-1]
        return probabilities  #.cpu()

    @property
    def model_config(self):
        if self._model_config is None:
            with self.mc(self.model_name, self.device) as model:
                self._model_config = copy.deepcopy(model.config)
        return self._model_config


enc = tiktoken.get_encoding("gpt2")
mdc = ModelOnDeviceCache(provide_model_on_cpu)
def model_provider(model_name, device, threads):
    model = mdc(model_name, device)
    start_ids = "\n"
    x = torch.tensor(enc.encode(start_ids), dtype=torch.long, device=device)[None, ...]
    return model, nullcontext(), x

def provide_prediction_service(model_name, device, threads=1, temperature=1.0) -> LLMPredictionService:
    logging.debug(f"provide_prediction_service({model_name}, {device}, {threads=}, {temperature=})")
    torch.set_num_threads(threads) # FIXME: Move this to the right place
    if "threads" in config.debug:
        import pdb
        pdb.set_trace()
    predictor = LLMPredictor(mdc, model_name, device, temperature)
    return LLMPredictionService(predictor)


################################################################
# Surgical tools

# TBD

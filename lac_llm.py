"""copy of lac.py hacked up to support lactok_compressor.py"""
##!/usr/bin/env python
"""
Compress using a trained model as a predictor
The name ``lac'' is meant to suggest "LLM Arithmetic Coder"
"""
import argparse
import json
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


__version_bytes__ = bytes([0, 1])
__version__ = f"{'.'.join(str(int(b)) for b in __version_bytes__)}"


def magic_number_bytes():
    bs = b"LACZ"
    # bit-reverse the bytes
    rv = bytes([int(bin(i | 256)[3:][::-1], 2) for i in bs])
    try:
        rv.decode()
    except UnicodeDecodeError:
        # This is where we want to be
        pass
    else:
        raise ValueError(
            "Our magic number is UTF decodable, which risks collision with real text files"
        )
    finally:
        return rv


def get_nvcc_version_info() -> dict[str]:
    # FIXME: is nvcc necessarily present on any machine that can run our code? NO, e.g. cpu
    try:
        output = subprocess.check_output("nvcc --version", shell=True).decode()
    except subprocess.CalledProcessError:
        release_info = build_info = None
    else:
        # Regular expression to match the release and build information
        release_pattern = r"release (\d+\.\d+), V(\d+\.\d+\.\d+)"
        build_pattern = r"Build (cuda_\d+\.\d+\.r\d+\.\d+/compiler\.\d+_\d+)"

        # Search for matches in the output
        release_match = re.search(release_pattern, output)
        build_match = re.search(build_pattern, output)

        # Extract the matched groups if found
        release_info = release_match.group(2) if release_match else None
        build_info = build_match.group(1) if build_match else None

    return release_info, build_info

def get_python_version_number_string():
    return sys.version.split()[0]


def get_versions() -> dict[str]:
    sys_cuda_release, sys_cuda_build = get_nvcc_version_info()
    rv = {
        "lacz": __version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "sys_cuda_build": sys_cuda_build,
        "sys_cuda_release": sys_cuda_release,
        "python": get_python_version_number_string(),
        "np": np.__version__,
    }
    return rv


def lacz_header() -> bytes:
    rv = []
    rv.append(magic_number_bytes())
    rv.append(__version_bytes__)
    vjz = zlib.compress(json.dumps(get_versions()).encode("utf-8"))
    rv.append(struct.pack("!H", len(vjz)))  # prepend the length
    rv.append(vjz)
    rv = b"".join(rv)
    return rv


def get_header_and_advance(f):
    magic_bytes = f.read(4)
    if magic_bytes != magic_number_bytes():
        raise ValueError(f"Wrong magic number for lacz (got {hexlify(magic_bytes)}, expected {hexlify(magic_number_bytes())}")
    version_bytes = f.read(2)
    zjson_header_len = struct.unpack("!H", f.read(2))[0]
    logging.debug(f"{zjson_header_len=}")
    vjz = f.read(zjson_header_len)
    versions = json.loads(zlib.decompress(vjz))
    return version_bytes, versions


def provide_model(model_name="internal", device="cpu", threads=1, verbose=False):
    r"""
    HACKED by TAS
    After Karpathy
    """
    # -----------------------------------------------------------------------------
    init_from = (
        #"resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        #"gpt2"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        # (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`)
    )
    init_from = (model_name, "resume")[model_name == "internal"]
    ckpt_path = "ckpt-0600.pt"  # HACK to allow setting by configurator.py
    start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    temperature = (
        1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    )
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

    device = "cuda:3"
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
        verbose and print(f"ckpt_path {ckpt_path}, device {device}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith("gpt2"):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

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
        print(f"Loading meta from {meta_path}...")
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

    def accept(self, tok):
        assert tok < self.size
        self.idx = torch.cat((self.idx, torch.tensor([[tok]]).to(self.idx.device)), 1)

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
    def __init__(self, model_provider, threads=1):
        self.model_provider = model_provider
        self.threads = threads
        self._cache = {}

    def __call__(self, model_name, device, threads):
        k = (model_name, device)
        if k in self._cache:
            lp, idx = self._cache[k]
        else:
            model, ctx, idx = self.model_provider(device=device, threads=threads)
            lp = LLMPredictor(model, ctx)
            self._cache[k] = (lp, idx)
        return LLMPredictionService(lp, idx)
        


device = "cuda:2"

# model, ctx, idx = provide_model(device=device, threads=16)
# def provide_prediction_service(device="cpu"):
#     lp = LLMPredictor(model, ctx)
#     lps = LLMPredictionService(lp, idx)
#     return lps

llm_psp = LLMPredictionServiceProvider(provide_model, 1)
def provide_prediction_service(model_name, device="cpu", threads=1):
    return llm_psp(model_name, device, threads)

#
################################################################



if False:

    if args.decompress and args.format in ["auto", "bighead"]:
        input_version_bytes, input_versions = get_header_and_advance(header_source)
        input_version_str = f"{'.'.join(str(int(b)) for b in input_version_bytes)}"
        # Is this something we can decompress?
        # FIXME: get smarter
        if input_version_bytes != __version_bytes__:
            s = f"Input file is version {input_version_str} and I am {__version__}. I'm not smart enough to know if I can do this."
            logging.warning(s)
            sys.stderr.write(s)

    device = args.device
    temperature = args.temperature

    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    args.verbose and print(f"<|endoftext|> is {eot_token}")


if False:
    device = "cuda:3"
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

    model, dtype, idx = provide_model(device=device, threads=16)
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

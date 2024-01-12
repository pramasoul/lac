"""Experiments on lac"""
# import argparse
# import copy
import logging
# import os
# import pickle
# import psutil
# import re
# import struct
# import subprocess
# import sys
# import zlib

import numpy as np
import torch
# from torch.cuda.amp import autocast
# from torch.nn import functional as F
# import tiktoken

# from binascii import hexlify
# from contextlib import contextmanager, nullcontext
# from gpt_model import GPTConfig, GPT
# from pprint import pprint

from pathlib import Path

from config import SingletonConfig

config = SingletonConfig()

class MNAL:
    """mmap'ed numpy array log"""
    def __init__(self, filename, capacity, flush_interval=1):
        self.filename = filename
        self.capacity = capacity
        self.flush_interval = flush_interval
        self.mma = None
        self.ix = 0

    def __repr__(self):
        return f"MNAL({self.filename}, {self.ix}{['','(FULL)'][self.ix == self.capacity]}, {self.flush_interval})"

    def log(self, a):
        if self.ix >= self.capacity:
            return
        "log_logit_logging" in config.debug and logging.info(f"log: {self}, {a.shape=}")
        if self.mma is None:
            self.mma = np.memmap(self.filename, dtype=a.dtype, mode='w+', shape=(self.capacity,) + a.shape)
            "log_logit_logging" in config.debug and logging.info(f"log creating mma {self}, {a.shape=} {a.dtype=}")
        self.mma[self.ix] = a
        self.ix += 1
        if self.ix == self.capacity:
            "log_logit_logging" in config.debug and logging.info(f"log {self} at capacity, no more will be logged here")
            self.sync()
        if self.ix % self.flush_interval == 0:
            self.sync()

    def sync(self):
        "log_logit_logging" in config.debug and logging.info(f"sync {self}")
        if self.mma is not None:
            self.mma.flush()
        
    def __del__(self):
        self.sync()


def log_model_output():
    # Model output callback is called as config.model_output_callback(idx, logits)
    #config.model_output_callback = lambda idx, logits: logging.info(f"model {idx.shape=}, {idx[0, -1]=}, {logits.shape=} {len(logits.squeeze())=}")
    def log_it(idx, logits):
        "log_logit_logging" in config.debug and logging.info(f"model {idx.shape=}, {idx[0, -1]=}, {logits.shape=} {len(logits.squeeze())=}")
        logits_mnal.log(logits.cpu().numpy())
        idx_mnal.log(idx[:, -1].cpu().numpy())

    capacity = 1<<17 # Enough for Frankenstein
    interval = 10
    log_dir = Path(".").resolve()
    for s in config.debug:
        if s.startswith("log_dir:"):
            log_dir = Path(s.split(":", maxsplit=1)[1]).expanduser().resolve()
            break
    "log_logit_logging" in config.debug and logging.info(f"logging to {log_dir}")
    logits_mnal = MNAL(log_dir/"logits.mnal", capacity, interval)
    idx_mnal = MNAL(log_dir/"idx.mnal", capacity, interval)
    config.model_output_callback = log_it


def alter_stdout_chunk_size():
    for s in config.debug:
        if s.startswith("stdout_chunk_size:"):
            size = int(s.split(":", maxsplit=1)[1])
            import tlacz
            tlacz.stdout_chunk_size = size
            logging.info(f"altered stdout_chunk_size to {tlacz.stdout_chunk_size}")
            break


def attach_experiments(experiment_names):
    # Configure selected experiments
    for name in experiment_names:
        if name in globals() and callable(globals()[name]):
            logging.info(f"Attaching experiment {name}")
            v = eval(f"{name}()")
            if v:
                logging.info(f"Experiment {name}() returned {v}")
        else:
            logging.warning(f"No method for experiment {name}")

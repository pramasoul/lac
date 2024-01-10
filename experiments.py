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

from config import SingletonConfig

config = SingletonConfig()

def log_model_output():
    # Model output callback is called as config.model_output_callback(idx, logits)
    config.model_output_callback = lambda idx, logits: logging.info(f"model {idx.shape=}, {idx[0, -1]=}, {logits.shape=} {len(logits.squeeze())=}")


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

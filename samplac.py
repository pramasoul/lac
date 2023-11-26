"""
Compress using a trained model as a predictor
"""
import argparse
import json
import os
import pickle
import struct
import subprocess
import sys
import zlib

import numpy as np
import torch
import tiktoken

from pprint import pprint
from contextlib import nullcontext
from modelac import GPTConfig, GPT
from ac_for_z import ACSampler, packbits, unpackbits


__version_bytes__ = bytes([0, 1])
__version__ = f"{'.'.join(str(int(b)) for b in __version_bytes__)}"
#__version__ = "0." + __version__ # While under development

def magic_number_bytes():
    bs = b'LACZ'
    #bit-reverse the bytes
    rv = bytes([int(bin(i|256)[3:][::-1],2) for i in bs])
    try:
        rv.decode()
    except UnicodeDecodeError:
        # This is where we want to be
        pass
    else:
        raise ValueError("Our magic number is UTF decodable, which risks collision with real text files")
    finally:
        return rv

def get_versions() -> dict[str]:
    rv = {
        "lacz": __version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "sys_cuda": subprocess.check_output("nvcc --version", shell=True).decode(),
        "python": sys.version,
        "np": np.__version__
    }
    return rv

def lacz_header() -> bytes:
    rv = []
    rv.append(magic_number_bytes())
    rv.append(__version_bytes__)
    vjz = zlib.compress(json.dumps(get_versions()).encode('utf-8'))
    rv.append(struct.pack('!H', len(vjz))) # put in the length
    rv.append(vjz)
    rv = b''.join(rv)
    return rv


def get_header_and_advance(f):
    magic_bytes = f.read(4)
    if magic_bytes != magic_number_bytes():
        raise ValueError("Wrong magic number for lacz")
    version_bytes = f.read(2)
    zjson_header_len = struct.unpack('!H', f.read(2))[0]
    print(f"{zjson_header_len=}")
    vjz = f.read(zjson_header_len)
    versions = json.loads(zlib.decompress(vjz))
    return version_bytes, versions


def provide_model(args):
    r"""
    HACKED by TAS
    After Karpathy
    """
    # -----------------------------------------------------------------------------
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    ckpt_path = 'ckpt-0600.pt'  # HACK to allow setting by configurator.py
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    seed = 1337
    device = args.device # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    #dtype = 'float16' # DEBUG
    #FIXME:
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    #ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if init_from == 'resume':
        args.verbose and print(f"ckpt_path {ckpt_path}, device {device}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    return model, dtype, x


from toktst import tokens_encoded_from

def main(argv):
    parser = argparse.ArgumentParser(description="Arithmetic compression with LLM predictor.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='-',
                        help="Input filename, or '-' for stdin")
    parser.add_argument('-o', '--output', type=str, default='-',
                        help="Output filename, or '-' for stdout")
    parser.add_argument('--flushalot',
                        action='store_true', help="flush output frequently")
    parser.add_argument('-d', '--decompress',
                        action='store_true', help="decompress rather than compress")
    parser.add_argument('--device', type=str, default='cpu',
                        help="device to run the model, e.g 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.")
    parser.add_argument('--threads', type=int, default=4,
                        help="number of threads to use if device is cpu")
    parser.add_argument('-T', '--temperature', type=float, default=1.0,
                        help="model's logits scaling")
    parser.add_argument('--precision', type=int, default=48,
                        help="fractional bits in the probability register")
    parser.add_argument('-v', '--verbose', default=0,
                        action='count', help="verbosity about internals")

    args = parser.parse_args()

    versions = get_versions()
    args.verbose and pprint(versions)

    if args.output and args.output != '-':
        outf = open(args.output, 'w+b')
    else:
        outf = sys.stdout.buffer

    device = args.device
    temperature = args.temperature

    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.encode('<|endoftext|>', allowed_special={"<|endoftext|>"})[0]
    args.verbose and print(f"<|endoftext|> is {eot_token}")

    sampler = ACSampler(precision=args.precision,
                        end_of_text_token=eot_token)
    if args.verbose > 1:
        def bpt(v,s=[0,0]):
            s[0] += 1
            s[1] += v
            print(f"\x1b[Kcompressed {s[0]} tokens to {s[1]:.2f} bits, bpt {v:8.3f}, avgbpt {s[1]/s[0]:6.3f}",end="\r")
            #print(f"compressed {s[0]} tokens to {s[1]} bits, bpt={v}, avgbpt = {s[1]/s[0]}")
        sampler.bits_per_token = bpt
        
    if args.decompress:
        def input_generator(in_name):
            def int_yielder(f):
                b = f.read(1)
                while b:
                    yield int.from_bytes(b, 'big')
                    b = f.read(1)
                
            if in_name and in_name != '-':
                with open(in_name, 'rb') as inf:
                    yield from int_yielder(inf)
            else:
                yield from int_yielder(sys.stdin.buffer)
        sampler.decompress_bits = unpackbits(input_generator(args.input))
        #enc = tiktoken.get_encoding("gpt2")
        def decompressed_token_writer(tok):
            outf.write(enc.decode_single_token_bytes(tok))
            if outf == sys.stdout.buffer or args.flushalot:
                outf.flush()
        sampler.decompress_output = decompressed_token_writer
        def bpt(v,s=[0,0]):
            s[0] += 1
            s[1] += v
            print(f"\x1b[Kdecompressed {s[0]} digits from {s[1]} bits, bpt={v}, avgbpt = {s[1]/s[0]}",end="\r")
            #print(f"decompressed {s[0]} digits from {s[1]} bits, bpt={v}, avgbpt = {s[1]/s[0]}")

        sampler.bits_per_token = args.verbose > 1 and bpt or None
        def decomp_done():
            sampler.on_decompress_done = None
            sampler.decompress_output = None
            sampler.bits_per_token = None
            args.verbose and print("\ndone decompressing")
            if outf != sys.stdout.buffer:
                outf.close()
        sampler.on_decompress_done = decomp_done

    else:
        # run compression
        def tokens_with_eot():
            yield from tokens_encoded_from(args.input)
            yield eot_token
            yield eot_token # FIXME: needed twice
        sampler.compress_tokens = tokens_with_eot()
        def compressed_byte_writer(v):
            outf.write(bytes((v,)))
            if outf == sys.stdout.buffer or args.flushalot:
                outf.flush()
        #sampler.compress_output = packbits(lambda v: outf.write(bytes((v,))))
        sampler.compress_output = packbits(compressed_byte_writer)
        def comp_done():
            sampler.on_compress_done = None
            sampler.flush_compress()
            sampler.compress_output.flush()
            sampler.bits_per_token = None
            sampler.compress_output = None
            args.verbose and print("\ndone compressing")
            if outf != sys.stdout.buffer:
                outf.close()
        sampler.on_compress_done = comp_done

    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

    # Attempt determinism
    torch.use_deterministic_algorithms(True)
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8
    # (may limit overall performance) or :4096:8 (will increase library
    # footprint in GPU memory by approximately 24MiB).
    if device_type == 'cuda':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    model, dtype, x = provide_model(args)
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    # Common to both compression and decompression, we operate the model in predicition mode
    with torch.no_grad():
        with ctx:
            y = model.compresserate(x, sampler, device=device, temperature=temperature)

if __name__ == "__main__":
    main(sys.argv)

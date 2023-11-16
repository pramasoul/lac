import argparse
import struct
import sys
import tiktoken

import numpy as np

def uint16s_from(file_name):
    # Define the dtype for the uint16, using '>u2' for big-endian (network order)
    dtype = np.dtype('u2')  # Use '<u2' for little-endian if needed
    if file_name != '-':
        f = open(file_name, 'rb')
    while True:
        if file_name == '-':
            buf = sys.stdin.buffer.read(4096)
            chunk = np.frombuffer(buf, dtype=dtype)
        else:
            chunk = np.fromfile(f, dtype=dtype, count=2048)
        if not chunk.size:  # End of file
            break
        for value in chunk:
            yield value

def tokens_encoded_from(file_name):
    enc = tiktoken.get_encoding("gpt2")
    # Without reading the entire file before tokenizing, we cannot
    # trust the tokenization at the margins of each buffer of text
    # we tokenize.
    # FIXME: for now we trust that there will be line breaks often
    # enough, and that a line break is a guarantee of token break.
    if file_name == '-':
        inf = sys.stdin
    else:
        inf = open(file_name, 'r')
    for line in inf:
        for tok in enc.encode(line):
            yield tok

def tokens_from(file_name):
    yield from uint16s_from(file_name)

def decoded_tokens_from(file_name):
    enc = tiktoken.get_encoding("gpt2")
    for tok in tokens_from(file_name):
        yield enc.decode_single_token_bytes(tok)

def write_uint16s_to_binary_stream(int_list, output_stream):
    # ChatGPT-4
    # Format string: 'H' for uint16, repeated for each integer in the list
    format_str = f'{len(int_list)}H'
    # Pack the list of integers into a bytes object
    packed_data = struct.pack(format_str, *int_list)
    # Write the packed data to the output stream
    output_stream.write(packed_data)

def write_one_uint16_to_binary_stream(v, output_stream):
    packed_data = struct.pack('H', v)
    output_stream.write(packed_data)

def main(argv):
    parser = argparse.ArgumentParser(description="A quick tokenizer test.")
    parser.add_argument('-i', '--input', type=str, default='-',
                        help="Input filename, or '-' for stdin")
    parser.add_argument('-o', '--output', type=str, default='-',
                        help="Output filename, or '-' for stdout")
    parser.add_argument('-d', '--decompress',
                        action='store_true', help="decompress rather than compress")
    #parser.add_argument('-c', '--stdout',
    #                    action='store_true', help="write to stdout")

    args = parser.parse_args()

    if args.output and args.output != '-':
        outf = open(args.output, 'w+b')
    else:
        outf = sys.stdout.buffer

    if args.decompress:
        for v in decoded_tokens_from(args.input):
            outf.write(v)
    else:
        for b in tokens_encoded_from(args.input):
            write_one_uint16_to_binary_stream(b, outf)

if __name__ == "__main__":
    main(sys.argv)

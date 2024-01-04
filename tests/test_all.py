import pytest
import pathlib
import subprocess

# A hack to do comprehensive testing using all the hardware
# Nothing automatic about it
# Run with -n 4 so they work in parallel

@pytest.fixture(scope="session")
def outf():
    with open("./all_test.out", "a") as f:
        yield f

def test_w_cpu(outf):
    things = ("ac2",
              "lac_llm",
              "lactok_compressor",
              "tlacz",
              "tlac",
    )
    rcs = [thing_test(outf, thing) for thing in things]
    assert not any(rcs)

def test_w_cuda1(outf):
    tms = [("lactok_compressor", "internal"),
           ("tlacz", "internal"),
           ("tlac", "internal"),
           ("lactok_compressor", "gpt2"),
           ("tlacz", "gpt2"),
           ("tlac", "gpt2"),
    ]
    rcs = [thing_test(outf, thing, f"--device cuda:1 --model {model} -n 4") for thing, model in tms]
    assert not any(rcs)

def test_w_cuda2(outf):
    tms = [("lactok_compressor", "gpt2-medium"),
           ("tlacz", "gpt2-medium"),
           ("tlac", "gpt2-medium"),
           ("lactok_compressor", "gpt2-large"),
           ("tlacz", "gpt2-large"),
           ("tlac", "gpt2-large"),
    ]
    rcs = [thing_test(outf, thing, f"--device cuda:2 --model {model} -n 4") for thing, model in tms]
    assert not any(rcs)

def test_w_cuda3(outf):
    tms = [("lactok_compressor", "gpt2-xl"),
           ("tlacz", "gpt2-xl"),
           ("tlac", "gpt2-xl"),
    ]
    rcs = [thing_test(outf, thing, f"--device cuda:3 --model {model}") for thing, model in tms]
    assert not any(rcs)



def thing_test(outf, thing, args=""):
    out = subprocess.run(
        f"date; pytest tests/test_{thing}.py {args} -v", shell=True, capture_output=True
    )
    outf.write("#"*80 + "\n#" + " "*78 + "#\n" +
               f"pytest tests/test_{thing}.py {args} -v\n" +
               out.stdout.decode('utf8') +
               " "*78 + "#\n" + "#"*80 + "\n#")
    return out.returncode

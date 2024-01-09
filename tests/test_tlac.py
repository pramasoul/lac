# Test tlacz compressor as a program



import pytest
#import torch
import ctypes
import json
import logging
import subprocess
import tempfile

from binascii import hexlify, unhexlify
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Callable, List

import numpy as np

# import unittest.mock as mock
from unittest.mock import mock_open

import tlacz as lac
from tlacz import LacFile


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)  # StreamHandler logs to console


# Now you can use logging in your tests
def test_example():
    logging.debug("This is a debug message.")


def test_create_file(tmp_path):
    CONTENT = "content"
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text(CONTENT, encoding="utf-8")
    assert p.read_text(encoding="utf-8") == CONTENT
    assert len(list(tmp_path.iterdir())) == 1


@contextmanager
def mock_file(initial_data=None):
    """A context manager for mocking file operations using BytesIO."""
    # Create a BytesIO object with optional initial data
    if initial_data is not None:
        if isinstance(initial_data, str):
            initial_data = initial_data.encode()  # Convert string to bytes
        file_obj = BytesIO(initial_data)
    else:
        file_obj = BytesIO()

    try:
        yield file_obj
    finally:
        # Perform any necessary cleanup (if any)
        file_obj.close()


def test_mock_file():
    # Usage example
    with mock_file("Initial text") as mock_file_obj:
        # Perform file operations
        mock_file_obj.write(b" More data")
        mock_file_obj.seek(0)
        content = mock_file_obj.read()
        assert content.decode() == " More dataxt"


def test_read_file(mocker):
    mock_file_contents = "mock file data"
    mocker.patch("builtins.open", mock_open(read_data=mock_file_contents))

    with open("mock_file.txt", "r") as file:
        data = file.read()

    assert data == mock_file_contents


def test_write_file(mocker):
    mock_write = mock_open()
    mocker.patch("builtins.open", mock_write)

    data_to_write = "data to be written"
    with open("mock_file.txt", "w") as file:
        file.write(data_to_write)

    mock_write.assert_called_once_with("mock_file.txt", "w")
    mock_write().write.assert_called_once_with(data_to_write)


def test_multiple_file_mocks(mocker):
    # Define the mock file contents for different files
    file_mocks = {
        "foo": mock_open(read_data="data from foo").return_value,
        "bar": mock_open(read_data="data from bar").return_value,
    }

    # Side effect function to return different mocks based on filename
    def side_effect(filename, *args, **kwargs):
        return file_mocks[filename]

    # Patch 'open' with the side effect
    mocker.patch("builtins.open", side_effect=side_effect)

    # Test reading from 'foo'
    with open("foo", "r") as file:
        data = file.read()
    assert data == "data from foo"

    # Test reading from 'bar'
    with open("bar", "r") as file:
        data = file.read()
    assert data == "data from bar"


def my_function():
    # Example function that reads from one file and writes to another
    with open("read_file.txt", "r") as f:
        logging.debug(f"my_function read_file.txt is {f}")
        data = f.read()

    logging.debug(f"my_function data is {data}")

    with open("write_file.txt", "w") as f:
        logging.debug(f"my_function write_file.txt is {f}")
        f.write(data[::-1])  # Just an example transformation


def test_my_function(mocker):
    mock_file_handles = {
        ("read_file.txt", "r"): mock_open(read_data="Hello World").return_value,
        ("write_file.txt", "w"): mock_open().return_value,
    }

    def side_effect(filename, mode):
        rv = mock_file_handles[(filename, mode)]
        logging.debug(f"side_effect({filename}, {mode}) returning {rv}")
        return rv

    mocker.patch("builtins.open", side_effect=side_effect, create=True)

    my_function()

    # Check if the read file was opened correctly
    logging.debug(f"{mock_file_handles[('read_file.txt', 'r')]=}")
    mock_file_handles[("read_file.txt", "r")].read.assert_called_once_with()

    # Add any other assertions you need for the write file
    mock_file_handles[("write_file.txt", "w")].write.assert_called_once_with(
        "dlroW olleH"
    )


@pytest.fixture(scope="session")
def tmp_dir_s(tmp_path_factory):
    base_temp = tmp_path_factory.getbasetemp()
    temp_dir = Path(tempfile.mkdtemp(dir=base_temp))
    yield temp_dir
    # Optional: Cleanup after the session
    # shutil.rmtree(temp_dir)

def test_tmp_dir_s(tmp_dir_s):
    file_path = tmp_dir_s / "example.txt"
    with open(file_path, "w") as file:
        file.write("Hello, world!")


@pytest.fixture(scope="session")
def lac_name():
    return "./tlacz.py"

# Create a short.lacz file for multiple subsequent tests
@pytest.fixture(scope="session")
def short_lacz_c(lac_name, short_text, tmp_dir_s):
    file_path = tmp_dir_s / "short_c.lacz"
    out = subprocess.check_output(
        f"echo -n {short_text} | {lac_name} -o {file_path}", shell=True
    )
    return file_path

def test_short_lacz_c(lac_name, short_lacz_c, short_text):
    out = subprocess.check_output(
        f"{lac_name} -d {short_lacz_c} --stdout", shell=True
    )
    assert out.decode() == short_text 
    
@pytest.fixture(scope="session")
def short_lacz_g(lac_name, short_text, tmp_dir_s):
    file_path = tmp_dir_s / "short_g.lacz"
    out = subprocess.check_output(
        f"echo -n {short_text} | {lac_name} --device cuda -o {file_path}", shell=True
    )
    return file_path

def test_short_lacz_g(lac_name, short_lacz_g, short_text):
    out = subprocess.check_output(
        f"{lac_name} --device cuda -d {short_lacz_g} --stdout", shell=True
    )
    assert out.decode() == short_text 
    

def test_lac_runnable(lac_name):
    out = subprocess.check_output(f"{lac_name} -h", shell=True).decode()
    assert out.startswith("usage: ")


def test_lac_compress_LacFile_read_c(lac_name, short_lacz_c, short_text):
    assert LacFile(short_lacz_c).read().decode('utf8') == short_text

def test_lac_compress_LacFile_read_g(lac_name, short_lacz_g, short_text):
    assert LacFile(short_lacz_g, device="cuda").read().decode('utf8') == short_text


    """
Various ways lac can be invoked from the command line:
  * lac -o foo bar baz => error "-o/--output can only be used with a single input file"
  * lac - => compress stdin to stdout
  * lac -c - => compress stdin to stdout
  * lac -d - => decompress stdin to stdout
  * lac foo.txt => compress foo.txt to foo.txt.lacz
  * lac -d foo.txt.lacz => decompress foo.txt.lacz to foo.txt
  * lac -o bar foo => compress foo to bar
  * lac -o foo bar -d => decompress bar to foo
  * lac -d bar -o foo => decompress bar to foo
  * lac -o bar - => compress stdin to bar
  * lac -o foo -d => decompress stdin to foo
  * lac -o - - => compress stdin to stdout
  * lac -o - => compress stdin to stdout
  * lac foo bar => compress foo to foo.lacz, and bar to bar.lacz
  * lac -d foo.lacz bar.lacz => decompress foo.lacz to foo, and bar.lacz to bar
  * lac -c foo bar => compress foo and bar to stdout
    """

def test_just_echo():
    test_text = "This is only a test."
    out = subprocess.check_output(
        f"echo -n {test_text}", shell=True
    )
    assert out.decode() == test_text 

def test_output_with_multiple_inputs_complains(lac_name):
    # lac -o foo bar baz => error "-o/--output can only be used with a single input file"
    result = subprocess.run([lac_name, "-o", "foo", "bar", "baz"], stderr=subprocess.PIPE, text=True)
    assert result.returncode != 0
    assert "-o/--output can only be used with a single input file" in result.stderr


def test_stdin_to_stdout_1(lac_name):
    # lac - => compress stdin to stdout
    # lac -d - => decompress stdin to stdout
    test_text = "This is only a test."
    out = subprocess.check_output(
        f"echo -n {test_text} | {lac_name} - | {lac_name} -d", shell=True
    )
    assert out.decode() == test_text 


def test_stdin_to_stdout_2(lac_name):
    # lac -c - => compress stdin to stdout
    # lac -d - => decompress stdin to stdout
    test_text = "This is only a test."
    out = subprocess.check_output(
        f"echo -n {test_text} | {lac_name} -c - | {lac_name} -d -", shell=True
    )
    assert out.decode() == test_text 

def test_stdin_to_stdout_3(lac_name):
    # lac -o - - => compress stdin to stdout
    test_text = "This is only a test."
    out = subprocess.check_output(
        f"echo -n {test_text} | {lac_name} -o - - | {lac_name} -d -", shell=True
    )
    assert out.decode() == test_text 

def test_stdin_to_stdout_4(lac_name):
    # lac -o - - => compress stdin to stdout
    test_text = "This is only a test."
    out = subprocess.check_output(
        f"echo -n {test_text} | {lac_name} -o - | {lac_name} -d -", shell=True
    )
    assert out.decode() == test_text 


def test_decompress_file_cpu(lac_name, short_text, short_lacz_c):
    # lac -d foo.lacz --stdout => decompress foo.lacz to stdout
    out = subprocess.check_output(
        f"{lac_name} -d {short_lacz_c} --stdout", shell=True
    )
    assert out.decode() == short_text

def test_decompress_file_to_file_cpu(lac_name, short_text, short_lacz_c, tmp_path):
    # lac -d foo.lacz -o outfile => decompress foo.lacz to outfile
    tmp_file = tmp_path / "outfile"
    out = subprocess.check_output(
        f"{lac_name} -d {short_lacz_c} -o {tmp_file}", shell=True
    )
    assert tmp_file.read_text(encoding="utf-8") == short_text

def test_decompress_file_to_file_cpu_t1(lac_name, short_text, short_lacz_c, tmp_path):
    # lac -d foo.lacz -o outfile
    tmp_file = tmp_path / "outfile"
    out = subprocess.check_output(
        f"{lac_name} --threads=1 -d {short_lacz_c} -o {tmp_file}", shell=True
    )
    assert tmp_file.read_text(encoding="utf-8") == short_text

def test_compress_file_to_file_dot_lacz(lac_name, short_text, tmp_path):
    # lac foo.txt => compress foo.txt to foo.txt.lacz
    # lac -d foo.txt.lacz => decompress foo.txt.lacz to foo.txt
    test_text = short_text
    input_file_path = tmp_path / "foo.txt"
    input_file_path.write_text(test_text, encoding="utf-8")
    assert input_file_path.read_text(encoding="utf-8") == test_text
    compress_out = subprocess.check_output(
        f"{lac_name} {input_file_path}", shell=True
    ).decode()
    compressed_file_path = tmp_path / "foo.txt.lacz"
    input_file_path.unlink()
    with pytest.raises(FileNotFoundError):
        input_file_path.read_bytes()
    assert len(compressed_file_path.read_bytes()) > 4
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {compressed_file_path}", shell=True
    ).decode()
    assert input_file_path.read_text(encoding="utf-8") == test_text

def test_compress_foo_decompress_bar(lac_name, short_text, tmp_path):
    # lac -o bar foo => compress foo to bar
    # lac -o foo bar -d => decompress bar to foo
    # lac -d bar -o foo => decompress bar to foo
    test_text = short_text
    foo = tmp_path / "foo"
    foo.write_text(test_text, encoding="utf-8")
    assert foo.read_text(encoding="utf-8") == test_text
    bar = tmp_path / "bar"
    compress_out = subprocess.check_output(
        f"{lac_name} -o {bar} {foo}", shell=True
    ).decode()
    foo.unlink()
    with pytest.raises(FileNotFoundError):
        foo.read_bytes()
    assert len(bar.read_bytes()) > 4
    decompress_out = subprocess.check_output(
        f"{lac_name} -o {foo} {bar} -d", shell=True
    ).decode()
    assert foo.read_text(encoding="utf-8") == test_text
    foo.unlink()
    with pytest.raises(FileNotFoundError):
        foo.read_bytes()
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {bar} -o {foo}", shell=True
    ).decode()
    assert foo.read_text(encoding="utf-8") == test_text


def test_compress_stdin_to_bar(tmp_path, lac_name, short_text):
    # lac -o bar - => compress stdin to bar
    test_text = short_text
    bar = tmp_path / "bar"
    foo = tmp_path / "foo"
    foo.write_text(test_text)
    compress_out = subprocess.check_output(
        f"{lac_name} -o {bar} < {foo}", shell=True
    ).decode()
    assert len(bar.read_bytes()) > 4
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {bar} --stdout", shell=True
    ).decode()
    assert decompress_out == test_text


def test_decompress_stdin_to_foo(tmp_path, lac_name, short_lacz_c, short_text):
    test_text = short_text
    foo = tmp_path / "foo"
    # lac -o foo -d => decompress stdin to foo
    decompress_out = subprocess.check_output(
        f"{lac_name} -o {foo} -d < {short_lacz_c}", shell=True
    ).decode()
    assert foo.read_text(encoding="utf-8") == test_text


def test_compress_foo_bar(tmp_path, lac_name):
    # lac foo bar => compress foo to foo.lacz, and bar to bar.lacz
    foo = tmp_path / "foo"
    foo.write_text("This is foo")
    bar = tmp_path / "bar"
    bar.write_text("This is bar")
    compress_out = subprocess.check_output(
        f"{lac_name} {foo} {bar}", shell=True
    ).decode()
    foo_lacz = tmp_path / "foo.lacz"
    bar_lacz = tmp_path / "bar.lacz"
    assert foo_lacz.read_bytes() != bar_lacz.read_bytes()
    assert len(foo_lacz.read_bytes()) > 4
    assert len(bar_lacz.read_bytes()) > 4
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {foo_lacz} --stdout", shell=True
    ).decode()
    assert decompress_out == "This is foo"
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {bar_lacz} --stdout", shell=True
    ).decode()
    assert decompress_out == "This is bar"
    # lac -d foo.lacz bar.lacz => decompress foo.lacz to foo, and bar.lacz to bar
    foo.unlink()
    with pytest.raises(FileNotFoundError):
        foo.read_bytes()
    bar.unlink()
    with pytest.raises(FileNotFoundError):
        bar.read_bytes()
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {foo_lacz} {bar_lacz}", shell=True
    ).decode()
    assert foo.read_text(encoding="utf-8") == "This is foo"
    assert bar.read_text(encoding="utf-8") == "This is bar"

def test_compress_bar_foo(tmp_path, lac_name):
    # lac foo bar => compress foo to foo.lacz, and bar to bar.lacz
    foo = tmp_path / "foo"
    foo.write_text("This is foo")
    bar = tmp_path / "bar"
    bar.write_text("This is bar")
    compress_out = subprocess.check_output(
        f"{lac_name} {bar} {foo}", shell=True
    ).decode()
    foo_lacz = tmp_path / "foo.lacz"
    bar_lacz = tmp_path / "bar.lacz"
    assert foo_lacz.read_bytes() != bar_lacz.read_bytes()
    assert len(foo_lacz.read_bytes()) > 4
    assert len(bar_lacz.read_bytes()) > 4
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {foo_lacz} --stdout", shell=True
    ).decode()
    assert decompress_out == "This is foo"
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {bar_lacz} --stdout", shell=True
    ).decode()
    assert decompress_out == "This is bar"
    # lac -d foo.lacz bar.lacz => decompress foo.lacz to foo, and bar.lacz to bar
    foo.unlink()
    with pytest.raises(FileNotFoundError):
        foo.read_bytes()
    bar.unlink()
    with pytest.raises(FileNotFoundError):
        bar.read_bytes()
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {foo_lacz} {bar_lacz}", shell=True
    ).decode()
    assert foo.read_text(encoding="utf-8") == "This is foo"
    assert bar.read_text(encoding="utf-8") == "This is bar"

def test_compress_foo_bar_baz(tmp_path, lac_name):
    # lac foo bar => compress foo to foo.lacz, and bar to bar.lacz
    foo = tmp_path / "foo"
    foo.write_text("This is foo")
    bar = tmp_path / "bar"
    bar.write_text("This is bar")
    baz = tmp_path / "baz"
    baz.write_text("This is baz")
    compress_out = subprocess.check_output(
        f"{lac_name} {foo} {bar} {baz}", shell=True
    ).decode()
    foo_lacz = tmp_path / "foo.lacz"
    bar_lacz = tmp_path / "bar.lacz"
    baz_lacz = tmp_path / "baz.lacz"
    assert foo_lacz.read_bytes() != bar_lacz.read_bytes() != baz_lacz.read_bytes()
    assert len(foo_lacz.read_bytes()) > 4
    assert len(bar_lacz.read_bytes()) > 4
    assert len(baz_lacz.read_bytes()) > 4
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {foo_lacz} --stdout", shell=True
    ).decode()
    assert decompress_out == "This is foo"
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {bar_lacz} --stdout", shell=True
    ).decode()
    assert decompress_out == "This is bar"
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {baz_lacz} --stdout", shell=True
    ).decode()
    assert decompress_out == "This is baz"
    # lac -d foo.lacz bar.lacz => decompress foo.lacz to foo, and bar.lacz to bar
    foo.unlink()
    with pytest.raises(FileNotFoundError):
        foo.read_bytes()
    bar.unlink()
    with pytest.raises(FileNotFoundError):
        bar.read_bytes()
    baz.unlink()
    with pytest.raises(FileNotFoundError):
        baz.read_bytes()
    decompress_out = subprocess.check_output(
        f"{lac_name} -d {foo_lacz} {bar_lacz} {baz_lacz}", shell=True
    ).decode()
    assert foo.read_text(encoding="utf-8") == "This is foo"
    assert bar.read_text(encoding="utf-8") == "This is bar"
    assert baz.read_text(encoding="utf-8") == "This is baz"


def test_compress_foo_bar_to_stdout(tmp_path, lac_name):
    # lac -cd foo bar => compress foo and bar to stdout
    foo = tmp_path / "foo"
    foo.write_text("This is foo")
    bar = tmp_path / "bar"
    bar.write_text("This is bar")
    compress_out = subprocess.check_output(
        f"{lac_name} -c {foo} {bar} | {lac_name} -d", shell=True
    ).decode()
    assert compress_out == "This is foo" + "This is bar"





def test_lac_compress_file_to_file_decompress_to_stdout_cpu(tmp_path, lac_name):
    test_text = "This is only a test."
    input_file_path = tmp_path / "in.txt"
    input_file_path.write_text(test_text, encoding="utf-8")
    assert input_file_path.read_text(encoding="utf-8") == test_text
    compressed_file_path = tmp_path / "out.lacz"
    output_file_path = tmp_path / "out.txt"
    compress_out = subprocess.check_output(
        f"{lac_name} {input_file_path} -o {compressed_file_path}", shell=True
    ).decode()
    decompress_out = subprocess.check_output(
        f"{lac_name} {compressed_file_path} -d -o -", shell=True
    ).decode()
    assert decompress_out == test_text


@pytest.mark.slow
def test_lac_compress_file_to_file_decompress_cpu_raw(tmp_path, lac_name):
    test_text = "This is only a test."
    input_file_path = tmp_path / "in.txt"
    input_file_path.write_text(test_text, encoding="utf-8")
    assert input_file_path.read_text(encoding="utf-8") == test_text
    compressed_file_path = tmp_path / "out.lacz"
    output_file_path = tmp_path / "out.txt"
    compress_out = subprocess.check_output(
        f"{lac_name} -i {input_file_path} -o {compressed_file_path} -F raw", shell=True
    ).decode()
    decompress_out = subprocess.check_output(
        f"{lac_name} -i {compressed_file_path} -F raw -d", shell=True
    ).decode()
    assert decompress_out == test_text
    # result = subprocess.run(["{lac_name}", "-i", str(input_file_path), "-o", str(compressed_file_path)],
    #                         stdout=subprocess.PIPE,
    #                         stderr=subprocess.PIPE,
    #                         text=True)

    # result = subprocess.run(["{lac_name}", "-i", str(compressed_file_path), "-o", str(output_file_path), "-d"],
    #                         stdout=subprocess.PIPE,
    #                         stderr=subprocess.PIPE,
    #                         text=True)
    # #assert result.stdout == test_text
    # assert output_file_path.read_text(encoding="utf-8") == test_text


@pytest.mark.slow
def test_lac_compress_file_to_file_decompress_cpu_auto(tmp_path, lac_name):
    test_text = "This is only a test."
    input_file_path = tmp_path / "in.txt"
    input_file_path.write_text(test_text, encoding="utf-8")
    assert input_file_path.read_text(encoding="utf-8") == test_text
    compressed_file_path = tmp_path / "out.lacz"
    output_file_path = tmp_path / "out.txt"
    compress_out = subprocess.check_output(
        f"{lac_name} -i {input_file_path} -o {compressed_file_path}", shell=True
    ).decode()
    decompress_out = subprocess.check_output(
        f"{lac_name} -i {compressed_file_path} -d", shell=True
    ).decode()
    assert decompress_out == test_text


@pytest.mark.slow
def test_lac_compress_file_to_file_decompress_cuda(tmp_path, lac_name):
    test_text = "This is only a test."
    input_file_path = tmp_path / "in.txt"
    input_file_path.write_text(test_text, encoding="utf-8")
    assert input_file_path.read_text(encoding="utf-8") == test_text
    compressed_file_path = tmp_path / "out.lacz"
    output_file_path = tmp_path / "out.txt"
    compress_out = subprocess.check_output(
        f"{lac_name} -i {input_file_path} -o {compressed_file_path} --device cuda",
        shell=True,
    ).decode()
    decompress_out = subprocess.check_output(
        f"{lac_name} -i {compressed_file_path} --device cuda -d", shell=True
    ).decode()
    assert decompress_out == test_text


@pytest.fixture(scope="session")
def short_text():
    return r"Hello, world!"


@pytest.fixture(scope="session")
def medium_text():
    return r"""A variety of specific techniques for arithmetic coding have
historically been covered by US patents, although various well-known
methods have since passed into the public domain as the patents have
expired. Techniques covered by patents may be essential for
implementing the algorithms for arithmetic coding that are specified
in some formal international standards. When this is the case, such
patents are generally available for licensing under what is called
"reasonable and non-discriminatory" (RAND) licensing terms (at least
as a matter of standards-committee policy). In some well-known
instances, (including some involving IBM patents that have since
expired), such licenses were available for free, and in other
instances, licensing fees have been required. The availability of
licenses under RAND terms does not necessarily satisfy everyone who
might want to use the technology, as what may seem "reasonable" for a
company preparing a proprietary commercial software product may seem
much less reasonable for a free software or open source project."""


@pytest.fixture(scope="session")
def long_text():
    return r"""You will rejoice to hear that no disaster has accompanied the
commencement of an enterprise which you have regarded with such evil
forebodings. I arrived here yesterday, and my first task is to assure
my dear sister of my welfare and increasing confidence in the success
of my undertaking.

I am already far north of London, and as I walk in the streets of
Petersburgh, I feel a cold northern breeze play upon my cheeks, which
braces my nerves and fills me with delight. Do you understand this
feeling? This breeze, which has travelled from the regions towards
which I am advancing, gives me a foretaste of those icy climes.
Inspirited by this wind of promise, my daydreams become more fervent
and vivid. I try in vain to be persuaded that the pole is the seat of
frost and desolation; it ever presents itself to my imagination as the
region of beauty and delight. There, Margaret, the sun is for ever
visible, its broad disk just skirting the horizon and diffusing a
perpetual splendour. There—for with your leave, my sister, I will put
some trust in preceding navigators—there snow and frost are banished;
and, sailing over a calm sea, we may be wafted to a land surpassing in
wonders and in beauty every region hitherto discovered on the habitable
globe. Its productions and features may be without example, as the
phenomena of the heavenly bodies undoubtedly are in those undiscovered
solitudes. What may not be expected in a country of eternal light? I
may there discover the wondrous power which attracts the needle and may
regulate a thousand celestial observations that require only this
voyage to render their seeming eccentricities consistent for ever. I
shall satiate my ardent curiosity with the sight of a part of the world
never before visited, and may tread a land never before imprinted by
the foot of man. These are my enticements, and they are sufficient to
conquer all fear of danger or death and to induce me to commence this
laborious voyage with the joy a child feels when he embarks in a little
boat, with his holiday mates, on an expedition of discovery up his
native river. But supposing all these conjectures to be false, you
cannot contest the inestimable benefit which I shall confer on all
mankind, to the last generation, by discovering a passage near the pole
to those countries, to reach which at present so many months are
requisite; or by ascertaining the secret of the magnet, which, if at
all possible, can only be effected by an undertaking such as mine.

These reflections have dispelled the agitation with which I began my
letter, and I feel my heart glow with an enthusiasm which elevates me
to heaven, for nothing contributes so much to tranquillise the mind as
a steady purpose—a point on which the soul may fix its intellectual
eye. This expedition has been the favourite dream of my early years. I
have read with ardour the accounts of the various voyages which have
been made in the prospect of arriving at the North Pacific Ocean
through the seas which surround the pole. You may remember that a
history of all the voyages made for purposes of discovery composed the
whole of our good Uncle Thomas’ library. My education was neglected,
yet I was passionately fond of reading. These volumes were my study
day and night, and my familiarity with them increased that regret which
I had felt, as a child, on learning that my father’s dying injunction
had forbidden my uncle to allow me to embark in a seafaring life.

These visions faded when I perused, for the first time, those poets
whose effusions entranced my soul and lifted it to heaven. I also
became a poet and for one year lived in a paradise of my own creation;
I imagined that I also might obtain a niche in the temple where the
names of Homer and Shakespeare are consecrated. You are well
acquainted with my failure and how heavily I bore the disappointment.
But just at that time I inherited the fortune of my cousin, and my
thoughts were turned into the channel of their earlier bent.

Six years have passed since I resolved on my present undertaking. I
can, even now, remember the hour from which I dedicated myself to this
great enterprise. I commenced by inuring my body to hardship. I
accompanied the whale-fishers on several expeditions to the North Sea;
I voluntarily endured cold, famine, thirst, and want of sleep; I often
worked harder than the common sailors during the day and devoted my
nights to the study of mathematics, the theory of medicine, and those
branches of physical science from which a naval adventurer might derive
the greatest practical advantage. Twice I actually hired myself as an
under-mate in a Greenland whaler, and acquitted myself to admiration. I
must own I felt a little proud when my captain offered me the second
dignity in the vessel and entreated me to remain with the greatest
earnestness, so valuable did he consider my services."""

def do_test_lac_compress_file_to_file_decompress_cuda_raw(tmp_path, test_text, lac_name):
    input_file_path = tmp_path / "in.txt"
    input_file_path.write_text(test_text, encoding="utf-8")
    assert input_file_path.read_text(encoding="utf-8") == test_text
    compressed_file_path = tmp_path / "out.lacz"
    output_file_path = tmp_path / "out.txt"
    compress_out = subprocess.check_output(
        f"{lac_name} -i {input_file_path} -o {compressed_file_path} -F raw --device cuda",
        shell=True,
    ).decode()
    decompress_out = subprocess.check_output(
        f"{lac_name} -i {compressed_file_path} -F raw --device cuda -d", shell=True
    ).decode()
    assert decompress_out == test_text
    assert compressed_file_path.stat().st_size / len(test_text) < 0.2

@pytest.mark.slow
def test_lac_compress_medium_file_to_file_decompress_cuda_raw(tmp_path, medium_text, lac_name):
    do_test_lac_compress_file_to_file_decompress_cuda_raw(tmp_path, medium_text, lac_name)

@pytest.mark.slower
def test_lac_compress_long_file_to_file_decompress_cuda_raw(tmp_path, long_text, lac_name):
    do_test_lac_compress_file_to_file_decompress_cuda_raw(tmp_path, long_text, lac_name)


# @pytest.mark.skip(reason="Implement me")

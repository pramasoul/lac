# Test lac compressor

import pytest
import torch
import ctypes
import json
import logging
import subprocess

from binascii import hexlify, unhexlify
from contextlib import contextmanager
from io import BytesIO
from typing import Callable, List

import numpy as np

# import unittest.mock as mock
from unittest.mock import mock_open

import lac


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


# @pytest.fixture(scope="session")
def brief_text_file(tmp_path):
    return tmp_file


def test_lacz_header():
    header = lac.lacz_header()
    with mock_file(header) as f:
        version_bytes, versions = lac.get_header_and_advance(f)
        assert f.tell() == len(header)
    logging.debug(f"{versions=}")
    assert len(version_bytes) == 2
    assert version_bytes[0] == 0  # Pre-release
    assert isinstance(versions, dict)
    # Verify all the entries we expect are actually present
    assert (
        set(
            [
                "cuda",
                "cudnn",
                "lacz",
                "np",
                "python",
                "sys_cuda_build",
                "sys_cuda_release",
                "torch",
            ]
        )
        - set(versions.keys())
        == set()
    )


def test_lac_runnable():
    out = subprocess.check_output("./lac.py -h", shell=True).decode()
    assert out.startswith("usage: ")


@pytest.mark.slow
def test_lac_compress_file_to_file_decompress_cpu_raw(tmp_path):
    test_text = "This is only a test."
    input_file_path = tmp_path / "in.txt"
    input_file_path.write_text(test_text, encoding="utf-8")
    assert input_file_path.read_text(encoding="utf-8") == test_text
    compressed_file_path = tmp_path / "out.lacz"
    output_file_path = tmp_path / "out.txt"
    compress_out = subprocess.check_output(
        f"./lac.py -i {input_file_path} -o {compressed_file_path} -F raw", shell=True
    ).decode()
    decompress_out = subprocess.check_output(
        f"./lac.py -i {compressed_file_path} -F raw -d", shell=True
    ).decode()
    assert decompress_out == test_text
    # result = subprocess.run(["./lac.py", "-i", str(input_file_path), "-o", str(compressed_file_path)],
    #                         stdout=subprocess.PIPE,
    #                         stderr=subprocess.PIPE,
    #                         text=True)

    # result = subprocess.run(["./lac.py", "-i", str(compressed_file_path), "-o", str(output_file_path), "-d"],
    #                         stdout=subprocess.PIPE,
    #                         stderr=subprocess.PIPE,
    #                         text=True)
    # #assert result.stdout == test_text
    # assert output_file_path.read_text(encoding="utf-8") == test_text


@pytest.mark.slow
def test_lac_compress_file_to_file_decompress_cpu_auto(tmp_path):
    test_text = "This is only a test."
    input_file_path = tmp_path / "in.txt"
    input_file_path.write_text(test_text, encoding="utf-8")
    assert input_file_path.read_text(encoding="utf-8") == test_text
    compressed_file_path = tmp_path / "out.lacz"
    output_file_path = tmp_path / "out.txt"
    compress_out = subprocess.check_output(
        f"./lac.py -i {input_file_path} -o {compressed_file_path}", shell=True
    ).decode()
    decompress_out = subprocess.check_output(
        f"./lac.py -i {compressed_file_path} -d", shell=True
    ).decode()
    assert decompress_out == test_text


@pytest.mark.slow
def test_lac_compress_file_to_file_decompress_cuda(tmp_path):
    test_text = "This is only a test."
    input_file_path = tmp_path / "in.txt"
    input_file_path.write_text(test_text, encoding="utf-8")
    assert input_file_path.read_text(encoding="utf-8") == test_text
    compressed_file_path = tmp_path / "out.lacz"
    output_file_path = tmp_path / "out.txt"
    compress_out = subprocess.check_output(
        f"./lac.py -i {input_file_path} -o {compressed_file_path} --device cuda",
        shell=True,
    ).decode()
    decompress_out = subprocess.check_output(
        f"./lac.py -i {compressed_file_path} --device cuda -d", shell=True
    ).decode()
    assert decompress_out == test_text


@pytest.fixture
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


@pytest.fixture
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

def do_test_lac_compress_file_to_file_decompress_cuda_raw(tmp_path, test_text):
    input_file_path = tmp_path / "in.txt"
    input_file_path.write_text(test_text, encoding="utf-8")
    assert input_file_path.read_text(encoding="utf-8") == test_text
    compressed_file_path = tmp_path / "out.lacz"
    output_file_path = tmp_path / "out.txt"
    compress_out = subprocess.check_output(
        f"./lac.py -i {input_file_path} -o {compressed_file_path} -F raw --device cuda",
        shell=True,
    ).decode()
    decompress_out = subprocess.check_output(
        f"./lac.py -i {compressed_file_path} -F raw --device cuda -d", shell=True
    ).decode()
    assert decompress_out == test_text
    assert compressed_file_path.stat().st_size / len(test_text) < 0.2

@pytest.mark.slow
def test_lac_compress_medium_file_to_file_decompress_cuda_raw(tmp_path, medium_text):
    do_test_lac_compress_file_to_file_decompress_cuda_raw(tmp_path, medium_text)

@pytest.mark.slower
def test_lac_compress_long_file_to_file_decompress_cuda_raw(tmp_path, long_text):
    do_test_lac_compress_file_to_file_decompress_cuda_raw(tmp_path, long_text)


# @pytest.mark.skip(reason="Implement me")

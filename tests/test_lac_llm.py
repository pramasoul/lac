"""Test script for lac_llm"""

import logging
import sys

import numpy as np
import torch

import pytest

#from binascii import hexlify, unhexlify
from contextlib import contextmanager
from io import BytesIO, DEFAULT_BUFFER_SIZE
#from typing import Callable, List, Tuple

from unittest.mock import mock_open

#import tok_compressor as tc
#import lactok_compressor as latc
from lac_llm import FlatPredictionService, CountingPredictionService
import lac_llm as ll


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


def test_llm_predictor():
    lp = ll.LLMPredictor(ll.model, ll.ctx)
    assert np.allclose(lp(torch.tensor([[42]])), lp(torch.tensor([[42]])))
    assert not np.allclose(lp(torch.tensor([[42]])), lp(torch.tensor([[137]])))
    lp2 = ll.LLMPredictor(ll.model, ll.ctx)
    assert np.allclose(lp(torch.tensor([[42]])), lp2(torch.tensor([[42]])))
    assert np.allclose(lp(torch.tensor([[137]])), lp2(torch.tensor([[137]])))
    assert np.allclose(lp(torch.tensor([[137, 196]])), lp2(torch.tensor([[137, 196]])))
    assert not np.allclose(lp(torch.tensor([[42, 137, 196]])), lp(torch.tensor([[42, 137, 196, 777]])))


def test__llm_prediction_service_1():
    lp = ll.LLMPredictor(ll.model, ll.ctx)
    lps = ll.LLMPredictionService(lp, ll.idx)
    p = lps.probabilities
    tk_log = []
    tk = torch.topk(p, 3)
    tk_log.append(tk)
    lps.accept(tk.indices[1])
    p = lps.probabilities
    tk = torch.topk(p, 3)
    tk_log.append(tk)
    lps.accept(tk.indices[1])
    p = lps.probabilities
    tk = torch.topk(p, 3)
    tk_log.append(tk)
    lps.accept(tk.indices[1])
    assert lps.idx.tolist() == [[198, 464, 749, 2219]]

def test_llm_prediction_service_2():
    lps = ll.provide_prediction_service()
    p = lps.probabilities
    tk_log = []
    tk = torch.topk(p, 3)
    tk_log.append(tk)
    lps.accept(tk.indices[1])
    p = lps.probabilities
    tk = torch.topk(p, 3)
    tk_log.append(tk)
    lps.accept(tk.indices[1])
    p = lps.probabilities
    tk = torch.topk(p, 3)
    tk_log.append(tk)
    lps.accept(tk.indices[1])
    assert lps.idx.tolist() == [[198, 464, 749, 2219]]


def test_llm_prediction_service_multi_1():
    # two different prediction services
    lps1 = ll.provide_prediction_service()
    lps2 = ll.provide_prediction_service()
    assert lps1 != lps2

    # Newborn twins
    p1 = lps1.probabilities
    p2 = lps2.probabilities
    assert p1.allclose(p2)

    # First service, round 1
    tk = torch.topk(p1, 3)
    t = tk.indices[1]           # 2nd most probable
    assert t == 464
    lps1.accept(t)
    # Acceptance changes probabilities
    prev_p1 = p1
    p1 = lps1.probabilities
    assert not p1.allclose(prev_p1)

    # Second service, do same
    tk = torch.topk(p2, 3)
    t = tk.indices[1]           # 2nd most probable
    assert t == 464
    lps2.accept(t)
    p2 = lps2.probabilities

    # Second service, round 2
    assert p2.allclose(p1)      # Same path, expect same probabilities
    tk = torch.topk(p2, 3)
    t = tk.indices[0]           # most probable
    assert t == 717
    lps2.accept(t)
    p2 = lps2.probabilities

    # First service, round 2
    assert not p2.allclose(p1)  # Different path (number of steps), expect different probabilities
    tk = torch.topk(p2, 3)
    t = tk.indices[2]           # 3rd most probable
    assert t == 286
    lps1.accept(t)
    p1 = lps1.probabilities

    assert not p1.allclose(p2)




@pytest.mark.skip(reason="unfinished")
def test_llm_prediction_service_multi_2():
    n = 3
    lpss = [ll.provide_prediction_service() for i in range(n)]
    tk_log = []
    for i in range(4):
        tken = []
        for lps in lpss:
            p = lps.probabilities
            tk = torch.topk(p, 3)
            tken.apppend(dk)
            lps.accept(tk.indices[1])

    assert lps.idx.tolist() == [[198, 464, 749, 2219]]


def test_flat_prediction_service():
    for n in (1,2,31,1024,50257,123456):
        ps =  FlatPredictionService(n)
        assert np.sum(ps.probabilities) == n
        ps.accept(n-1)
        ps.restart()
        assert np.sum(ps.probabilities) == n


def test_counting_prediction_service():
    for n in (1,2,31,1024,50257,123456):
        ps =  CountingPredictionService(n)
        assert np.allclose(np.sum(ps.probabilities), n * 0.01)
        ps.accept(n-1)
        assert np.allclose(np.sum(ps.probabilities), n * 0.01 + 1.0)
        assert ps.probabilities[n-1] == 1.01
        ps.restart()
        assert np.allclose(np.sum(ps.probabilities), n * 0.01)

        



    

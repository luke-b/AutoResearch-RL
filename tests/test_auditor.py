import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auditor.causality_auditor import check_causality_leak

def test_clean_slice():
    code = "def get_batch(data, i):\n    return data[i:i+64]"
    assert check_causality_leak(code) == False

def test_dirty_forward_index():
    code = "def eval(data, i):\n    return data[i+1]"
    assert check_causality_leak(code) == True

def test_dirty_shift():
    code = "def eval(data):\n    return data.shift(-1)"
    assert check_causality_leak(code) == True

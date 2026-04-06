import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.orchestrator import Orchestrator

def test_smoke_test_pass():
    orc = Orchestrator()
    assert orc.run_smoke_test("def foo(): pass") == True

def test_smoke_test_fail():
    orc = Orchestrator()
    assert orc.run_smoke_test("def foo() pass") == False

def test_capacity_simulation():
    orc = Orchestrator()
    # 12M parameters in int6 should be ~9MB
    size = orc.simulate_compression_and_capacity("print('hi')", num_parameters=12_000_000)
    assert size < 16_000_000
    assert size > 8_000_000

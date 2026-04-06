import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gpu_cluster.sprt import SPRTFilter

def test_sprt_no_abort_early():
    # Not enough points yet
    filter = SPRTFilter(sota_threshold=1.0)
    assert filter.update_and_check(50, 1.5) == False
    assert filter.update_and_check(100, 1.4) == False

def test_sprt_abort_bad_curve():
    filter = SPRTFilter(sota_threshold=0.9)
    # Give it data that clearly asymptotes > 1.0
    for step in [50, 100, 150, 200, 250]:
        loss = 2.0 * (step)**(-0.5) + 1.1
        should_abort = filter.update_and_check(step, loss)
        if should_abort:
            break

    assert should_abort == True

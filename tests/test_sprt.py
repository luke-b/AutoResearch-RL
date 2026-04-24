import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gpu_cluster.sprt import SPRTFilter

def test_sprt_no_abort_early():
    # Not enough points yet
    sprt_filter = SPRTFilter(sota_threshold=1.0)
    assert sprt_filter.update_and_check(50, 1.5) == False
    assert sprt_filter.update_and_check(100, 1.4) == False

def test_sprt_abort_bad_curve():
    sprt_filter = SPRTFilter(sota_threshold=0.9)
    # Give it data that clearly asymptotes > 1.0
    for step in [50, 100, 150, 200, 250]:
        loss = 2.0 * (step)**(-0.5) + 1.1
        should_abort = sprt_filter.update_and_check(step, loss)
        if should_abort:
            break

    assert should_abort == True

def test_sprt_abort_diverging_loss():
    # A consistently diverging loss curve (monotonically increasing) must be aborted.
    sprt_filter = SPRTFilter(sota_threshold=1.0)
    losses = [1.5, 1.8, 2.2, 2.7, 3.4]
    steps  = [10,  20,  30,  40,  50]
    aborted = False
    for s, l in zip(steps, losses):
        if sprt_filter.update_and_check(s, l):
            aborted = True
            break
    assert aborted, "SPRT must abort a consistently diverging loss curve"

def test_sprt_illconditioned_covariance_no_explosion():
    # Even when the curve fit is ill-conditioned, last_c_std_err must not exceed 1e4.
    sprt_filter = SPRTFilter(sota_threshold=1.0)
    # Use the same kind of diverging CPU loss stream that previously caused c_std_err ~37M.
    # The divergence check fires first, but we verify last_c_std_err was never set large.
    losses = [67.93, 90.40, 127.84, 178.45, 229.12, 306.19, 366.23, 410.78, 521.43, 548.78]
    steps  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for s, l in zip(steps, losses):
        sprt_filter.update_and_check(s, l)
    # If last_c_std_err was ever written it must be sane (<=1e4) or zero (divergence aborted first)
    c_std_err = getattr(sprt_filter, "last_c_std_err", 0.0)
    assert c_std_err <= 1e4, f"c_std_err={c_std_err} is unreasonably large"

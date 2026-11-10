"""Post-hoc host-sensitivity audit using independently pinned C2 source values."""
import pytest
from g2_joint_verification_frontier import max_beta_for_two_x, modeled_speedup


HOST_B_GRANITE=0.7052066730549075
HOST_C_GRANITE=0.3797642644870673
HOST_C_QWEN_B1=0.49043314352149003


def test_beta_threshold_granite_hosts():
    assert max_beta_for_two_x(HOST_B_GRANITE,4,.9,.05)==pytest.approx(0.30046156410023434)
    assert max_beta_for_two_x(HOST_C_GRANITE,4,.9,.05)==pytest.approx(0.5579448089624447)


def test_no_reuse_eight_draft_sensitivity_changes_with_host():
    # Both are optimistic modeled speedups, not runtime measurements.
    assert modeled_speedup(HOST_B_GRANITE,8,1,1,.05)<2
    assert modeled_speedup(HOST_C_GRANITE,8,1,1,.05)>2


def test_qwen_host_c_separate_batch_one_not_pooled_with_old_b4():
    assert max_beta_for_two_x(HOST_C_QWEN_B1,4,.9,.05)==pytest.approx(0.43204155917883125)

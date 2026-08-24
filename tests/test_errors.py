"""What goes wrong, and which side has to fix it.

Every case here is one a backend author meets in the first hour, and each message says
which side the fix belongs to.
"""

import pytest
import torch

from tileops.backend import OpNotAvailableError, UnknownTargetError
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from tileops.ops.reduction.softmax import SoftmaxFwdOp

F16 = torch.float16


def test_an_op_this_target_did_not_register_is_an_error_not_a_fallback():
    """No cross-target fallback: in-tree kernels do not run on the target's devices."""
    op = SoftmaxFwdOp()

    with pytest.raises(OpNotAvailableError, match="registers no kernel builder"):
        op(torch.randn(8, 8, dtype=F16))


def test_naming_a_target_nobody_registered():
    op = RMSNormFwdOp(normalized_shape=(8,), target="nope")

    with pytest.raises(UnknownTargetError, match="known targets"):
        op(torch.randn(4, 8, dtype=F16), torch.randn(8, dtype=F16))


def test_a_failed_call_settles_no_target():
    """One rejected call must not aim the instance for the rest of its life."""
    op = RMSNormFwdOp(normalized_shape=(8,))

    with pytest.raises(ValueError):
        op(torch.randn(4, 9, dtype=F16), torch.randn(8, dtype=F16))
    assert op._settled_target is None

    op(torch.randn(4, 8, dtype=F16), torch.randn(8, dtype=F16))
    assert op._settled_target == "torch_cpu"

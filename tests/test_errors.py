"""What goes wrong, and which side has to fix it.

Every case here is one a backend author meets in the first hour. Two of them are TileOPs
gaps rather than backend bugs, and the messages say which is which.
"""

import pytest
import torch

from tileops.backend import OpNotAvailableError, UnknownTargetError
from tileops.ops.gemm import GemmOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from tileops.ops.reduction.softmax import SoftmaxFwdOp

from conftest import requires_cuda_runtime

F16 = torch.float16


@requires_cuda_runtime
def test_a_registered_builder_is_not_enough_if_the_op_is_not_wired():
    """``pending.build_gemm`` is registered and correct; GemmOp hands over no tensors.

    The fix is in TileOPs: an ``inputs=`` at that op's get-kernel call site.
    """
    op = GemmOp()

    with pytest.raises(OpNotAvailableError, match="not wired to external targets yet"):
        op(torch.randn(8, 8, dtype=F16), torch.randn(8, 8, dtype=F16))


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

"""A tuning request and a config query reach the target's kernels.

The kernel ``build_kernel`` returns may define ``request_tune()`` and ``config``. The op
layer calls the first while the op is in tuned mode, on the kernels already built and on
each one built later, and ``op.kernel_config()`` reads the second.
"""

import warnings

import torch

from tileops.ops.gemm.gemm import GemmFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp

N = 64


def _run(op, rows):
    op(torch.randn(rows, N, dtype=torch.float16), torch.randn(N, dtype=torch.float16))


def _kernels(op, interface):
    return list(op.built_entries(interface).values())


def test_a_request_reaches_the_kernels_built_so_far():
    op = RMSNormFwdOp(normalized_shape=(N,))
    _run(op, 4)

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # a kernel defining request_tune draws no warning
        op.request_tune()
        op.request_tune()                # a later request asks again; the kernel is idempotent

    assert all(kernel.tuned for kernel in _kernels(op, "rms_norm"))


def test_a_kernel_built_after_the_request_is_tuned_before_its_first_call():
    op = RMSNormFwdOp(normalized_shape=(N,))
    op.request_tune()

    _run(op, 4)
    _run(op, 8)

    kernels = _kernels(op, "rms_norm")
    assert len(kernels) == 2 and all(kernel.tuned for kernel in kernels)


def test_kernel_config_reports_the_kernels_config():
    op = RMSNormFwdOp(normalized_shape=(N,), eps=1e-5)
    assert op.kernel_config() is None, "nothing is built before the first call"

    _run(op, 4)

    assert op.kernel_config() == {"normalized_shape": (N,), "eps": 1e-5}


def test_every_op_this_backend_serves_reports_its_config():
    op = GemmFwdOp()
    op(torch.randn(16, 8, dtype=torch.float16), torch.randn(12, 8, dtype=torch.float16))

    assert op.kernel_config() == {"trans_a": False, "trans_b": True}

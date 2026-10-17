"""A tuning request and a config query reach the target's kernels.

The kernel ``build_kernel`` returns may define ``request_tune()`` and ``config``. The op
layer calls the first while the op is in tuned mode, on the kernels already built and on
each one built later before its first call, and ``op.kernel_config()`` reads the second.
"""

import warnings

import pytest
import torch

from tileops.ops.gemm.gemm import GemmFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp

from tileops_cpu.ops.rms_norm import CpuRMSNorm

N = 64
# The warning TileOPs gives when a tuning request reaches a kernel without request_tune().
UNTUNABLE = ".*defines no request_tune"


def _run(op, rows):
    op(torch.randn(rows, N, dtype=torch.float16), torch.randn(N, dtype=torch.float16))


@pytest.fixture
def record(monkeypatch):
    """Count each kernel's ``request_tune()`` calls, and note whether it was tuned at its
    first call."""
    tunes, tuned_at_first_call = {}, {}
    request_tune, call = CpuRMSNorm.request_tune, CpuRMSNorm.__call__

    def counting_request_tune(kernel):
        tunes[id(kernel)] = tunes.get(id(kernel), 0) + 1
        request_tune(kernel)

    def noting_call(kernel, *args, **kwargs):
        tuned_at_first_call.setdefault(id(kernel), kernel.tuned)
        return call(kernel, *args, **kwargs)

    monkeypatch.setattr(CpuRMSNorm, "request_tune", counting_request_tune)
    monkeypatch.setattr(CpuRMSNorm, "__call__", noting_call)
    return tunes, tuned_at_first_call


def test_each_request_reaches_the_kernels_built_so_far(record):
    tunes, _ = record
    op = RMSNormFwdOp(normalized_shape=(N,))
    _run(op, 4)
    (kernel,) = op.built_entries("rms_norm").values()

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=UNTUNABLE)
        op.request_tune()
        op.request_tune()

    assert tunes[id(kernel)] == 2, "every request calls it again; the kernel is idempotent"
    assert kernel.tuned


def test_a_kernel_built_after_the_request_is_tuned_before_its_first_call(record):
    tunes, tuned_at_first_call = record
    op = RMSNormFwdOp(normalized_shape=(N,))

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=UNTUNABLE)
        op.request_tune()
        _run(op, 4)
        _run(op, 8)

    kernels = list(op.built_entries("rms_norm").values())
    assert len(kernels) == 2
    assert all(tunes[id(k)] == 1 and tuned_at_first_call[id(k)] for k in kernels)


def test_kernel_config_reports_the_kernels_config():
    op = RMSNormFwdOp(normalized_shape=(N,), eps=1e-5)
    assert op.kernel_config() is None, "nothing is built before the first call"

    _run(op, 4)

    assert op.kernel_config() == {"normalized_shape": (N,), "eps": 1e-5}


def test_every_op_this_backend_serves_reports_its_config():
    op = GemmFwdOp()
    op(torch.randn(16, 8, dtype=torch.float16), torch.randn(12, 8, dtype=torch.float16))

    assert op.kernel_config() == {"trans_a": False, "trans_b": True}

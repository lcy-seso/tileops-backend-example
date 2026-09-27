"""The target takes the op over, and the op layer still does everything else.

Numerics against the ``ref_api`` the manifest names, and evidence that validation,
contiguity normalization and output handling did not move to the backend.
"""

import pytest
import torch

from tileops.backend import BUILTIN, OpNotAvailableError
from tileops.ops.gemm.gemm import GemmFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp

from tileops_cpu.ops.gemm import CpuGemm
from tileops_cpu.ops.rms_norm import CpuRMSNorm
from tileops_cpu.target import TARGET

DTYPES = [torch.float16, torch.bfloat16]
N = 256


def _ref(x, weight, eps=None):
    return torch.nn.functional.rms_norm(x, (N,), weight, eps=eps)


@pytest.mark.parametrize("dtype", DTYPES)
def test_result_matches_the_manifest_ref_api(dtype):
    x = torch.randn(8, N, dtype=dtype)
    weight = torch.randn(N, dtype=dtype)

    out = RMSNormFwdOp(normalized_shape=(N,))(x, weight)

    assert out.dtype == dtype, "manifest: output dtype is same_as(x)"
    assert out.shape == x.shape, "manifest shape_rules: output.shape == x.shape"
    torch.testing.assert_close(out, _ref(x, weight), rtol=1e-3, atol=1e-3)


def test_the_target_is_what_served_it():
    op = RMSNormFwdOp(normalized_shape=(N,))
    op(torch.randn(4, N, dtype=torch.float16), torch.randn(N, dtype=torch.float16))

    assert op.settled_target == TARGET
    assert isinstance(next(iter(op.built_kernels("rms_norm").values())), CpuRMSNorm)


def test_eps_arrives_as_the_op_holds_it(monkeypatch):
    """The manifest defaults ``eps`` to null, and the builder is handed ``None``."""
    import tileops_cpu.ops.rms_norm as module

    seen = []
    original = module.CpuRMSNorm

    def record(normalized_shape, eps, dtype):
        seen.append(eps)
        return original(normalized_shape, eps, dtype)

    monkeypatch.setattr(module, "CpuRMSNorm", record)
    x, weight = torch.randn(4, N, dtype=torch.float16), torch.randn(N, dtype=torch.float16)

    out = RMSNormFwdOp(normalized_shape=(N,))(x, weight)
    torch.testing.assert_close(out, _ref(x, weight), rtol=1e-3, atol=1e-3)
    RMSNormFwdOp(normalized_shape=(N,), eps=1e-5)(x, weight)

    assert seen == [torch.finfo(torch.float32).eps, 1e-5], "None is the ref_api default"


def test_an_omitted_weight_arrives_as_none():
    """``weight`` is optional in the manifest: its slot stays, holding ``None``."""
    x = torch.randn(4, N, dtype=torch.float16)
    op = RMSNormFwdOp(normalized_shape=(N,))

    out = op(x)

    torch.testing.assert_close(out, _ref(x, None), rtol=1e-3, atol=1e-3)
    (key,) = op.built_kernels("rms_norm")
    assert key[1:] == ((torch.float16, (4, N)), None), "the absent input keys as None"


def test_a_non_contiguous_input_reaches_the_kernel_contiguous():
    """Contiguity is normalized by the op layer, for every target."""
    x = torch.randn(8, 2 * N, dtype=torch.float16)[:, ::2]
    weight = torch.randn(N, dtype=torch.float16)
    assert not x.is_contiguous()

    out = RMSNormFwdOp(normalized_shape=(N,))(x, weight)

    torch.testing.assert_close(out, _ref(x, weight), rtol=1e-3, atol=1e-3)


def test_the_op_layer_still_rejects_what_the_manifest_forbids():
    """Validation stays in front of the seam: the backend is never asked."""
    op = RMSNormFwdOp(normalized_shape=(N,))

    with pytest.raises(ValueError):
        op(torch.randn(8, N + 1, dtype=torch.float16), torch.randn(N, dtype=torch.float16))

    with pytest.raises(ValueError):
        op(torch.randn(8, N, dtype=torch.float16), torch.randn(N, dtype=torch.bfloat16))


def test_the_output_does_not_alias_an_input():
    """No input is declared mutated, so nothing may share storage with one."""
    x = torch.randn(8, N, dtype=torch.float16)
    before = x.clone()

    out = RMSNormFwdOp(normalized_shape=(N,))(x, torch.randn(N, dtype=torch.float16))

    assert out.data_ptr() != x.data_ptr()
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def test_a_second_op_is_served_by_its_own_builder():
    """Two registrations, two builders: the op decides which one it asks for."""
    a = torch.randn(64, 32, dtype=torch.float16)
    b = torch.randn(48, 32, dtype=torch.float16)   # NT by default: b is [N, K]

    op = GemmFwdOp()
    out = op(a, b)

    assert op.settled_target == TARGET
    assert isinstance(next(iter(op.built_kernels("gemm").values())), CpuGemm)
    torch.testing.assert_close(out, (a.float() @ b.float().t()).half(), rtol=1e-2, atol=1e-2)


def test_builtin_escapes_the_target():
    """``target=BUILTIN`` asks for the in-tree kernels, which do not run on CPU, and says so."""
    op = RMSNormFwdOp(normalized_shape=(N,), target=BUILTIN)

    with pytest.raises(OpNotAvailableError, match="in-tree kernels do not run on cpu"):
        op(torch.randn(4, N, dtype=torch.float16), torch.randn(N, dtype=torch.float16))

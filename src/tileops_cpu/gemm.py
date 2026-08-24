"""The second builder this backend registers: a CPU GEMM.

One builder per ``(op, target)``. Which of the op's kernels a call wants — ``GemmFwdOp``
splits into ``gemm_kernel`` and ``gemv_kernel`` — is not passed to a backend, so a builder
that cares reads the shapes off its :class:`~tileops.backend.TensorSpec` arguments and
decides for itself. This one does not care: ``torch.matmul`` covers both.

The op has to be registered under the name the manifest gives it — ``GemmFwdOp``. A
builder registered under any other spelling is never called, and nothing reports it: the op
layer looks up ``(op, target)`` and finds nothing registered for the op it is serving.
"""

import torch

__all__ = ["CpuGemm", "build_gemm"]


class CpuGemm:
    """``d = a @ b``, with either operand optionally transposed first.

    Args:
        trans_a: Transpose *a* before multiplying.
        trans_b: Transpose *b* before multiplying.
        dtype: Storage dtype. The accumulation runs in float32.
    """

    def __init__(self, trans_a: bool, trans_b: bool, dtype: torch.dtype) -> None:
        self.trans_a = bool(trans_a)
        self.trans_b = bool(trans_b)
        self.dtype = dtype

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        lhs = a.float().t() if self.trans_a else a.float()
        rhs = b.float().t() if self.trans_b else b.float()
        return (lhs @ rhs).to(self.dtype)

    def __repr__(self) -> str:
        return f"CpuGemm(trans_a={self.trans_a}, trans_b={self.trans_b}, dtype={self.dtype})"


def build_gemm(a, b, *, trans_a, trans_b):
    """Build the kernel for one GEMM call shape.

    Same rule as every builder: the signature is the op's manifest signature. ``m``, ``n``
    and ``k`` are readable from ``a.shape`` and ``b.shape`` if a backend wants to
    specialize on them; this one does not, so it builds one kernel per dtype and transpose
    pair regardless of shape.

    Args:
        a: Left operand.
        b: Right operand.
        trans_a: Whether *a* is transposed, per the manifest param of that name.
        trans_b: Whether *b* is transposed.

    Returns:
        Something callable with the two tensors described above.
    """
    return CpuGemm(trans_a, trans_b, a.dtype)

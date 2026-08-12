"""A builder that is correct, registered, and cannot be reached yet.

Registering a builder is not by itself enough for a target to serve an op. The op's own
get-kernel call site has to hand over the tensors the builder is described with::

    self.get_or_build_kernel("gemm_kernel", (a, b), key=..., build=...)
                                            ^^^^^^ this

Without it TileOPs cannot compute the memoization key for the external path, so rather than
quietly running an in-tree kernel that cannot launch on the target's hardware, it raises::

    OpNotAvailableError: target 'torch_cpu' serves GemmOp, but its 'gemm_kernel' call site
    does not hand over the tensors a builder is described with; that op is not wired to
    external targets yet

That is a TileOPs-side gap, not a backend bug, and it is mechanical to close. Today
``RMSNormFwdOp`` is the only op wired; the rest arrive op by op. Keeping this builder here
means the day GemmOp is wired, this backend serves it with no change on this side.

Writing builders ahead of the wiring is the normal order of work, so this file is part of
what the example demonstrates.
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

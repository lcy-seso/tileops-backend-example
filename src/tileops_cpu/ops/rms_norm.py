"""RMS norm: the kernel, and the builder that constructs it.

A real backend compiles in the kernel's constructor; this one calls PyTorch. Two rules
shape every kernel class:

* **The constructor takes compile-time parameters only** — values that would be baked into
  generated code. Anything that varies per call stays in ``__call__``. Put a leading
  dimension in a constructor and decode rebuilds a kernel every step.
* **The op layer has already done its half.** Inputs arrive validated, contiguous, in
  ``signature.inputs`` order. Re-checking them here would duplicate a contract that lives in
  the manifest. Parameters arrive as the op instance holds them, so a manifest default of
  null arrives as ``None`` and the builder decides what it means.
"""

import math

import torch

from tileops.backend import TensorSpec

__all__ = ["CpuRMSNorm", "build_rms_norm"]


class CpuRMSNorm:
    """RMS norm over the trailing ``normalized_shape`` axes.

    Mirrors ``torch.nn.functional.rms_norm``, the ``ref_api`` the manifest names for
    ``RMSNormFwdOp``::

        y = x * rsqrt(mean(x ** 2, trailing_axes) + eps) * weight

    An absent ``weight`` scales by one.

    Args:
        normalized_shape: Trailing-axis shape the reduction runs over.
        eps: Denominator epsilon, a number.
        dtype: The input dtype this instance was built for. Storage dtype only — the
            reduction runs in float32 and casts back at the boundary, because the sum of
            squares of a 16-bit row overflows well before the row is long enough to matter.

    Example:
        >>> kernel = CpuRMSNorm((4096,), 1e-6, torch.float16)
        >>> y = kernel(torch.randn(8, 4096, dtype=torch.float16),
        ...            torch.randn(4096, dtype=torch.float16))
    """

    def __init__(self, normalized_shape, eps: float, dtype: torch.dtype) -> None:
        self.normalized_shape = tuple(int(d) for d in normalized_shape)
        self.eps = float(eps)
        self.dtype = dtype
        # Where a real backend would emit and compile code. Note what is *not* available
        # here: how many rows this will be called with. That is deliberate — see the
        # module docstring.
        self._axes = tuple(range(-len(self.normalized_shape), 0))
        self._elems = math.prod(self.normalized_shape)

    def __call__(self, x: torch.Tensor, weight: "torch.Tensor | None") -> torch.Tensor:
        """Run one call.

        Args:
            x: Contiguous, dtype and trailing shape as built for.
            weight: Contiguous, shape ``normalized_shape``, or ``None`` when the call
                omitted it.

        Returns:
            A new tensor shaped like *x*, dtype ``same_as(x)`` per the manifest. Fresh
            storage: the manifest declares no input as mutated, so nothing may alias.
        """
        acc = x.float()
        y = acc * torch.rsqrt(acc.pow(2).mean(dim=self._axes, keepdim=True) + self.eps)
        if weight is not None:
            y = y * weight.float()
        return y.to(self.dtype)

    def __repr__(self) -> str:
        return (f"CpuRMSNorm(normalized_shape={self.normalized_shape}, "
                f"eps={self.eps}, dtype={self.dtype})")


def build_rms_norm(x: TensorSpec, weight: "TensorSpec | None", *, normalized_shape, eps):
    """Build the kernel that serves one RMS norm call shape.

    The signature is the op's manifest signature and nothing else: ``signature.inputs`` in
    declaration order as positional :class:`~tileops.backend.TensorSpec`, then
    ``signature.params`` by keyword. The manifest declares ``weight`` optional, so a call
    that omits it arrives here as ``None``. ``eps`` defaults to null in the manifest and
    arrives as the op instance holds it: ``None`` unless the caller gave a number.

    A ``TensorSpec`` carries device, dtype and shape — no data, and no reference to the
    tensor. So a builder cannot key on values it would then be memoized against, and cannot
    keep a tensor alive for the process lifetime by holding the kernel.

    Args:
        x: The tensor to normalize.
        weight: The affine scale, or ``None``.
        normalized_shape: Trailing axes the reduction runs over.
        eps: Denominator epsilon, or ``None`` for the ``ref_api`` default.

    Returns:
        Something callable with the two tensors described above.
    """
    # ``eps=None`` means what it means in torch.nn.functional.rms_norm: the machine epsilon
    # of the float32 accumulation.
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    return CpuRMSNorm(normalized_shape, eps, x.dtype)

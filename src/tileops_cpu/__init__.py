"""Everything this backend tells TileOPs. Copy this file's shape; replace its kernels.

A backend joins TileOPs with two calls at module top level, and one entry point in
``pyproject.toml`` that makes this module get imported. Nothing else is registered: not the
kernel classes, not which dtypes or shapes are supported, not a priority. TileOPs picks a
target; the target picks a kernel.

Importing this must not compile anything — TileOPs runs it while the first Op is being
constructed. Compilation belongs inside a ``build_kernel``, which is called later, once per
distinct input signature.
"""

from tileops.backend import TensorSpec, register_detector, register_kernel_builder

from .gemm import build_gemm
from .kernels import CpuRMSNorm

#: The name this distribution gives its set of kernels. Chosen, not derived: a target name
#: is not a ``torch.device.type``. One torch type can cover several vendors' kernels, and
#: some hardware arrives as ``privateuseone``, which names no vendor at all. The detector
#: below is where the device question gets answered.
TARGET = "torch_cpu"

__all__ = ["TARGET", "build_gemm", "build_rms_norm"]


def _detect(device) -> bool:
    """Answer whether *device* is the kind these kernels are written for.

    Devices only. Whether a particular dtype, shape or parameter combination is supported
    is ``build_kernel``'s answer — it is the one that sees them. Return ``False`` rather
    than raising: a detector that raises is reported as this distribution's bug, because
    stepping over it would hand the device to somebody else.
    """
    return device.type == "cpu"


def build_rms_norm(x: TensorSpec, weight: TensorSpec, *, normalized_shape, eps):
    """Build the kernel that serves one RMS norm call shape.

    The signature is the op's manifest signature and nothing else: ``signature.inputs`` in
    declaration order as positional :class:`~tileops.backend.TensorSpec`, then
    ``signature.params`` by keyword. ``eps`` defaults to null in the manifest, and arrives
    here as the number the op settled on.

    A ``TensorSpec`` carries device, dtype and shape — no data, and no reference to the
    tensor. So a builder cannot key on values it would then be memoized against, and cannot
    keep a tensor alive for the process lifetime by holding the kernel.

    Args:
        x: The tensor to normalize.
        weight: The affine scale.
        normalized_shape: Trailing axes the reduction runs over.
        eps: Denominator epsilon, already resolved to a float.

    Returns:
        Something callable with the two tensors described above.
    """
    return CpuRMSNorm(normalized_shape, eps, x.dtype)


register_detector(target=TARGET, detect=_detect)

# One registration per op taken over, under the name the manifest gives the op.
register_kernel_builder(op="RMSNormFwdOp", target=TARGET, build_kernel=build_rms_norm)
register_kernel_builder(op="GemmFwdOp", target=TARGET, build_kernel=build_gemm)

"""Everything this backend tells TileOPs. Copy this file's shape; replace its kernels.

A backend joins TileOPs with two kinds of call at module top level, plus one entry point in
``pyproject.toml`` that makes this module get imported: one detector for the devices it
claims, and one builder per op it takes over. Nothing else is registered — not the kernel
classes, not which dtypes or shapes are supported, not a priority. TileOPs picks a target;
the target picks a kernel.

Importing this must not compile anything — TileOPs runs it while the first Op is being
constructed. Compilation belongs inside a ``build_kernel``, which is called later, once per
device and input signature.
"""

from tileops.backend import register_detector, register_kernel_builder

from .ops import BUILDERS
from .target import TARGET, detect

__all__ = ["BUILDERS", "TARGET", "detect"]

register_detector(target=TARGET, detect=detect)

for _op, _build_kernel in BUILDERS.items():
    register_kernel_builder(op=_op, target=TARGET, build_kernel=_build_kernel)

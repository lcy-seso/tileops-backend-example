"""The ops this backend serves, one module each.

A module holds one op's kernel and the builder that constructs it: the two change together,
and the builder's signature is the op's manifest signature. :data:`BUILDERS` is the whole
list of what this target claims — the keys are manifest keys, spelled exactly as the
manifest spells them, because a registration under any other spelling is never called.

Adding an op: write ``ops/<name>.py``, then add one line here.
"""

from .gemm import build_gemm
from .rms_norm import build_rms_norm

__all__ = ["BUILDERS", "build_gemm", "build_rms_norm"]

#: Manifest key → the builder this backend registers for it.
BUILDERS = {
    "RMSNormFwdOp": build_rms_norm,
    "GemmFwdOp": build_gemm,
}

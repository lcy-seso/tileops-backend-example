"""Shared markers.

Almost everything here runs on a machine with no GPU — that is the point of a torch-only
backend. Two cases cannot, and the reason is worth knowing: they reach code that touches
CUDA before anything target-specific runs.
"""

import pytest
import torch

#: A CUDA driver has to be present, though no test needs a kernel to launch.
#:
#: * ``GemmFwdOp.forward`` builds a ``GemmCall``. It states no ``arch``, so
#:   ``CallSpec.__post_init__`` reads ``get_sm_version()`` → ``torch.cuda.current_device()``
#:   — before the get-kernel call site, and therefore before the CPU target that serves the
#:   call has any say. A GEMM served by this backend still needs a driver present.
#: * ``target=BUILTIN`` constructs the in-tree kernel, which initializes CUDA.
#:
#: The first is a platform assumption to report upstream when it blocks real hardware; the
#: second is what ``BUILTIN`` means.
requires_cuda_runtime = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TileOPs reaches torch.cuda on this path before any target-specific code runs",
)

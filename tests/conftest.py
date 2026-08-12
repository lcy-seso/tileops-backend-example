"""Shared markers.

Almost everything here runs on a machine with no GPU — that is the point of a torch-only
backend. Two cases cannot, and the reason is worth knowing: TileOPs still touches CUDA on
paths that have nothing to do with the target that was selected.
"""

import pytest
import torch

#: A CUDA driver has to be present, though no test needs a kernel to launch.
#:
#: * ``GemmOp.forward`` builds a ``CallSpec``, whose ``__post_init__`` calls
#:   ``get_sm_version()`` → ``torch.cuda.current_device()``, before it ever reaches the
#:   get-kernel call site. So the "op is not wired" error is only observable where a driver
#:   exists, even though the target that serves the op is a CPU one.
#: * ``target=BUILTIN`` constructs the in-tree kernel, which initializes CUDA.
#:
#: Both are TileOPs-side platform assumptions on the way out; a backend author hits them
#: today and should recognize them for what they are.
requires_cuda_runtime = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TileOPs reaches torch.cuda on this path before any target-specific code runs",
)

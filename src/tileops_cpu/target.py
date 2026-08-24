"""What this distribution calls its kernels, and which devices they are for."""

import torch

__all__ = ["TARGET", "detect"]

#: The name this distribution gives its set of kernels. Chosen, not derived: a target name
#: is not a ``torch.device.type``. One torch type can cover several vendors' kernels, and
#: some hardware arrives as ``privateuseone``, which names no vendor at all.
TARGET = "torch_cpu"


def detect(device: torch.device) -> bool:
    """Answer whether *device* is the kind these kernels are written for.

    Devices only. Whether a particular dtype, shape or parameter combination is supported
    is ``build_kernel``'s answer — it is the one that sees them. Return ``False`` rather
    than raising: a detector that raises is reported as this distribution's bug, because
    stepping over it would hand the device to somebody else.
    """
    return device.type == "cpu"

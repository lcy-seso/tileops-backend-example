"""Installing the wheel is the whole integration step.

Nothing here imports ``tileops_cpu`` to make it work: TileOPs finds it through the entry
point while the first Op is being constructed.
"""

from importlib.metadata import entry_points

import torch

from tileops.backend import default_target, load_failures, registered_targets

TARGET = "torch_cpu"


def _construct_an_op():
    """Discovery is triggered by op construction, not by importing tileops."""
    from tileops.ops.norm.rms_norm import RMSNormFwdOp

    return RMSNormFwdOp(normalized_shape=(64,))


def test_the_entry_point_is_declared():
    declared = {ep.name: ep.value for ep in entry_points(group="tileops.backends")}
    assert declared[TARGET] == "tileops_cpu"


def test_the_target_is_registered_without_anyone_importing_it():
    _construct_an_op()

    assert TARGET in registered_targets()
    assert registered_targets("RMSNormFwdOp") == [TARGET]


def test_this_backend_loaded_cleanly():
    _construct_an_op()

    assert not [f for f in load_failures() if TARGET in f or "tileops_cpu" in f]


def test_nothing_is_the_default_until_someone_says_so():
    """Detection, not a default. A backend does not get to declare itself the default."""
    _construct_an_op()

    assert default_target() is None


def test_detection_is_what_selects_it():
    op = _construct_an_op()
    op(torch.randn(4, 64, dtype=torch.float16), torch.randn(64, dtype=torch.float16))

    assert op._settled_target == TARGET, "selected from the device, with no target= given"

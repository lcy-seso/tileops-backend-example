"""How often the builder is called, and what decides it.

The contract, which TileOPs fixes and a backend cannot change: the external path remembers
a kernel under the device plus the *input signature* — ``(dtype, shape)`` per input, in
``signature.inputs`` order. Two calls agreeing on both get the same kernel object.
"""

import torch

from tileops.ops.norm.rms_norm import RMSNormFwdOp

N = 64


def _run(op, rows, dtype=torch.float16):
    return op(torch.randn(rows, N, dtype=dtype), torch.randn(N, dtype=dtype))


def test_the_same_input_signature_reuses_the_same_kernel():
    op = RMSNormFwdOp(normalized_shape=(N,))

    _run(op, 4)
    _run(op, 4)

    built = op.built_kernels("rms_norm")
    assert len(built) == 1, "one build for one signature"


def test_the_key_is_the_device_then_dtype_and_shape_per_input():
    op = RMSNormFwdOp(normalized_shape=(N,))
    _run(op, 4)

    (key,) = op.built_kernels("rms_norm")
    device, *inputs = key
    assert device.type == "cpu", "the device a kernel was built for is part of the key"
    assert inputs == [(torch.float16, (4, N)), (torch.float16, (N,))], "x then weight"


def test_a_new_shape_asks_the_builder_again():
    op = RMSNormFwdOp(normalized_shape=(N,))

    _run(op, 4)
    _run(op, 8)

    built = op.built_kernels("rms_norm")
    assert len(built) == 2
    assert len({id(k) for k in built.values()}) == 2, "distinct kernels, not one reused"


def test_a_new_dtype_asks_the_builder_again():
    op = RMSNormFwdOp(normalized_shape=(N,))

    _run(op, 4, torch.float16)
    _run(op, 4, torch.bfloat16)

    assert len(op.built_kernels("rms_norm")) == 2


def test_each_op_instance_memoizes_on_its_own():
    first, second = RMSNormFwdOp(normalized_shape=(N,)), RMSNormFwdOp(normalized_shape=(N,))

    _run(first, 4)
    _run(second, 4)

    assert len(first.built_kernels("rms_norm")) == 1
    assert len(second.built_kernels("rms_norm")) == 1

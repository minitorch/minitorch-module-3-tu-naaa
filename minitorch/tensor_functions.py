"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4.
        a, b = ctx.saved_values
        return (grad_output.f.mul_zip(b, grad_output), grad_output.f.mul_zip(a, grad_output))
        # raise NotImplementedError("Need to implement for Task 2.4")


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(t1)
        return t1.f.sigmoid_map(t1)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        (t1,) = ctx.saved_values
        tmp = t1.f.sigmoid_map(t1)
        ones = tensor([1.0])  # 创建一个tensor: [1.0]
        return grad_output.f.mul_zip(tmp.f.mul_zip(ones - tmp, tmp), grad_output)
        # raise NotImplementedError("Need to implement for Task 2.4")


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)
        # raise NotImplementedError("Need to implement for Task 2.4")


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)
        # raise NotImplementedError("Need to implement for Task 2.4")


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(t1)
        return t1.f.exp_map(t1)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4.
        (t1,) = ctx.saved_values
        return grad_output.f.mul_zip(t1.f.exp_map(t1), grad_output)
        # raise NotImplementedError("Need to implement for Task 2.4")


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        return a.f.lt_zip(a, b)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4.
        return 0.0, 0.0
        # raise NotImplementedError("Need to implement for Task 2.4")


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        return a.f.eq_zip(a, b)
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4.
        return 0.0, 0.0
        # raise NotImplementedError("Need to implement for Task 2.4")


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        return a.f.is_close_zip(a, b)
        # raise NotImplementedError("Need to implement for Task 2.3")


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3.
        ctx.save_for_backward(order)
        order_list = [int(order[i]) for i in range(order.size)]  # tensor -> list
        return a._new(a._tensor.permute(*order_list))  # 创建新tensor
        # raise NotImplementedError("Need to implement for Task 2.3")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:  # 把梯度从新维度顺序“还原”回去
        # TODO: Implement for Task 2.4.
        (order,) = ctx.saved_values
        order_list = [int(order[i]) for i in range(order.size)]  # tensor -> list
        new_order_list = [0] * len(order_list)  # 还原permute前的order
        for i, order_i in enumerate(order_list):  # 原order: (p1, ..., pn) - (i, pi) - new_order中第pi个位子对应i
            new_order_list[order_i] = i  # e.g. 原0新2，原1新0，原2新1->新0原1，新1原2，新2原0 - (2,0,1)->(1,2,0)
        return grad_output.permute(*new_order_list), 0.0  # 对a的梯度；对order的梯度（order是常量，梯度为0）
        # raise NotImplementedError("Need to implement for Task 2.4")


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:  # f(M, N) = M matmul N
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:  # df/dM = grad_output matmul N^T, df/dN = M^T matmul grad_output
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    return minitorch.Tensor.make(
        [0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )

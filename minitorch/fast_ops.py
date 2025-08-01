from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers   用NumPy数组处理维度索引，避免慢的Python列表操作
    * When `out` and `in` are stride-aligned, avoid indexing   当输入输出的strides匹配时，直接逐个线性扫描，避免昂贵的index计算

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # 优化3
        if np.array_equal(out_shape, in_shape) and np.array_equal(out_strides, in_strides):  # 直接使用==会返回一个布尔数组而不是布尔值 ×
            for i in prange(np.prod(out_shape)):
                out[i] = fn(in_storage[i])
            return

        for ordinal in prange(np.prod(out_shape)):  # 优化1
            # 注意：要为每个线程创建自己的局部变量out_index、in_index
            out_index = np.zeros(len(out_shape), dtype=np.int32)  # 优化2
            in_index = np.zeros(len(in_shape), dtype=np.int32)  # 优化2

            to_index(ordinal, out_shape, out_index)  # 原本的to_index会改变循环变量ordinal，得改写to_index
            broadcast_index(out_index, out_shape, in_shape, in_index)  

            in_pos = index_to_position(in_index, in_strides)  
            out_pos = index_to_position(out_index, out_strides)  

            out[out_pos] = fn(in_storage[in_pos])

        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        if np.array_equal(out_shape, a_shape) and np.array_equal(out_strides, a_strides) and np.array_equal(out_shape, b_shape) and np.array_equal(out_strides, b_strides):  # 优化3
            for i in prange(np.prod(out_shape)):
                out[i] = fn(a_storage[i], b_storage[i])
            return
        
        for ordinal in prange(np.prod(out_shape)):  # 优化1
            out_index = np.zeros(len(out_shape), dtype=np.int32)  # 优化2
            a_index = np.zeros(len(a_shape), dtype=np.int32)  # 优化2
            b_index = np.zeros(len(b_shape), dtype=np.int32)  # 优化2

            to_index(ordinal, out_shape, out_index)  
            broadcast_index(out_index, out_shape, a_shape, a_index)  
            broadcast_index(out_index, out_shape, b_shape, b_index)

            a_pos = index_to_position(a_index, a_strides) 
            b_pos = index_to_position(b_index, b_strides)  
            out_pos = index_to_position(out_index, out_strides)  

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos]) 

        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        for ordinal in prange(np.prod(out_shape)):  # 优化1
            out_index = np.zeros(len(out_shape), dtype=np.int32)  # 优化2、3
            a_index = np.zeros(len(a_shape), dtype=np.int32)  # 优化2、3

            to_index(ordinal, out_shape, out_index)  
            for i in range(len(out_shape)):
                if i != reduce_dim:
                    a_index[i] = out_index[i]  
                else:
                    a_index[i] = 0  

            a_index[reduce_dim] = 0
            a_pos = index_to_position(a_index, a_strides)
            now = a_storage[a_pos]
            for i in range(1, a_shape[reduce_dim]):  
                a_index[reduce_dim] = i 
                a_pos = index_to_position(a_index, a_strides)  # 用@njit编译且逻辑简单，Numba会接受并inline，不违反优化3
                now = fn(now, a_storage[a_pos])  # 传参进来的且inline，不违反优化3
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = now

        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(  # 被高层MatMul(in tensor_functions.py)调用
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.  内层循环：不要写入输出张量out(昂贵 & 多线程竞争) / 每次只做一次乘法


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    batch_size = a_shape[0]
    m = a_shape[1]  
    n = a_shape[-1]  # a: batchsize × ... × m × n (前面的维度可以任意broadcast)
    k = b_shape[-1]  # b: batchsize × ... × n × k (前面的维度可以任意broadcast)

    for b in prange(batch_size):  # 优化1
        for i in range(m):  # 遍历a的行
            for j in range(k):  # 遍历b的列
                tmp = 0.0  # 优化3
                for l in range(n):  # out[b][i][j] = sigma(a[b][i][l] * b[b][l][j])
                    # 优化2
                    a_pos = b * a_batch_stride + i * a_strides[1] + l * a_strides[2]  # 展开index_to_position: sigma(index[i] * strides[i])
                    b_pos = b * b_batch_stride + l * b_strides[1] + j * b_strides[2]
                    tmp += a_storage[a_pos] * b_storage[b_pos]  # 优化3

                out_pos = b * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                out[out_pos] = tmp

    # raise NotImplementedError("Need to implement for Task 3.2")


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None

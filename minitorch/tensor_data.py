from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:  # 根据张量的多维下标index和对应步幅strides，计算在底层一维存储中的位置
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """

    # TODO: Implement for Task 2.1.
    # stride[0]: 沿第一个维度(行)走一步，要跳过几个元素
    # stride[1]: 沿第二个维度(列)走一步，要跳过几个元素
    # e.g. 形状为(3, 4)的二维张量，按行存储 -> stride是[4, 1]
    # index: 要访问的元素的下标
    # e.g. index=(2, 3) -> a[2][3]
    # return index[0] * strides[0] + index[1] * strides[1]  #only 二维
    sum = 0
    for i in range(len(index)):
        sum += (index[i] * strides[i])
    return sum
    # raise NotImplementedError("Need to implement for Task 2.1")


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # TODO: Implement for Task 2.1.
    # shape = (x, y, z) -> x块，每个块是y行z列的矩阵
    # index = (i, j, k)
    # k = ordinal % z, ordinal //= z; j = ordinal % y, ordinal //= y; i = ordinal
    # tmp = ordinal  # 不修改ordinal
    ordinal = ordinal + 0
    for i in range(len(shape)-1, -1, -1):  # 从len(shape)-1到0，倒序遍历
        out_index[i] = ordinal % shape[i]
        ordinal //= shape[i]
    # raise NotImplementedError("Need to implement for Task 2.1")


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor  广播后的张量索引
        big_shape : tensor shape of bigger tensor  广播后的张量形状
        shape : tensor shape of smaller tensor  原始的张量形状
        out_index : multidimensional index of smaller tensor  big_index转换过来的小张量的索引

    Returns:
        None
    """
    # TODO: Implement for Task 2.2.
    # (big_shape) = (1, ..., 1, shape)
        # 新add的维度：没有对应的out_index
        # 某个维度上被广播了size：out_index[i] = 0
        # 其他：out_index[i] = big_index[i+offset]
    offset = len(big_shape) - len(shape)
    for i in range(len(shape)):
        if shape[i] < big_shape[i + offset]:
            out_index[i] = 0 
        else:
            out_index[i] = big_index[i + offset]
    # raise NotImplementedError("Need to implement for Task 2.2")


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # TODO: Implement for Task 2.2.
    if len(shape1) < len(shape2):
        shape1 = (1,) * (len(shape2) - len(shape1)) + shape1
    elif len(shape2) < len(shape1):
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2
    out_shape = [0] * len(shape1)
    for i in range(len(shape1)):
        if shape1[i] != shape2[i] and shape1[i] != 1 and shape2[i] != 1:
            raise IndexingError
            # raise IndexingError(
            #     f"Cannot broadcast {shape1} and {shape2} at dimension {i}."
            # )
        else:
            out_shape[i] = max(shape1[i], shape2[i])
    return tuple(out_shape)
    # raise NotImplementedError("Need to implement for Task 2.2")


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:  # 只是重排，不改变物理地址!!!相当于shape[i]和stride[i]绑定，一起permute
        """
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # TODO: Implement for Task 2.1.
        # e.g. order=(2, 0, 1): 原来的第2维变为第0维，第0维变为第1维，第1维变为第2维
        tmp_shape = []
        tmp_strides = []
        for i, order_i in enumerate(order):
            tmp_shape.append(self.shape[order_i])
            tmp_strides.append(self.strides[order_i])
        return TensorData(self._storage, tuple(tmp_shape), tuple(tmp_strides))
        # raise NotImplementedError("Need to implement for Task 2.1")

        # storage变化时候的stride重算：
            # shape = (2, 3, 4) -> stride = (3×4, 4, 1)
            # (s0, s1, ..., sn-1) -> (sn-1×sn-2×...×s1, ..., sn-1×sn-2, sn-1, 1)
            # tmp_strides = np.zeros_like(self.strides, dtype=np.int32)
            # now = 1
            # for i in range(len(tmp_shape)-1, -1, -1):
            #     tmp_strides[i] = now
            #     now *= tmp_shape[i]

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s

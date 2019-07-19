import operator as _op
from functools import partialmethod
from itertools import count
from numbers import Number
from typing import Callable, Union, List, Optional

import numpy as np

_LAZY_LIKE = Union['LazyVariable', Number, np.ndarray, Callable]


class LazyVariable:
    """A constant, calculation on me is not evaluated until executed."""
    _counter = count(0)

    # See: https://stackoverflow.com/a/35892101/3383878
    __array_priority__ = 2

    def __init__(self, value: _LAZY_LIKE, length=None):
        self._id = next(LazyVariable._counter)

        if isinstance(value, LazyVariable):
            assert length is None, f"We should not use length here"  # TODO(hx)
            # maintain the same len
            self._length = value._length
            self._func = value
        elif callable(value):
            # an intermediate calculation result
            self._length = length
            self._func = value
        else:
            assert length is None, f"We should not use length here"  # TODO(hx)
            value = np.asarray(value)
            if value.ndim > 1:
                # TODO(hx) support this?
                raise NotImplementedError(value.shape)
            self._func = lambda: value
            if value.ndim > 0:
                self._length = len(value)
            else:
                self._length = None

    @property
    def length(self) -> Optional[int]:
        return self._length

    def assign(self, other):
        # check length compatibility
        # assign changes my value, hence return a raw function, and do not
        # chain any function further.
        la = self._length
        try:
            lb = len(other)
            if la != lb:
                msg = f"self {la}, other {lb}"
                raise ValueError(f"Incompatible len: {msg}")
        except TypeError:
            # lb has no length
            if la is not None:
                raise ValueError(f"Incompatible len: self {la}, other None.")

        def wrapper():
            if callable(other):
                self._func = other
            else:
                self._func = lambda: np.asarray(other)
            return self

        return wrapper

    @property
    def func(self) -> Callable:
        return self._func

    def __call__(self, *args, **kwargs):
        return self._func()

    execute = __call__  # just another name

    def _get_value_binary_op(self, other, op, reverse=False):
        # check length compatibility
        # todo(hx) we did a simple broad cast, could be dangerous
        this = self
        la = this._length
        try:
            lb = len(other)
            if la != lb:
                if la == 1 or la is None:
                    this = StackedLazyVariable([this] * lb)
                elif lb == 1 or lb is None:
                    other = StackedLazyVariable([other] * la)
                else:
                    msg = f"self {la}, other {lb}"
                    raise ValueError(f"Incompatible len: {msg}")
        except TypeError:
            # lb has no length
            if la is not None:
                other = StackedLazyVariable([other] * la)

        def wrapper():
            a = this.execute()
            if isinstance(other, LazyVariable):
                b = other()
            else:
                b = np.asarray(other)
            if reverse:
                return op(b, a)
            else:
                return op(a, b)

        return LazyVariable(wrapper, length=this.length)

    __add__ = partialmethod(_get_value_binary_op, op=_op.add, reverse=False)
    __radd__ = partialmethod(_get_value_binary_op, op=_op.add, reverse=True)
    __sub__ = partialmethod(_get_value_binary_op, op=_op.sub, reverse=False)
    __rsub__ = partialmethod(_get_value_binary_op, op=_op.sub, reverse=True)
    __mul__ = partialmethod(_get_value_binary_op, op=_op.mul, reverse=False)
    __rmul__ = partialmethod(_get_value_binary_op, op=_op.mul, reverse=True)
    __truediv__ = partialmethod(_get_value_binary_op, op=_op.truediv,
                                reverse=False)
    __rtruediv__ = partialmethod(_get_value_binary_op, op=_op.truediv,
                                 reverse=True)

    def __pow__(self, power, modulo=None):
        assert modulo is None, "Not supported."

        def wrapper():
            a = self.execute()
            return a ** power

        return LazyVariable(wrapper, length=self._length)

    def __neg__(self):
        def neg():
            return - self.execute()

        return LazyVariable(neg, self._length)

    def __getitem__(self, item):
        assert isinstance(item, int), 'Not supported yet.'  # todo: hx

        if item >= self._length:
            raise IndexError(item)

        def _getitem():
            # todo(hx) this could be expensive
            a = self.execute()
            return a[item]

        return LazyVariable(_getitem, length=1)

    def __len__(self):
        if self._length is None:
            raise TypeError(f"len() of unsized object")
        else:
            return self._length

    @property
    def id(self) -> int:
        return self._id

    def __str__(self):
        return f"LazyVariable_{self._id}"

    def __repr__(self):
        return "LazyVariable_{0:d}({1:s})".format(self._id, repr(self._func))

    def __hash__(self):
        """The hash is unique to every instance.
        This makes it possible to uniquely bind a variable to another
        value/object using the id."""
        return self._id

    # Copy is ambiguous for a placeholder, hence it is disabled a.t.m. TODO(hx)
    def __copy__(self):
        raise NotImplementedError(type(str(self)))

    # Copy is ambiguous for a placeholder, hence it is disabled a.t.m. TODO(hx)
    def __deepcopy__(self):
        raise NotImplementedError(type(str(self)))


class StackedLazyVariable(LazyVariable):
    def __init__(
            self,
            vars: List[LazyVariable]
    ):
        self._value_list = [
            v if isinstance(v, LazyVariable) else LazyVariable(v)
            for v in vars
        ]
        super(StackedLazyVariable, self).__init__(
            value=self.__call__, length=len(vars)
        )

    def __call__(self, *args, **kwargs):
        return np.asarray([v() for v in self._value_list])

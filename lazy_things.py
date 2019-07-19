from numbers import Number
from typing import Callable, Union

__all__ = ['Node']

_LAZY_NUMBER_LIKE = Union['Node', Number, Callable]


class Node:
    def __init__(self, value, parent=None):
        if callable(value):
            self._func = value
            self._root = False
            self._parent = parent
        else:
            self._func = lambda: value
            self._root = True
            if parent is not None:
                raise ValueError(f'Parent should be None for root node.')
            self._parent = value

    @property
    def root(self) -> bool:
        return self._root

    @property
    def parent(self):
        return self._parent

    @property
    def func_name(self):
        return self._func.__name__

    @property
    def func(self):
        return self._func

    def __radd__(self, other):
        def wrapped_radd():
            return other + self.execute()

        return Node(wrapped_radd, parent=self)

    def __add__(self, other):
        def wrapped_add():
            return self.execute() + other

        return Node(wrapped_add, parent=self)

    def execute(self):
        return self._func()

#
# class Node:
#     """A constant, calculation on me is not evaluated until executed."""
#     _counter = count(0)
#
#     def __init__(self, value: _LAZY_NUMBER_LIKE):
#         self._id = next(Node._counter)
#
#         if isinstance(value, Node):
#             assert length is None, f"We should not use length here"  # TODO(hx)
#             # maintain the same len
#             self._length = value._length
#             self._func = value
#         elif callable(value):
#             # an intermediate calculation result
#             self._length = length
#             self._func = value
#         else:
#             assert length is None, f"We should not use length here"  # TODO(hx)
#             value = np.asarray(value)
#             if value.ndim > 1:
#                 # TODO(hx) support this?
#                 raise NotImplementedError(value.shape)
#             self._func = lambda: value
#             if value.ndim > 0:
#                 self._length = len(value)
#             else:
#                 self._length = None
#
#     @property
#     def length(self) -> Optional[int]:
#         return self._length
#
#     def assign(self, other):
#         # check length compatibility
#         # assign changes my value, hence return a raw function, and do not
#         # chain any function further.
#         la = self._length
#         try:
#             lb = len(other)
#             if la != lb:
#                 msg = f"self {la}, other {lb}"
#                 raise ValueError(f"Incompatible len: {msg}")
#         except TypeError:
#             # lb has no length
#             if la is not None:
#                 raise ValueError(f"Incompatible len: self {la}, other None.")
#
#         def wrapper():
#             if callable(other):
#                 self._func = other
#             else:
#                 self._func = lambda: np.asarray(other)
#             return self
#
#         return wrapper
#
#     @property
#     def func(self) -> Callable:
#         return self._func
#
#     def __call__(self, *args, **kwargs):
#         return self._func()
#
#     execute = __call__  # just another name
#
#     def _get_value_binary_op(self, other, op, reverse=False):
#         # check length compatibility
#         # todo(hx) we did a simple broad cast, could be dangerous
#         this = self
#         la = this._length
#         try:
#             lb = len(other)
#             if la != lb:
#                 if la == 1 or la is None:
#                     this = StackedLazyVariable([this] * lb)
#                 elif lb == 1 or lb is None:
#                     other = StackedLazyVariable([other] * la)
#                 else:
#                     msg = f"self {la}, other {lb}"
#                     raise ValueError(f"Incompatible len: {msg}")
#         except TypeError:
#             # lb has no length
#             if la is not None:
#                 other = StackedLazyVariable([other] * la)
#
#         def wrapper():
#             a = this.execute()
#             if isinstance(other, Node):
#                 b = other()
#             else:
#                 b = np.asarray(other)
#             if reverse:
#                 return op(b, a)
#             else:
#                 return op(a, b)
#
#         return Node(wrapper, length=this.length)
#
#     __add__ = partialmethod(_get_value_binary_op, op=_op.add, reverse=False)
#     __radd__ = partialmethod(_get_value_binary_op, op=_op.add, reverse=True)
#     __sub__ = partialmethod(_get_value_binary_op, op=_op.sub, reverse=False)
#     __rsub__ = partialmethod(_get_value_binary_op, op=_op.sub, reverse=True)
#     __mul__ = partialmethod(_get_value_binary_op, op=_op.mul, reverse=False)
#     __rmul__ = partialmethod(_get_value_binary_op, op=_op.mul, reverse=True)
#     __truediv__ = partialmethod(_get_value_binary_op, op=_op.truediv,
#                                 reverse=False)
#     __rtruediv__ = partialmethod(_get_value_binary_op, op=_op.truediv,
#                                  reverse=True)
#
#     def __pow__(self, power, modulo=None):
#         assert modulo is None, "Not supported."
#
#         def wrapper():
#             a = self.execute()
#             return a ** power
#
#         return Node(wrapper, length=self._length)
#
#     def __neg__(self):
#         def neg():
#             return - self.execute()
#
#         return Node(neg, self._length)
#
#     def __getitem__(self, item):
#         assert isinstance(item, int), 'Not supported yet.'  # todo: hx
#
#         if item >= self._length:
#             raise IndexError(item)
#
#         def _getitem():
#             # todo(hx) this could be expensive
#             a = self.execute()
#             return a[item]
#
#         return Node(_getitem, length=1)
#
#     def __len__(self):
#         if self._length is None:
#             raise TypeError(f"len() of unsized object")
#         else:
#             return self._length
#
#     @property
#     def id(self) -> int:
#         return self._id
#
#     def __str__(self):
#         return f"LazyVariable_{self._id}"
#
#     def __repr__(self):
#         return "LazyVariable_{0:d}({1:s})".format(self._id, repr(self._func))
#
#     def __hash__(self):
#         """The hash is unique to every instance.
#         This makes it possible to uniquely bind a variable to another
#         value/object using the id."""
#         return self._id
#
#     # Copy is ambiguous for a placeholder, hence it is disabled a.t.m. TODO(hx)
#     def __copy__(self):
#         raise NotImplementedError(type(str(self)))
#
#     # Copy is ambiguous for a placeholder, hence it is disabled a.t.m. TODO(hx)
#     def __deepcopy__(self):
#         raise NotImplementedError(type(str(self)))
#
#
# class StackedLazyVariable(Node):
#     def __init__(
#             self,
#             vars: List[Node]
#     ):
#         self._value_list = [
#             v if isinstance(v, Node) else Node(v)
#             for v in vars
#         ]
#         super(StackedLazyVariable, self).__init__(
#             value=self.__call__, length=len(vars)
#         )
#
#     def __call__(self, *args, **kwargs):
#         return np.asarray([v() for v in self._value_list])

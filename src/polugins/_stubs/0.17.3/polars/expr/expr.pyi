from _typeshed import Incomplete
from datetime import timedelta
from polars.dataframe import DataFrame as DataFrame
from polars.datatypes import Categorical as Categorical, FLOAT_DTYPES as FLOAT_DTYPES, INTEGER_DTYPES as INTEGER_DTYPES, Struct as Struct, UInt32 as UInt32, Utf8 as Utf8, is_polars_dtype as is_polars_dtype, py_type_to_dtype as py_type_to_dtype
from polars.dependencies import _check_for_numpy as _check_for_numpy, numpy as np
from polars.expr.binary import ExprBinaryNameSpace as ExprBinaryNameSpace
from polars.expr.categorical import ExprCatNameSpace as ExprCatNameSpace
from polars.expr.datetime import ExprDateTimeNameSpace as ExprDateTimeNameSpace
from polars.expr.list import ExprListNameSpace as ExprListNameSpace
from polars.expr.meta import ExprMetaNameSpace as ExprMetaNameSpace
from polars.expr.string import ExprStringNameSpace as ExprStringNameSpace
from polars.expr.struct import ExprStructNameSpace as ExprStructNameSpace
from polars.lazyframe import LazyFrame as LazyFrame
from polars.polars import PyExpr as PyExpr
from polars.series import Series as Series
from polars.type_aliases import ApplyStrategy as ApplyStrategy, ClosedInterval as ClosedInterval, FillNullStrategy as FillNullStrategy, InterpolationMethod as InterpolationMethod, IntoExpr as IntoExpr, NullBehavior as NullBehavior, PolarsDataType as PolarsDataType, PythonLiteral as PythonLiteral, RankMethod as RankMethod, RollingInterpolationMethod as RollingInterpolationMethod, SearchSortedSide as SearchSortedSide
from polars.utils._parse_expr_input import expr_to_lit_or_expr as expr_to_lit_or_expr, selection_to_pyexpr_list as selection_to_pyexpr_list
from polars.utils.convert import _timedelta_to_pl_duration as _timedelta_to_pl_duration
from polars.utils.decorators import deprecated_alias as deprecated_alias, redirect as redirect
from polars.utils.meta import threadpool_size as threadpool_size
from polars.utils.various import find_stacklevel as find_stacklevel, sphinx_accessor as sphinx_accessor
from typing import Any, Callable, Collection, Iterable, NoReturn, Sequence, TypeVar
from typing_extensions import Concatenate, Self

T = TypeVar('T')
P: Incomplete

class Expr:
    _pyexpr: PyExpr
    _accessors: set[str]
    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Self: ...
    def _to_pyexpr(self, other: Any) -> PyExpr: ...
    def _to_expr(self, other: Any) -> Expr: ...
    def _repr_html_(self) -> str: ...
    def __str__(self) -> str: ...
    def __bool__(self) -> NoReturn: ...
    def __abs__(self) -> Self: ...
    def __add__(self, other: Any) -> Self: ...
    def __radd__(self, other: Any) -> Self: ...
    def __and__(self, other: Expr) -> Self: ...
    def __rand__(self, other: Any) -> Self: ...
    def __eq__(self, other: Any) -> Self: ...
    def __floordiv__(self, other: Any) -> Self: ...
    def __rfloordiv__(self, other: Any) -> Self: ...
    def __ge__(self, other: Any) -> Self: ...
    def __gt__(self, other: Any) -> Self: ...
    def __invert__(self) -> Self: ...
    def __le__(self, other: Any) -> Self: ...
    def __lt__(self, other: Any) -> Self: ...
    def __mod__(self, other: Any) -> Self: ...
    def __rmod__(self, other: Any) -> Self: ...
    def __mul__(self, other: Any) -> Self: ...
    def __rmul__(self, other: Any) -> Self: ...
    def __ne__(self, other: Any) -> Self: ...
    def __neg__(self) -> Expr: ...
    def __or__(self, other: Expr) -> Self: ...
    def __ror__(self, other: Any) -> Self: ...
    def __pos__(self) -> Expr: ...
    def __pow__(self, power: int | float | Series | Expr) -> Self: ...
    def __rpow__(self, base: int | float | Expr) -> Expr: ...
    def __sub__(self, other: Any) -> Self: ...
    def __rsub__(self, other: Any) -> Self: ...
    def __truediv__(self, other: Any) -> Self: ...
    def __rtruediv__(self, other: Any) -> Self: ...
    def __xor__(self, other: Expr) -> Self: ...
    def __rxor__(self, other: Expr) -> Self: ...
    def __getstate__(self) -> Any: ...
    def __setstate__(self, state: Any) -> None: ...
    def __array_ufunc__(self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any) -> Self: ...
    def to_physical(self) -> Self: ...
    def any(self) -> Self: ...
    def all(self) -> Self: ...
    def arg_true(self) -> Self: ...
    def sqrt(self) -> Self: ...
    def log10(self) -> Self: ...
    def exp(self) -> Self: ...
    def alias(self, name: str) -> Self: ...
    def exclude(self, columns: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType], *more_columns: str | PolarsDataType) -> Self: ...
    def keep_name(self) -> Self: ...
    def pipe(self, function: Callable[Concatenate[Expr, P], T], *args: P.args, **kwargs: P.kwargs) -> T: ...
    def prefix(self, prefix: str) -> Self: ...
    def suffix(self, suffix: str) -> Self: ...
    def map_alias(self, function: Callable[[str], str]) -> Self: ...
    def is_not(self) -> Self: ...
    def is_null(self) -> Self: ...
    def is_not_null(self) -> Self: ...
    def is_finite(self) -> Self: ...
    def is_infinite(self) -> Self: ...
    def is_nan(self) -> Self: ...
    def is_not_nan(self) -> Self: ...
    def agg_groups(self) -> Self: ...
    def count(self) -> Self: ...
    def len(self) -> Self: ...
    def slice(self, offset: int | Expr, length: int | Expr | None = ...) -> Self: ...
    def append(self, other: Expr, *, upcast: bool = ...) -> Self: ...
    def rechunk(self) -> Self: ...
    def drop_nulls(self) -> Self: ...
    def drop_nans(self) -> Self: ...
    def cumsum(self, *, reverse: bool = ...) -> Self: ...
    def cumprod(self, *, reverse: bool = ...) -> Self: ...
    def cummin(self, *, reverse: bool = ...) -> Self: ...
    def cummax(self, *, reverse: bool = ...) -> Self: ...
    def cumcount(self, *, reverse: bool = ...) -> Self: ...
    def floor(self) -> Self: ...
    def ceil(self) -> Self: ...
    def round(self, decimals: int) -> Self: ...
    def dot(self, other: Expr | str) -> Self: ...
    def mode(self) -> Self: ...
    def cast(self, dtype: PolarsDataType | type[Any], *, strict: bool = ...) -> Self: ...
    def sort(self, *, descending: bool = ..., nulls_last: bool = ...) -> Self: ...
    def top_k(self, k: int = ...) -> Self: ...
    def bottom_k(self, k: int = ...) -> Self: ...
    def arg_sort(self, *, descending: bool = ..., nulls_last: bool = ...) -> Self: ...
    def arg_max(self) -> Self: ...
    def arg_min(self) -> Self: ...
    def search_sorted(self, element: Expr | int | float | Series, side: SearchSortedSide = ...) -> Self: ...
    def sort_by(self, by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr, descending: bool | Sequence[bool] = ...) -> Self: ...
    def take(self, indices: int | list[int] | Expr | Series | np.ndarray[Any, Any]) -> Self: ...
    def shift(self, periods: int = ...) -> Self: ...
    def shift_and_fill(self, fill_value: int | float | bool | str | Expr | list[Any], *, periods: int = ...) -> Self: ...
    def fill_null(self, value: Any | None = ..., strategy: FillNullStrategy | None = ..., limit: int | None = ...) -> Self: ...
    def fill_nan(self, value: int | float | Expr | None) -> Self: ...
    def forward_fill(self, limit: int | None = ...) -> Self: ...
    def backward_fill(self, limit: int | None = ...) -> Self: ...
    def reverse(self) -> Self: ...
    def std(self, ddof: int = ...) -> Self: ...
    def var(self, ddof: int = ...) -> Self: ...
    def max(self) -> Self: ...
    def min(self) -> Self: ...
    def nan_max(self) -> Self: ...
    def nan_min(self) -> Self: ...
    def sum(self) -> Self: ...
    def mean(self) -> Self: ...
    def median(self) -> Self: ...
    def product(self) -> Self: ...
    def n_unique(self) -> Self: ...
    def approx_unique(self) -> Self: ...
    def null_count(self) -> Self: ...
    def arg_unique(self) -> Self: ...
    def unique(self, *, maintain_order: bool = ...) -> Self: ...
    def first(self) -> Self: ...
    def last(self) -> Self: ...
    def over(self, expr: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Self: ...
    def is_unique(self) -> Self: ...
    def is_first(self) -> Self: ...
    def is_duplicated(self) -> Self: ...
    def quantile(self, quantile: float | Expr, interpolation: RollingInterpolationMethod = ...) -> Self: ...
    def filter(self, predicate: Expr) -> Self: ...
    def where(self, predicate: Expr) -> Self: ...
    def map(self, function: Callable[[Series], Series | Any], return_dtype: PolarsDataType | None = ..., *, agg_list: bool = ...) -> Self: ...
    def apply(self, function: Callable[[Series], Series] | Callable[[Any], Any], return_dtype: PolarsDataType | None = ..., *, skip_nulls: bool = ..., pass_name: bool = ..., strategy: ApplyStrategy = ...) -> Self: ...
    def flatten(self) -> Self: ...
    def explode(self) -> Self: ...
    def implode(self) -> Self: ...
    def take_every(self, n: int) -> Self: ...
    def head(self, n: int | Expr = ...) -> Self: ...
    def tail(self, n: int | Expr = ...) -> Self: ...
    def limit(self, n: int | Expr = ...) -> Self: ...
    def and_(self, *others: Any) -> Self: ...
    def or_(self, *others: Any) -> Self: ...
    def eq(self, other: Any) -> Self: ...
    def ge(self, other: Any) -> Self: ...
    def gt(self, other: Any) -> Self: ...
    def le(self, other: Any) -> Self: ...
    def lt(self, other: Any) -> Self: ...
    def ne(self, other: Any) -> Self: ...
    def add(self, other: Any) -> Self: ...
    def floordiv(self, other: Any) -> Self: ...
    def mod(self, other: Any) -> Self: ...
    def mul(self, other: Any) -> Self: ...
    def sub(self, other: Any) -> Self: ...
    def truediv(self, other: Any) -> Self: ...
    def pow(self, exponent: int | float | Series | Expr) -> Self: ...
    def xor(self, other: Any) -> Self: ...
    def is_in(self, other: Expr | Collection[Any] | Series) -> Self: ...
    def repeat_by(self, by: Expr | str) -> Self: ...
    def is_between(self, lower_bound: IntoExpr, upper_bound: IntoExpr, closed: ClosedInterval = ...) -> Self: ...
    def hash(self, seed: int = ..., seed_1: int | None = ..., seed_2: int | None = ..., seed_3: int | None = ...) -> Self: ...
    def reinterpret(self, *, signed: bool = ...) -> Self: ...
    def inspect(self, fmt: str = ...) -> Self: ...
    def interpolate(self, method: InterpolationMethod = ...) -> Self: ...
    def rolling_min(self, window_size: int | timedelta | str, weights: list[float] | None = ..., min_periods: int | None = ..., *, center: bool = ..., by: str | None = ..., closed: ClosedInterval = ...) -> Self: ...
    def rolling_max(self, window_size: int | timedelta | str, weights: list[float] | None = ..., min_periods: int | None = ..., *, center: bool = ..., by: str | None = ..., closed: ClosedInterval = ...) -> Self: ...
    def rolling_mean(self, window_size: int | timedelta | str, weights: list[float] | None = ..., min_periods: int | None = ..., *, center: bool = ..., by: str | None = ..., closed: ClosedInterval = ...) -> Self: ...
    def rolling_sum(self, window_size: int | timedelta | str, weights: list[float] | None = ..., min_periods: int | None = ..., *, center: bool = ..., by: str | None = ..., closed: ClosedInterval = ...) -> Self: ...
    def rolling_std(self, window_size: int | timedelta | str, weights: list[float] | None = ..., min_periods: int | None = ..., *, center: bool = ..., by: str | None = ..., closed: ClosedInterval = ...) -> Self: ...
    def rolling_var(self, window_size: int | timedelta | str, weights: list[float] | None = ..., min_periods: int | None = ..., *, center: bool = ..., by: str | None = ..., closed: ClosedInterval = ...) -> Self: ...
    def rolling_median(self, window_size: int | timedelta | str, weights: list[float] | None = ..., min_periods: int | None = ..., *, center: bool = ..., by: str | None = ..., closed: ClosedInterval = ...) -> Self: ...
    def rolling_quantile(self, quantile: float, interpolation: RollingInterpolationMethod = ..., window_size: int | timedelta | str = ..., weights: list[float] | None = ..., min_periods: int | None = ..., *, center: bool = ..., by: str | None = ..., closed: ClosedInterval = ...) -> Self: ...
    def rolling_apply(self, function: Callable[[Series], Any], window_size: int, weights: list[float] | None = ..., min_periods: int | None = ..., *, center: bool = ...) -> Self: ...
    def rolling_skew(self, window_size: int, *, bias: bool = ...) -> Self: ...
    def abs(self) -> Self: ...
    def rank(self, method: RankMethod = ..., *, descending: bool = ..., seed: int | None = ...) -> Self: ...
    def diff(self, n: int = ..., null_behavior: NullBehavior = ...) -> Self: ...
    def pct_change(self, n: int = ...) -> Self: ...
    def skew(self, *, bias: bool = ...) -> Self: ...
    def kurtosis(self, *, fisher: bool = ..., bias: bool = ...) -> Self: ...
    def clip(self, lower_bound: int | float, upper_bound: int | float) -> Self: ...
    def clip_min(self, lower_bound: int | float) -> Self: ...
    def clip_max(self, upper_bound: int | float) -> Self: ...
    def lower_bound(self) -> Self: ...
    def upper_bound(self) -> Self: ...
    def sign(self) -> Self: ...
    def sin(self) -> Self: ...
    def cos(self) -> Self: ...
    def tan(self) -> Self: ...
    def arcsin(self) -> Self: ...
    def arccos(self) -> Self: ...
    def arctan(self) -> Self: ...
    def sinh(self) -> Self: ...
    def cosh(self) -> Self: ...
    def tanh(self) -> Self: ...
    def arcsinh(self) -> Self: ...
    def arccosh(self) -> Self: ...
    def arctanh(self) -> Self: ...
    def reshape(self, dimensions: tuple[int, ...]) -> Self: ...
    def shuffle(self, seed: int | None = ...) -> Self: ...
    def sample(self, n: int | None = ..., *, fraction: float | None = ..., with_replacement: bool = ..., shuffle: bool = ..., seed: int | None = ...) -> Self: ...
    def ewm_mean(self, com: float | None = ..., span: float | None = ..., half_life: float | None = ..., alpha: float | None = ..., *, adjust: bool = ..., min_periods: int = ..., ignore_nulls: bool = ...) -> Self: ...
    def ewm_std(self, com: float | None = ..., span: float | None = ..., half_life: float | None = ..., alpha: float | None = ..., *, adjust: bool = ..., bias: bool = ..., min_periods: int = ..., ignore_nulls: bool = ...) -> Self: ...
    def ewm_var(self, com: float | None = ..., span: float | None = ..., half_life: float | None = ..., alpha: float | None = ..., *, adjust: bool = ..., bias: bool = ..., min_periods: int = ..., ignore_nulls: bool = ...) -> Self: ...
    def extend_constant(self, value: PythonLiteral | None, n: int) -> Self: ...
    def value_counts(self, *, multithreaded: bool = ..., sort: bool = ...) -> Self: ...
    def unique_counts(self) -> Self: ...
    def log(self, base: float = ...) -> Self: ...
    def log1p(self) -> Self: ...
    def entropy(self, base: float = ..., *, normalize: bool = ...) -> Self: ...
    def cumulative_eval(self, expr: Expr, min_periods: int = ..., *, parallel: bool = ...) -> Self: ...
    def set_sorted(self, *, descending: bool = ...) -> Self: ...
    def shrink_dtype(self) -> Self: ...
    def map_dict(self, remapping: dict[Any, Any], *, default: Any = ..., return_dtype: PolarsDataType | None = ...) -> Self: ...
    @property
    def arr(self) -> ExprListNameSpace: ...
    @property
    def bin(self) -> ExprBinaryNameSpace: ...
    @property
    def cat(self) -> ExprCatNameSpace: ...
    @property
    def dt(self) -> ExprDateTimeNameSpace: ...
    @property
    def meta(self) -> ExprMetaNameSpace: ...
    @property
    def str(self) -> ExprStringNameSpace: ...
    @property
    def struct(self) -> ExprStructNameSpace: ...

def _prepare_alpha(com: float | int | None = ..., span: float | int | None = ..., half_life: float | int | None = ..., alpha: float | int | None = ...) -> float: ...
def _prepare_rolling_window_args(window_size: int | timedelta | str, min_periods: int | None = ...) -> tuple[str, int]: ...

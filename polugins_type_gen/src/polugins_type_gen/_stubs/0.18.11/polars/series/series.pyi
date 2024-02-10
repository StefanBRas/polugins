#: version 0.18.11
import np as np
import pa as pa
import pd as pd
from builtins import PySeries
from datetime import date, datetime, timedelta
from polars.datatypes.classes import Array as Array, Boolean as Boolean, Categorical as Categorical, Date as Date, Datetime as Datetime, Decimal as Decimal, Duration as Duration, FLOAT_DTYPES as FLOAT_DTYPES, Float32 as Float32, Float64 as Float64, Int16 as Int16, Int32 as Int32, Int64 as Int64, Int8 as Int8, List as List, NUMERIC_DTYPES as NUMERIC_DTYPES, Object as Object, SIGNED_INTEGER_DTYPES as SIGNED_INTEGER_DTYPES, TEMPORAL_DTYPES as TEMPORAL_DTYPES, Time as Time, UInt32 as UInt32, UInt64 as UInt64, UNSIGNED_INTEGER_DTYPES as UNSIGNED_INTEGER_DTYPES, Unknown as Unknown, Utf8 as Utf8
from polars.datatypes.convert import dtype_to_ctype as dtype_to_ctype, is_polars_dtype as is_polars_dtype, maybe_cast as maybe_cast, numpy_char_code_to_dtype as numpy_char_code_to_dtype, py_type_to_dtype as py_type_to_dtype, supported_numpy_char_code as supported_numpy_char_code
from polars.dependencies import _check_for_numpy as _check_for_numpy, _check_for_pandas as _check_for_pandas, _check_for_pyarrow as _check_for_pyarrow
from polars.series.array import ArrayNameSpace as ArrayNameSpace
from polars.series.binary import BinaryNameSpace as BinaryNameSpace
from polars.series.categorical import CatNameSpace as CatNameSpace
from polars.series.datetime import DateTimeNameSpace as DateTimeNameSpace
from polars.series.list import ListNameSpace as ListNameSpace
from polars.series.string import StringNameSpace as StringNameSpace
from polars.series.struct import StructNameSpace as StructNameSpace
from polars.series.utils import expr_dispatch as expr_dispatch, get_ffi_func as get_ffi_func
from polars.slice import PolarsSlice as PolarsSlice
from polars.utils._construction import arrow_to_pyseries as arrow_to_pyseries, iterable_to_pyseries as iterable_to_pyseries, numpy_to_idxs as numpy_to_idxs, numpy_to_pyseries as numpy_to_pyseries, pandas_to_pyseries as pandas_to_pyseries, sequence_to_pyseries as sequence_to_pyseries, series_to_pyseries as series_to_pyseries
from polars.utils._wrap import wrap_df as wrap_df
from polars.utils.convert import _date_to_pl_date as _date_to_pl_date, _datetime_to_pl_timestamp as _datetime_to_pl_timestamp, _time_to_pl_time as _time_to_pl_time
from polars.utils.deprecation import deprecate_renamed_parameter as deprecate_renamed_parameter, issue_deprecation_warning as issue_deprecation_warning
from polars.utils.meta import get_index_type as get_index_type
from polars.utils.various import _is_generator as _is_generator, parse_version as parse_version, range_to_series as range_to_series, range_to_slice as range_to_slice, scale_bytes as scale_bytes, sphinx_accessor as sphinx_accessor
from typing import Any, ArrayLike, Callable, ClassVar as _ClassVar, Collection, Generator, NoReturn, Sequence

TYPE_CHECKING: bool
INTEGER_DTYPES: frozenset
_PYARROW_AVAILABLE: bool

class Series:
    _s: _ClassVar[None] = ...
    _accessors: _ClassVar[set] = ...
    def __init__(self, name: str | ArrayLike | None = ..., values: ArrayLike | None = ..., dtype: PolarsDataType | None = ...) -> None: ...
    @classmethod
    def _from_pyseries(cls, pyseries: PySeries) -> Self: ...
    @classmethod
    def _from_arrow(cls, name: str, values: pa.Array) -> Self:
        """Construct a Series from an Arrow Array."""
    @classmethod
    def _from_pandas(cls, name: str, values: pd.Series[Any] | pd.DatetimeIndex) -> Self:
        """Construct a Series from a pandas Series or DatetimeIndex."""
    def _get_ptr(self) -> int:
        """
        Get a pointer to the start of the values buffer of a numeric Series.

        This will raise an error if the
        ``Series`` contains multiple chunks

        """
    def __bool__(self) -> NoReturn: ...
    def __len__(self) -> int: ...
    def __and__(self, other: Series) -> Self: ...
    def __rand__(self, other: Series) -> Series: ...
    def __or__(self, other: Series) -> Self: ...
    def __ror__(self, other: Series) -> Self: ...
    def __xor__(self, other: Series) -> Self: ...
    def __rxor__(self, other: Series) -> Series: ...
    def _comp(self, other: Any, op: ComparisonOperator) -> Self: ...
    def __eq__(self, other: Any) -> Self | Expr: ...
    def __ne__(self, other: Any) -> Self | Expr: ...
    def __gt__(self, other: Any) -> Self | Expr: ...
    def __lt__(self, other: Any) -> Self | Expr: ...
    def __ge__(self, other: Any) -> Self | Expr: ...
    def __le__(self, other: Any) -> Self | Expr: ...
    def le(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series <= other``."""
    def lt(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series < other``."""
    def eq(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series == other``."""
    def eq_missing(self, other: Any) -> Self | Expr:
        """
        Method equivalent of equality operator ``expr == other`` where `None` == None`.

        This differs from default ``ne`` where null values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        """
    def ne(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series != other``."""
    def ne_missing(self, other: Any) -> Self | Expr:
        """
        Method equivalent of equality operator ``expr != other`` where `None` == None`.

        This differs from default ``ne`` where null values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        """
    def ge(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series >= other``."""
    def gt(self, other: Any) -> Self | Expr:
        """Method equivalent of operator expression ``series > other``."""
    def _arithmetic(self, other: Any, op_s: str, op_ffi: str) -> Self: ...
    def __add__(self, other: Any) -> Self | DataFrame | Expr: ...
    def __sub__(self, other: Any) -> Self | Expr: ...
    def __truediv__(self, other: Any) -> Series | Expr: ...
    def __floordiv__(self, other: Any) -> Series | Expr: ...
    def __invert__(self) -> Self: ...
    def __mul__(self, other: Any) -> Series | DataFrame | Expr: ...
    def __mod__(self, other: Any) -> Series | Expr: ...
    def __rmod__(self, other: Any) -> Series: ...
    def __radd__(self, other: Any) -> Series: ...
    def __rsub__(self, other: Any) -> Series: ...
    def __rtruediv__(self, other: Any) -> Series: ...
    def __rfloordiv__(self, other: Any) -> Series: ...
    def __rmul__(self, other: Any) -> Series: ...
    def __pow__(self, exponent: int | float | None | Series) -> Series: ...
    def __rpow__(self, other: Any) -> Series: ...
    def __matmul__(self, other: Any) -> float | Series | None: ...
    def __rmatmul__(self, other: Any) -> float | Series | None: ...
    def __neg__(self) -> Series: ...
    def __pos__(self) -> Series: ...
    def __abs__(self) -> Series: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: None = ...) -> Self: ...
    def __contains__(self, item: Any) -> bool: ...
    def __iter__(self) -> Generator[Any, None, None]: ...
    def _pos_idxs(self, size: int) -> Series: ...
    def _take_with_series(self, s: Series) -> Series: ...
    def __getitem__(self, item: int | Series | range | slice | np.ndarray[Any, Any] | list[int]) -> Any: ...
    def __setitem__(self, key: int | Series | np.ndarray[Any, Any] | Sequence[object] | tuple[object], value: Any) -> None: ...
    def __array__(self, dtype: Any = ...) -> np.ndarray[Any, Any]:
        """
        Numpy __array__ interface protocol.

        Ensures that `np.asarray(pl.Series(..))` works as expected, see
        https://numpy.org/devdocs/user/basics.interoperability.html#the-array-method.
        """
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Series:
        """Numpy universal functions."""
    def _repr_html_(self) -> str:
        """Format output data in HTML for display in Jupyter Notebooks."""
    def item(self, row: int | None = ...) -> Any:
        '''
        Return the series as a scalar, or return the element at the given row index.

        If no row index is provided, this is equivalent to ``s[0]``, with a check
        that the shape is (1,). With a row index, this is equivalent to ``s[row]``.

        Examples
        --------
        >>> s1 = pl.Series("a", [1])
        >>> s1.item()
        1
        >>> s2 = pl.Series("a", [9, 8, 7])
        >>> s2.cumsum().item(-1)
        24

        '''
    def estimated_size(self, unit: SizeUnit = ...) -> int | float:
        '''
        Return an estimation of the total (heap) allocated size of the Series.

        Estimated size is given in the specified unit (bytes by default).

        This estimation is the sum of the size of its buffers, validity, including
        nested arrays. Multiple arrays may share buffers and bitmaps. Therefore, the
        size of 2 arrays is not the sum of the sizes computed from this function. In
        particular, [`StructArray`]\'s size is an upper bound.

        When an array is sliced, its allocated size remains constant because the buffer
        unchanged. However, this function will yield a smaller number. This is because
        this function returns the visible size of the buffer, not its total capacity.

        FFI buffers are included in this estimation.

        Parameters
        ----------
        unit : {\'b\', \'kb\', \'mb\', \'gb\', \'tb\'}
            Scale the returned size to the given unit.

        Examples
        --------
        >>> s = pl.Series("values", list(range(1_000_000)), dtype=pl.UInt32)
        >>> s.estimated_size()
        4000000
        >>> s.estimated_size("mb")
        3.814697265625

        '''
    def sqrt(self) -> Series:
        """
        Compute the square root of the elements.

        Syntactic sugar for

        >>> pl.Series([1, 2]) ** 0.5
        shape: (2,)
        Series: '' [f64]
        [
            1.0
            1.414214
        ]

        """
    def cbrt(self) -> Series:
        """
        Compute the cube root of the elements.

        Optimization for

        >>> pl.Series([1, 2]) ** (1.0 / 3)
        shape: (2,)
        Series: '' [f64]
        [
            1.0
            1.259921
        ]

        """
    def any(self, drop_nulls: bool = ...) -> bool | None:
        """
        Check if any boolean value in the column is `True`.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        """
    def all(self, drop_nulls: bool = ...) -> bool | None:
        """
        Check if all boolean values in the column are `True`.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        """
    def log(self, base: float = ...) -> Series:
        """Compute the logarithm to a given base."""
    def log1p(self) -> Series:
        """Compute the natural logarithm of the input array plus one, element-wise."""
    def log10(self) -> Series:
        """Compute the base 10 logarithm of the input array, element-wise."""
    def exp(self) -> Series:
        """Compute the exponential, element-wise."""
    def drop_nulls(self) -> Series:
        """
        Drop all null values.

        Creates a new Series that copies data from this Series without null values.
        """
    def drop_nans(self) -> Series:
        """Drop NaN values."""
    def to_frame(self, name: str | None = ...) -> DataFrame:
        '''
        Cast this Series to a DataFrame.

        Parameters
        ----------
        name
            optionally name/rename the Series column in the new DataFrame.

        Examples
        --------
        >>> s = pl.Series("a", [123, 456])
        >>> df = s.to_frame()
        >>> df
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 123 │
        │ 456 │
        └─────┘

        >>> df = s.to_frame("xyz")
        >>> df
        shape: (2, 1)
        ┌─────┐
        │ xyz │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 123 │
        │ 456 │
        └─────┘

        '''
    def describe(self, percentiles: Sequence[float] | float | None = ...) -> DataFrame:
        '''
        Quick summary statistics of a series.

        Series with mixed datatypes will return summary statistics for the datatype of
        the first value.

        Parameters
        ----------
        percentiles
            One or more percentiles to include in the summary statistics (if the
            series has a numeric dtype). All values must be in the range `[0, 1]`.

        Returns
        -------
        DataFrame
            Mapping with summary statistics of a Series.

        Examples
        --------
        >>> series_num = pl.Series([1, 2, 3, 4, 5])
        >>> series_num.describe()
        shape: (9, 2)
        ┌────────────┬──────────┐
        │ statistic  ┆ value    │
        │ ---        ┆ ---      │
        │ str        ┆ f64      │
        ╞════════════╪══════════╡
        │ count      ┆ 5.0      │
        │ null_count ┆ 0.0      │
        │ mean       ┆ 3.0      │
        │ std        ┆ 1.581139 │
        │ min        ┆ 1.0      │
        │ max        ┆ 5.0      │
        │ median     ┆ 3.0      │
        │ 25%        ┆ 2.0      │
        │ 75%        ┆ 4.0      │
        └────────────┴──────────┘

        >>> series_str = pl.Series(["a", "a", None, "b", "c"])
        >>> series_str.describe()
        shape: (3, 2)
        ┌────────────┬───────┐
        │ statistic  ┆ value │
        │ ---        ┆ ---   │
        │ str        ┆ i64   │
        ╞════════════╪═══════╡
        │ count      ┆ 5     │
        │ null_count ┆ 1     │
        │ unique     ┆ 4     │
        └────────────┴───────┘

        '''
    def sum(self) -> int | float:
        '''
        Reduce this Series to the sum value.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.sum()
        6

        '''
    def mean(self) -> int | float | None:
        '''
        Reduce this Series to the mean value.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.mean()
        2.0

        '''
    def product(self) -> int | float:
        """Reduce this Series to the product value."""
    def pow(self, exponent: int | float | None | Series) -> Series:
        '''
        Raise to the power of the given exponent.

        Parameters
        ----------
        exponent
            The exponent. Accepts Series input.

        Examples
        --------
        >>> s = pl.Series("foo", [1, 2, 3, 4])
        >>> s.pow(3)
        shape: (4,)
        Series: \'foo\' [f64]
        [
                1.0
                8.0
                27.0
                64.0
        ]

        '''
    def min(self) -> PythonLiteral | None:
        '''
        Get the minimal value in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.min()
        1

        '''
    def max(self) -> PythonLiteral | None:
        '''
        Get the maximum value in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.max()
        3

        '''
    def nan_max(self) -> int | float | date | datetime | timedelta | str:
        """
        Get maximum value, but propagate/poison encountered NaN values.

        This differs from numpy's `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        """
    def nan_min(self) -> int | float | date | datetime | timedelta | str:
        """
        Get minimum value, but propagate/poison encountered NaN values.

        This differs from numpy's `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        """
    def std(self, ddof: int = ...) -> float | None:
        '''
        Get the standard deviation of this Series.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.std()
        1.0

        '''
    def var(self, ddof: int = ...) -> float | None:
        '''
        Get variance of this Series.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.var()
        1.0

        '''
    def median(self) -> float | None:
        '''
        Get the median of this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.median()
        2.0

        '''
    def quantile(self, quantile: float, interpolation: RollingInterpolationMethod = ...) -> float | None:
        '''
        Get the quantile value of this Series.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {\'nearest\', \'higher\', \'lower\', \'midpoint\', \'linear\'}
            Interpolation method.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.quantile(0.5)
        2.0

        '''
    def to_dummies(self, separator: str = ...) -> DataFrame:
        '''
        Get dummy/indicator variables.

        Parameters
        ----------
        separator
            Separator/delimiter used when generating column names.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_dummies()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a_1 ┆ a_2 ┆ a_3 │
        │ --- ┆ --- ┆ --- │
        │ u8  ┆ u8  ┆ u8  │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 0   ┆ 0   │
        │ 0   ┆ 1   ┆ 0   │
        │ 0   ┆ 0   ┆ 1   │
        └─────┴─────┴─────┘

        '''
    def cut(self, *args, **kwargs) -> DataFrame | Series:
        '''
        Bin continuous values into discrete categories.

        Parameters
        ----------
        breaks
            A list of unique cut points.
        labels
            Labels to assign to the bins. If given the length of labels must be
            len(breaks) + 1.
        break_point_label
            Name given to the breakpoint column/field. Only used if series == False or
            include_breaks == True
        category_label
            Name given to the category column. Only used if series == False
        series
            If True, return a categorical Series in the data\'s original order.
        left_closed
            Whether intervals should be [) instead of (]
        include_breaks
            Include the the right endpoint of the bin each observation falls in.
            If returning a DataFrame, it will be a column, and if returning a Series
            it will be a field in a Struct

        Returns
        -------
        DataFrame or Series

        Examples
        --------
        >>> a = pl.Series("a", [v / 10 for v in range(-30, 30, 5)])
        >>> a.cut([-1, 1], series=False)
        shape: (12, 3)
        ┌──────┬─────────────┬────────────┐
        │ a    ┆ break_point ┆ category   │
        │ ---  ┆ ---         ┆ ---        │
        │ f64  ┆ f64         ┆ cat        │
        ╞══════╪═════════════╪════════════╡
        │ -3.0 ┆ -1.0        ┆ (-inf, -1] │
        │ -2.5 ┆ -1.0        ┆ (-inf, -1] │
        │ -2.0 ┆ -1.0        ┆ (-inf, -1] │
        │ -1.5 ┆ -1.0        ┆ (-inf, -1] │
        │ …    ┆ …           ┆ …          │
        │ 1.0  ┆ 1.0         ┆ (-1, 1]    │
        │ 1.5  ┆ inf         ┆ (1, inf]   │
        │ 2.0  ┆ inf         ┆ (1, inf]   │
        │ 2.5  ┆ inf         ┆ (1, inf]   │
        └──────┴─────────────┴────────────┘
        >>> a.cut([-1, 1], series=True)
        shape: (12,)
        Series: \'a\' [cat]
        [
            "(-inf, -1]"
            "(-inf, -1]"
            "(-inf, -1]"
            "(-inf, -1]"
            "(-inf, -1]"
            "(-1, 1]"
            "(-1, 1]"
            "(-1, 1]"
            "(-1, 1]"
            "(1, inf]"
            "(1, inf]"
            "(1, inf]"
        ]
        >>> a.cut([-1, 1], series=True, left_closed=True)
        shape: (12,)
        Series: \'a\' [cat]
        [
            "[-inf, -1)"
            "[-inf, -1)"
            "[-inf, -1)"
            "[-inf, -1)"
            "[-1, 1)"
            "[-1, 1)"
            "[-1, 1)"
            "[-1, 1)"
            "[1, inf)"
            "[1, inf)"
            "[1, inf)"
            "[1, inf)"
        ]
        '''
    def qcut(self, *args, **kwargs) -> DataFrame | Series:
        '''
        Discretize continuous values into discrete categories based on their quantiles.

        Parameters
        ----------
        q
            Either a list of quantile probabilities between 0 and 1 or a positive
            integer determining the number of evenly spaced probabilities to use.
        labels
            Labels to assign to the quantiles. If given the length of labels must be
            len(breaks) + 1.
        break_point_label
            Name given to the breakpoint column/field. Only used if series == False or
            include_breaks == True
        category_label
            Name given to the category column. Only used if series == False.
        series
            If True, return a categorical Series in the data\'s original order
        left_closed
            Whether intervals should be [) instead of (]
        allow_duplicates
            If True, the resulting quantile breaks don\'t have to be unique. This can
            happen even with unique probs depending on the data. Duplicates will be
            dropped, resulting in fewer bins.
        include_breaks
            Include the the right endpoint of the bin each observation falls in.
            If returning a DataFrame, it will be a column, and if returning a Series
            it will be a field in a Struct

        Returns
        -------
        DataFrame or Series

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Examples
        --------
        >>> a = pl.Series("a", range(-5, 3))
        >>> a.qcut(2, series=True)
        shape: (8,)
        Series: \'a\' [cat]
        [
                "(-inf, -1.5]"
                "(-inf, -1.5]"
                "(-inf, -1.5]"
                "(-inf, -1.5]"
                "(-1.5, inf]"
                "(-1.5, inf]"
                "(-1.5, inf]"
                "(-1.5, inf]"
        ]
        >>> a.qcut([0.0, 0.25, 0.75], series=False)
        shape: (8, 3)
        ┌─────┬─────────────┬───────────────┐
        │ a   ┆ break_point ┆ category      │
        │ --- ┆ ---         ┆ ---           │
        │ i64 ┆ f64         ┆ cat           │
        ╞═════╪═════════════╪═══════════════╡
        │ -5  ┆ -5.0        ┆ (-inf, -5]    │
        │ -4  ┆ -3.25       ┆ (-5, -3.25]   │
        │ -3  ┆ 0.25        ┆ (-3.25, 0.25] │
        │ -2  ┆ 0.25        ┆ (-3.25, 0.25] │
        │ -1  ┆ 0.25        ┆ (-3.25, 0.25] │
        │ 0   ┆ 0.25        ┆ (-3.25, 0.25] │
        │ 1   ┆ inf         ┆ (0.25, inf]   │
        │ 2   ┆ inf         ┆ (0.25, inf]   │
        └─────┴─────────────┴───────────────┘
        >>> a.qcut([0.0, 0.25, 0.75], series=True)
        shape: (8,)
        Series: \'a\' [cat]
        [
            "(-inf, -5]"
            "(-5, -3.25]"
            "(-3.25, 0.25]"
            "(-3.25, 0.25]"
            "(-3.25, 0.25]"
            "(-3.25, 0.25]"
            "(0.25, inf]"
            "(0.25, inf]"
        ]
        >>> a.qcut([0.0, 0.25, 0.75], series=True, left_closed=True)
        shape: (8,)
        Series: \'a\' [cat]
        [
            "[-5, -3.25)"
            "[-5, -3.25)"
            "[-3.25, 0.25)"
            "[-3.25, 0.25)"
            "[-3.25, 0.25)"
            "[-3.25, 0.25)"
            "[0.25, inf)"
            "[0.25, inf)"
        ]
        '''
    def rle(self) -> Series:
        '''
        Get the lengths of runs of identical values.

        Returns
        -------
        Series
            Series of data type :class:`Struct` with Fields "lengths" and "values".

        Examples
        --------
        >>> s = pl.Series("s", [1, 1, 2, 1, None, 1, 3, 3])
        >>> s.rle().struct.unnest()
        shape: (6, 2)
        ┌─────────┬────────┐
        │ lengths ┆ values │
        │ ---     ┆ ---    │
        │ i32     ┆ i64    │
        ╞═════════╪════════╡
        │ 2       ┆ 1      │
        │ 1       ┆ 2      │
        │ 1       ┆ 1      │
        │ 1       ┆ null   │
        │ 1       ┆ 1      │
        │ 2       ┆ 3      │
        └─────────┴────────┘
        '''
    def rle_id(self) -> Series:
        '''
        Map values to run IDs.

        Similar to RLE, but it maps each value to an ID corresponding to the run into
        which it falls. This is especially useful when you want to define groups by
        runs of identical values rather than the values themselves.

        Returns
        -------
        Series

        See Also
        --------
        rle

        Examples
        --------
        >>> s = pl.Series("s", [1, 1, 2, 1, None, 1, 3, 3])
        >>> s.rle_id()
        shape: (8,)
        Series: \'s\' [u32]
        [
            0
            0
            1
            2
            3
            4
            5
            5
        ]
        '''
    def hist(self, bins: list[float] | None = ...) -> DataFrame:
        '''
        Bin values into buckets and count their occurrences.

        Parameters
        ----------
        bins
            Discretizations to make.
            If None given, we determine the boundaries based on the data.
        bin_count
            If no bins provided, this will be used to determine
            the distance of the bins

        Returns
        -------
        DataFrame

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Examples
        --------
        >>> a = pl.Series("a", [1, 3, 8, 8, 2, 1, 3])
        >>> a.hist(bin_count=4)
        shape: (5, 3)
        ┌─────────────┬─────────────┬─────────┐
        │ break_point ┆ category    ┆ a_count │
        │ ---         ┆ ---         ┆ ---     │
        │ f64         ┆ cat         ┆ u32     │
        ╞═════════════╪═════════════╪═════════╡
        │ 0.0         ┆ (-inf, 0.0] ┆ 0       │
        │ 2.25        ┆ (0.0, 2.25] ┆ 3       │
        │ 4.5         ┆ (2.25, 4.5] ┆ 2       │
        │ 6.75        ┆ (4.5, 6.75] ┆ 0       │
        │ inf         ┆ (6.75, inf] ┆ 2       │
        └─────────────┴─────────────┴─────────┘

        '''
    def value_counts(self) -> DataFrame:
        '''
        Count the unique values in a Series.

        Parameters
        ----------
        sort
            Ensure the output is sorted from most values to least.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.value_counts().sort(by="a")
        shape: (3, 2)
        ┌─────┬────────┐
        │ a   ┆ counts │
        │ --- ┆ ---    │
        │ i64 ┆ u32    │
        ╞═════╪════════╡
        │ 1   ┆ 1      │
        │ 2   ┆ 2      │
        │ 3   ┆ 1      │
        └─────┴────────┘

        '''
    def unique_counts(self) -> Series:
        '''
        Return a count of the unique values in the order of appearance.

        Examples
        --------
        >>> s = pl.Series("id", ["a", "b", "b", "c", "c", "c"])
        >>> s.unique_counts()
        shape: (3,)
        Series: \'id\' [u32]
        [
            1
            2
            3
        ]

        '''
    def entropy(self, base: float = ...) -> float | None:
        """
        Computes the entropy.

        Uses the formula ``-sum(pk * log(pk)`` where ``pk`` are discrete probabilities.

        Parameters
        ----------
        base
            Given base, defaults to `e`
        normalize
            Normalize pk if it doesn't sum to 1.

        Examples
        --------
        >>> a = pl.Series([0.99, 0.005, 0.005])
        >>> a.entropy(normalize=True)
        0.06293300616044681
        >>> b = pl.Series([0.65, 0.10, 0.25])
        >>> b.entropy(normalize=True)
        0.8568409950394724

        """
    def cumulative_eval(self, expr: Expr, min_periods: int = ...) -> Series:
        '''
        Run an expression over a sliding window that increases `1` slot every iteration.

        Parameters
        ----------
        expr
            Expression to evaluate
        min_periods
            Number of valid values there should be in the window before the expression
            is evaluated. valid values = `length - null_count`
        parallel
            Run in parallel. Don\'t do this in a groupby or another operation that
            already has much parallelization.

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        This can be really slow as it can have `O(n^2)` complexity. Don\'t use this
        for operations that visit all elements.

        Examples
        --------
        >>> s = pl.Series("values", [1, 2, 3, 4, 5])
        >>> s.cumulative_eval(pl.element().first() - pl.element().last() ** 2)
        shape: (5,)
        Series: \'values\' [f64]
        [
            0.0
            -3.0
            -8.0
            -15.0
            -24.0
        ]

        '''
    def alias(self, name: str) -> Series:
        '''
        Rename the series.

        Parameters
        ----------
        name
            The new name.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.alias("b")
        shape: (3,)
        Series: \'b\' [i64]
        [
                1
                2
                3
        ]

        '''
    def rename(self, name: str) -> Series:
        '''
        Rename this Series.

        Parameters
        ----------
        name
            New name.
        in_place
            Modify the Series in-place.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.rename("b")
        shape: (3,)
        Series: \'b\' [i64]
        [
                1
                2
                3
        ]

        '''
    def chunk_lengths(self) -> list[int]:
        '''
        Get the length of each individual chunk.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("a", [4, 5, 6])

        Concatenate Series with rechunk = True

        >>> pl.concat([s, s2]).chunk_lengths()
        [6]

        Concatenate Series with rechunk = False

        >>> pl.concat([s, s2], rechunk=False).chunk_lengths()
        [3, 3]

        '''
    def n_chunks(self) -> int:
        '''
        Get the number of chunks that this Series contains.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.n_chunks()
        1
        >>> s2 = pl.Series("a", [4, 5, 6])

        Concatenate Series with rechunk = True

        >>> pl.concat([s, s2]).n_chunks()
        1

        Concatenate Series with rechunk = False

        >>> pl.concat([s, s2], rechunk=False).n_chunks()
        2

        '''
    def cummax(self) -> Series:
        '''
        Get an array with the cumulative max computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Examples
        --------
        >>> s = pl.Series("s", [3, 5, 1])
        >>> s.cummax()
        shape: (3,)
        Series: \'s\' [i64]
        [
            3
            5
            5
        ]

        '''
    def cummin(self) -> Series:
        '''
        Get an array with the cumulative min computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Examples
        --------
        >>> s = pl.Series("s", [1, 2, 3])
        >>> s.cummin()
        shape: (3,)
        Series: \'s\' [i64]
        [
            1
            1
            1
        ]

        '''
    def cumprod(self) -> Series:
        '''
        Get an array with the cumulative product computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cumprod()
        shape: (3,)
        Series: \'a\' [i64]
        [
            1
            2
            6
        ]

        '''
    def cumsum(self) -> Series:
        '''
        Get an array with the cumulative sum computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.cumsum()
        shape: (3,)
        Series: \'a\' [i64]
        [
            1
            3
            6
        ]

        '''
    def slice(self, offset: int, length: int | None = ...) -> Series:
        '''
        Get a slice of this Series.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to ``None``, all rows starting at the offset
            will be selected.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.slice(1, 2)
        shape: (2,)
        Series: \'a\' [i64]
        [
                2
                3
        ]

        '''
    def append(self, other: Series) -> Self:
        '''
        Append a Series to this one.

        Parameters
        ----------
        other
            Series to append.
        append_chunks
            .. deprecated:: 0.18.8
                This argument will be removed and ``append`` will change to always
                behave like ``append_chunks=True`` (the previous default). For the
                behavior of ``append_chunks=False``, use ``Series.extend``.

            If set to `True` the append operation will add the chunks from `other` to
            self. This is super cheap.

            If set to `False` the append operation will do the same as
            `DataFrame.extend` which extends the memory backed by this `Series` with
            the values from `other`.

            Different from `append chunks`, `extend` appends the data from `other` to
            the underlying memory locations and thus may cause a reallocation (which are
            expensive).

            If this does not cause a reallocation, the resulting data structure will not
            have any extra chunks and thus will yield faster queries.

            Prefer `extend` over `append_chunks` when you want to do a query after a
            single append. For instance during online operations where you add `n` rows
            and rerun a query.

            Prefer `append_chunks` over `extend` when you want to append many times
            before doing a query. For instance when you read in multiple files and when
            to store them in a single `Series`. In the latter case, finish the sequence
            of `append_chunks` operations with a `rechunk`.

        Warnings
        --------
        This method modifies the series in-place. The series is returned for
        convenience only.

        See Also
        --------
        extend

        Examples
        --------
        >>> a = pl.Series("a", [1, 2, 3])
        >>> b = pl.Series("b", [4, 5])
        >>> a.append(b)
        shape: (5,)
        Series: \'a\' [i64]
        [
            1
            2
            3
            4
            5
        ]

        The resulting series will consist of multiple chunks.

        >>> a.n_chunks()
        2

        '''
    def extend(self, other: Series) -> Self:
        '''
        Extend the memory backed by this Series with the values from another.

        Different from ``append``, which adds the chunks from ``other`` to the chunks of
        this series, ``extend`` appends the data from ``other`` to the underlying memory
        locations and thus may cause a reallocation (which is expensive).

        If this does `not` cause a reallocation, the resulting data structure will not
        have any extra chunks and thus will yield faster queries.

        Prefer ``extend`` over ``append`` when you want to do a query after a single
        append. For instance, during online operations where you add `n` rows
        and rerun a query.

        Prefer ``append`` over ``extend`` when you want to append many times
        before doing a query. For instance, when you read in multiple files and want
        to store them in a single ``Series``. In the latter case, finish the sequence
        of ``append`` operations with a `rechunk`.

        Parameters
        ----------
        other
            Series to extend the series with.

        Warnings
        --------
        This method modifies the series in-place. The series is returned for
        convenience only.

        See Also
        --------
        append

        Examples
        --------
        >>> a = pl.Series("a", [1, 2, 3])
        >>> b = pl.Series("b", [4, 5])
        >>> a.extend(b)
        shape: (5,)
        Series: \'a\' [i64]
        [
            1
            2
            3
            4
            5
        ]

        The resulting series will consist of a single chunk.

        >>> a.n_chunks()
        1

        '''
    def filter(self, predicate: Series | list[bool]) -> Self:
        '''
        Filter elements by a boolean mask.

        Parameters
        ----------
        predicate
            Boolean mask.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> mask = pl.Series("", [True, False, True])
        >>> s.filter(mask)
        shape: (2,)
        Series: \'a\' [i64]
        [
                1
                3
        ]

        '''
    def head(self, n: int = ...) -> Series:
        '''
        Get the first `n` elements.

        Parameters
        ----------
        n
            Number of elements to return. If a negative value is passed, return all
            elements except the last ``abs(n)``.

        See Also
        --------
        tail, slice

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.head(3)
        shape: (3,)
        Series: \'a\' [i64]
        [
                1
                2
                3
        ]

        Pass a negative value to get all rows `except` the last ``abs(n)``.

        >>> s.head(-3)
        shape: (2,)
        Series: \'a\' [i64]
        [
                1
                2
        ]

        '''
    def tail(self, n: int = ...) -> Series:
        '''
        Get the last `n` elements.

        Parameters
        ----------
        n
            Number of elements to return. If a negative value is passed, return all
            elements except the first ``abs(n)``.

        See Also
        --------
        head, slice

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.tail(3)
        shape: (3,)
        Series: \'a\' [i64]
        [
                3
                4
                5
        ]

        Pass a negative value to get all rows `except` the first ``abs(n)``.

        >>> s.tail(-3)
        shape: (2,)
        Series: \'a\' [i64]
        [
                4
                5
        ]

        '''
    def limit(self, n: int = ...) -> Series:
        """
        Get the first `n` elements.

        Alias for :func:`Series.head`.

        Parameters
        ----------
        n
            Number of elements to return. If a negative value is passed, return all
            elements except the last ``abs(n)``.

        See Also
        --------
        head

        """
    def take_every(self, n: int) -> Series:
        '''
        Take every nth value in the Series and return as new Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.take_every(2)
        shape: (2,)
        Series: \'a\' [i64]
        [
            1
            3
        ]

        '''
    def sort(self) -> Self:
        '''
        Sort this Series.

        Parameters
        ----------
        descending
            Sort in descending order.
        in_place
            Sort in-place.

        Examples
        --------
        >>> s = pl.Series("a", [1, 3, 4, 2])
        >>> s.sort()
        shape: (4,)
        Series: \'a\' [i64]
        [
                1
                2
                3
                4
        ]
        >>> s.sort(descending=True)
        shape: (4,)
        Series: \'a\' [i64]
        [
                4
                3
                2
                1
        ]

        '''
    def top_k(self, k: int = ...) -> Series:
        '''
        Return the `k` largest elements.

        This has time complexity:

        .. math:: O(n + k \\\\log{}n - \\frac{k}{2})

        Parameters
        ----------
        k
            Number of elements to return.

        See Also
        --------
        bottom_k

        Examples
        --------
        >>> s = pl.Series("a", [2, 5, 1, 4, 3])
        >>> s.top_k(3)
        shape: (3,)
        Series: \'a\' [i64]
        [
            5
            4
            3
        ]

        '''
    def bottom_k(self, k: int = ...) -> Series:
        '''
        Return the `k` smallest elements.

        This has time complexity:

        .. math:: O(n + k \\\\log{}n - \\frac{k}{2})

        Parameters
        ----------
        k
            Number of elements to return.

        See Also
        --------
        top_k

        Examples
        --------
        >>> s = pl.Series("a", [2, 5, 1, 4, 3])
        >>> s.bottom_k(3)
        shape: (3,)
        Series: \'a\' [i64]
        [
            1
            2
            3
        ]

        '''
    def arg_sort(self) -> Series:
        '''
        Get the index values that would sort this Series.

        Parameters
        ----------
        descending
            Sort in descending order.
        nulls_last
            Place null values last instead of first.

        Examples
        --------
        >>> s = pl.Series("a", [5, 3, 4, 1, 2])
        >>> s.arg_sort()
        shape: (5,)
        Series: \'a\' [u32]
        [
            3
            4
            1
            2
            0
        ]

        '''
    def arg_unique(self) -> Series:
        '''
        Get unique index as Series.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.arg_unique()
        shape: (3,)
        Series: \'a\' [u32]
        [
                0
                1
                3
        ]

        '''
    def arg_min(self) -> int | None:
        '''
        Get the index of the minimal value.

        Returns
        -------
        int

        Examples
        --------
        >>> s = pl.Series("a", [3, 2, 1])
        >>> s.arg_min()
        2

        '''
    def arg_max(self) -> int | None:
        '''
        Get the index of the maximal value.

        Returns
        -------
        int

        Examples
        --------
        >>> s = pl.Series("a", [3, 2, 1])
        >>> s.arg_max()
        0

        '''
    def search_sorted(self, element: int | float | Series | np.ndarray[Any, Any] | list[int] | list[float], side: SearchSortedSide = ...) -> int | Series:
        """
        Find indices where elements should be inserted to maintain order.

        .. math:: a[i-1] < v <= a[i]

        Parameters
        ----------
        element
            Expression or scalar value.
        side : {'any', 'left', 'right'}
            If 'any', the index of the first suitable location found is given.
            If 'left', the index of the leftmost suitable location found is given.
            If 'right', return the rightmost suitable location found is given.

        """
    def unique(self) -> Series:
        '''
        Get unique elements in series.

        Parameters
        ----------
        maintain_order
            Maintain order of data. This requires more work.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.unique().sort()
        shape: (3,)
        Series: \'a\' [i64]
        [
            1
            2
            3
        ]

        '''
    def take(self, indices: int | list[int] | Expr | Series | np.ndarray[Any, Any]) -> Series:
        '''
        Take values by index.

        Parameters
        ----------
        indices
            Index location used for selection.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.take([1, 3])
        shape: (2,)
        Series: \'a\' [i64]
        [
                2
                4
        ]

        '''
    def null_count(self) -> int:
        """Count the null values in this Series."""
    def has_validity(self) -> bool:
        """
        Return True if the Series has a validity bitmask.

        If there is none, it means that there are no null values.
        Use this to swiftly assert a Series does not have null values.

        """
    def is_empty(self) -> bool:
        '''
        Check if the Series is empty.

        Examples
        --------
        >>> s = pl.Series("a", [], dtype=pl.Float32)
        >>> s.is_empty()
        True

        '''
    def is_sorted(self) -> bool:
        """
        Check if the Series is sorted.

        Parameters
        ----------
        descending
            Check if the Series is sorted in descending order

        """
    def is_null(self) -> Series:
        '''
        Returns a boolean Series indicating which values are null.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, None])
        >>> s.is_null()
        shape: (4,)
        Series: \'a\' [bool]
        [
            false
            false
            false
            true
        ]

        '''
    def is_not_null(self) -> Series:
        '''
        Returns a boolean Series indicating which values are not null.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, None])
        >>> s.is_not_null()
        shape: (4,)
        Series: \'a\' [bool]
        [
            true
            true
            true
            false
        ]

        '''
    def is_finite(self) -> Series:
        '''
        Returns a boolean Series indicating which values are finite.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, np.inf])
        >>> s.is_finite()
        shape: (3,)
        Series: \'a\' [bool]
        [
                true
                true
                false
        ]

        '''
    def is_infinite(self) -> Series:
        '''
        Returns a boolean Series indicating which values are infinite.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, np.inf])
        >>> s.is_infinite()
        shape: (3,)
        Series: \'a\' [bool]
        [
                false
                false
                true
        ]

        '''
    def is_nan(self) -> Series:
        '''
        Returns a boolean Series indicating which values are not NaN.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.NaN])
        >>> s.is_nan()
        shape: (4,)
        Series: \'a\' [bool]
        [
                false
                false
                false
                true
        ]

        '''
    def is_not_nan(self) -> Series:
        '''
        Returns a boolean Series indicating which values are not NaN.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.NaN])
        >>> s.is_not_nan()
        shape: (4,)
        Series: \'a\' [bool]
        [
                true
                true
                true
                false
        ]

        '''
    def is_in(self, other: Series | Collection[Any]) -> Series:
        '''
        Check if elements of this Series are in the other Series.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [2, 4])
        >>> s2.is_in(s)
        shape: (2,)
        Series: \'b\' [bool]
        [
                true
                false
        ]

        >>> # check if some values are a member of sublists
        >>> sets = pl.Series("sets", [[1, 2, 3], [1, 2], [9, 10]])
        >>> optional_members = pl.Series("optional_members", [1, 2, 3])
        >>> print(sets)
        shape: (3,)
        Series: \'sets\' [list[i64]]
        [
            [1, 2, 3]
            [1, 2]
            [9, 10]
        ]
        >>> print(optional_members)
        shape: (3,)
        Series: \'optional_members\' [i64]
        [
            1
            2
            3
        ]
        >>> optional_members.is_in(sets)
        shape: (3,)
        Series: \'optional_members\' [bool]
        [
            true
            true
            false
        ]

        '''
    def arg_true(self) -> Series:
        '''
        Get index values where Boolean Series evaluate True.

        Returns
        -------
        Series
            Series of data type :class:`UInt32`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> (s == 2).arg_true()
        shape: (1,)
        Series: \'a\' [u32]
        [
                1
        ]

        '''
    def is_unique(self) -> Series:
        '''
        Get mask of all unique values.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.is_unique()
        shape: (4,)
        Series: \'a\' [bool]
        [
                true
                false
                false
                true
        ]

        '''
    def is_first(self) -> Series:
        """
        Get a mask of the first unique value.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        """
    def is_duplicated(self) -> Series:
        '''
        Get mask of all duplicated values.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.is_duplicated()
        shape: (4,)
        Series: \'a\' [bool]
        [
                false
                true
                true
                false
        ]

        '''
    def explode(self) -> Series:
        """
        Explode a list Series.

        This means that every item is expanded to a new row.

        Returns
        -------
        Series
            Series with the data type of the list elements.

        See Also
        --------
        Series.list.explode : Explode a list column.
        Series.str.explode : Explode a string column.

        """
    def series_equal(self, other: Series) -> bool:
        '''
        Check if series is equal with another Series.

        Parameters
        ----------
        other
            Series to compare with.
        null_equal
            Consider null values as equal.
        strict
            Don\'t allow different numerical dtypes, e.g. comparing `pl.UInt32` with a
            `pl.Int64` will return `False`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4, 5, 6])
        >>> s.series_equal(s)
        True
        >>> s.series_equal(s2)
        False

        '''
    def len(self) -> int:
        '''
        Length of this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.len()
        3

        '''
    def cast(self, dtype: PolarsDataType | type[int] | type[float] | type[str] | type[bool]) -> Self:
        '''
        Cast between data types.

        Parameters
        ----------
        dtype
            DataType to cast to.
        strict
            Throw an error if a cast could not be done for instance due to an overflow.

        Examples
        --------
        >>> s = pl.Series("a", [True, False, True])
        >>> s
        shape: (3,)
        Series: \'a\' [bool]
        [
            true
            false
            true
        ]

        >>> s.cast(pl.UInt32)
        shape: (3,)
        Series: \'a\' [u32]
        [
            1
            0
            1
        ]

        '''
    def to_physical(self) -> Series:
        '''
        Cast to physical representation of the logical dtype.

        - :func:`polars.datatypes.Date` -> :func:`polars.datatypes.Int32`
        - :func:`polars.datatypes.Datetime` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Time` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Duration` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Categorical` -> :func:`polars.datatypes.UInt32`
        - ``List(inner)`` -> ``List(physical of inner)``
        - Other data types will be left unchanged.

        Examples
        --------
        Replicating the pandas
        `pd.Series.factorize
        <https://pandas.pydata.org/docs/reference/api/pandas.Series.factorize.html>`_
        method.

        >>> s = pl.Series("values", ["a", None, "x", "a"])
        >>> s.cast(pl.Categorical).to_physical()
        shape: (4,)
        Series: \'values\' [u32]
        [
            0
            null
            1
            0
        ]

        '''
    def to_list(self) -> list[Any]:
        '''
        Convert this Series to a Python List. This operation clones data.

        Parameters
        ----------
        use_pyarrow
            Use pyarrow for the conversion.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]
        >>> type(s.to_list())
        <class \'list\'>

        '''
    def rechunk(self) -> Self:
        """
        Create a single chunk of memory for this Series.

        Parameters
        ----------
        in_place
            In place or not.

        """
    def reverse(self) -> Series:
        '''
        Return Series in reverse order.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3], dtype=pl.Int8)
        >>> s.reverse()
        shape: (3,)
        Series: \'a\' [i8]
        [
            3
            2
            1
        ]

        '''
    def is_between(self, lower_bound: IntoExpr, upper_bound: IntoExpr, closed: ClosedInterval = ...) -> Series:
        '''
        Get a boolean mask of the values that fall between the given start/end values.

        Parameters
        ----------
        lower_bound
            Lower bound value. Accepts expression input. Non-expression inputs
            (including strings) are parsed as literals.
        upper_bound
            Upper bound value. Accepts expression input. Non-expression inputs
            (including strings) are parsed as literals.
        closed : {\'both\', \'left\', \'right\', \'none\'}
            Define which sides of the interval are closed (inclusive).

        Examples
        --------
        >>> s = pl.Series("num", [1, 2, 3, 4, 5])
        >>> s.is_between(2, 4)
        shape: (5,)
        Series: \'num\' [bool]
        [
            false
            true
            true
            true
            false
        ]

        Use the ``closed`` argument to include or exclude the values at the bounds:

        >>> s.is_between(2, 4, closed="left")
        shape: (5,)
        Series: \'num\' [bool]
        [
            false
            true
            true
            false
            false
        ]

        You can also use strings as well as numeric/temporal values:

        >>> s = pl.Series("s", ["a", "b", "c", "d", "e"])
        >>> s.is_between("b", "d", closed="both")
        shape: (5,)
        Series: \'s\' [bool]
        [
            false
            true
            true
            true
            false
        ]

        '''
    def is_numeric(self) -> bool:
        '''
        Check if this Series datatype is numeric.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.is_numeric()
        True

        '''
    def is_integer(self, signed: bool | None = ...) -> bool:
        '''
        Check if this Series datatype is an integer (signed or unsigned).

        Parameters
        ----------
        signed
            * if `None`, both signed and unsigned integer dtypes will match.
            * if `True`, only signed integer dtypes will be considered a match.
            * if `False`, only unsigned integer dtypes will be considered a match.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3], dtype=pl.UInt32)
        >>> s.is_integer()
        True
        >>> s.is_integer(signed=False)
        True
        >>> s.is_integer(signed=True)
        False

        '''
    def is_temporal(self, excluding: OneOrMoreDataTypes | None = ...) -> bool:
        """
        Check if this Series datatype is temporal.

        Parameters
        ----------
        excluding
            Optionally exclude one or more temporal dtypes from matching.

        Examples
        --------
        >>> from datetime import date
        >>> s = pl.Series([date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)])
        >>> s.is_temporal()
        True
        >>> s.is_temporal(excluding=[pl.Date])
        False

        """
    def is_float(self) -> bool:
        '''
        Check if this Series has floating point numbers.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0])
        >>> s.is_float()
        True

        '''
    def is_boolean(self) -> bool:
        '''
        Check if this Series is a Boolean.

        Examples
        --------
        >>> s = pl.Series("a", [True, False, True])
        >>> s.is_boolean()
        True

        '''
    def is_utf8(self) -> bool:
        '''
        Check if this Series datatype is a Utf8.

        Examples
        --------
        >>> s = pl.Series("x", ["a", "b", "c"])
        >>> s.is_utf8()
        True

        '''
    def view(self) -> SeriesView:
        '''
        Get a view into this Series data with a numpy array.

        This operation doesn\'t clone data, but does not include missing values.
        Don\'t use this unless you know what you are doing.

        Parameters
        ----------
        ignore_nulls
            If True then nulls are converted to 0.
            If False then an Exception is raised if nulls are present.

        Examples
        --------
        >>> s = pl.Series("a", [1, None])
        >>> s.view(ignore_nulls=True)
        SeriesView([1, 0])

        '''
    def to_numpy(self, *args: Any) -> np.ndarray[Any, Any]:
        '''
        Convert this Series to numpy.

        This operation may clone data but is completely safe. Note that:

        - data which is purely numeric AND without null values is not cloned;
        - floating point ``nan`` values can be zero-copied;
        - booleans can\'t be zero-copied.

        To ensure that no data is cloned, set ``zero_copy_only=True``.

        Alternatively, if you want a zero-copy view and know what you are doing,
        use `.view()`.

        Parameters
        ----------
        *args
            args will be sent to pyarrow.Array.to_numpy.
        zero_copy_only
            If True, an exception will be raised if the conversion to a numpy
            array would require copying the underlying data (e.g. in presence
            of nulls, or for non-primitive types).
        writable
            For numpy arrays created with zero copy (view on the Arrow data),
            the resulting array is not writable (Arrow data is immutable).
            By setting this to True, a copy of the array is made to ensure
            it is writable.
        use_pyarrow
            Use `pyarrow.Array.to_numpy
            <https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.to_numpy>`_

            for the conversion to numpy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> arr = s.to_numpy()
        >>> arr  # doctest: +IGNORE_RESULT
        array([1, 2, 3], dtype=int64)
        >>> type(arr)
        <class \'numpy.ndarray\'>

        '''
    def to_arrow(self) -> pa.Array:
        '''
        Get the underlying Arrow Array.

        If the Series contains only a single chunk this operation is zero copy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s = s.to_arrow()
        >>> s  # doctest: +ELLIPSIS
        <pyarrow.lib.Int64Array object at ...>
        [
          1,
          2,
          3
        ]

        '''
    def to_pandas(self, *args: Any, **kwargs: Any) -> pd.Series[Any]:
        '''
        Convert this Series to a pandas Series.

        This requires that :mod:`pandas` and :mod:`pyarrow` are installed.
        This operation clones data, unless `use_pyarrow_extension_array=True`.

        Parameters
        ----------
        use_pyarrow_extension_array
            Further operations on this Pandas series, might trigger conversion to numpy.
            Use PyArrow backed-extension array instead of numpy array for pandas
            Series. This allows zero copy operations and preservation of nulls
            values.
            Further operations on this pandas Series, might trigger conversion
            to NumPy arrays if that operation is not supported by pyarrow compute
            functions.
        kwargs
            Arguments will be sent to :meth:`pyarrow.Table.to_pandas`.

        Examples
        --------
        >>> s1 = pl.Series("a", [1, 2, 3])
        >>> s1.to_pandas()
        0    1
        1    2
        2    3
        Name: a, dtype: int64
        >>> s1.to_pandas(use_pyarrow_extension_array=True)  # doctest: +SKIP
        0    1
        1    2
        2    3
        Name: a, dtype: int64[pyarrow]
        >>> s2 = pl.Series("b", [1, 2, None, 4])
        >>> s2.to_pandas()
        0    1.0
        1    2.0
        2    NaN
        3    4.0
        Name: b, dtype: float64
        >>> s2.to_pandas(use_pyarrow_extension_array=True)  # doctest: +SKIP
        0       1
        1       2
        2    <NA>
        3       4
        Name: b, dtype: int64[pyarrow]

        '''
    def to_init_repr(self, n: int = ...) -> str:
        '''
        Convert Series to instantiatable string representation.

        Parameters
        ----------
        n
            Only use first n elements.

        See Also
        --------
        polars.Series.to_init_repr
        polars.from_repr

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, None, 4], dtype=pl.Int16)
        >>> print(s.to_init_repr())
        pl.Series("a", [1, 2, None, 4], dtype=pl.Int16)
        >>> s_from_str_repr = eval(s.to_init_repr())
        >>> s_from_str_repr
        shape: (4,)
        Series: \'a\' [i16]
        [
            1
            2
            null
            4
        ]

        '''
    def set(self, filter: Series, value: int | float | str) -> Series:
        '''
        Set masked values.

        Parameters
        ----------
        filter
            Boolean mask.
        value
            Value with which to replace the masked values.

        Notes
        -----
        Use of this function is frequently an anti-pattern, as it can
        block optimisation (predicate pushdown, etc). Consider using
        `pl.when(predicate).then(value).otherwise(self)` instead.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.set(s == 2, 10)
        shape: (3,)
        Series: \'a\' [i64]
        [
                1
                10
                3
        ]

        It is better to implement this as follows:

        >>> s.to_frame().select(
        ...     pl.when(pl.col("a") == 2).then(10).otherwise(pl.col("a"))
        ... )
        shape: (3, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ i64     │
        ╞═════════╡
        │ 1       │
        │ 10      │
        │ 3       │
        └─────────┘

        '''
    def set_at_idx(self, idx: Series | np.ndarray[Any, Any] | Sequence[int] | int, value: int | float | str | bool | Sequence[int] | Sequence[float] | Sequence[bool] | Sequence[str] | Sequence[date] | Sequence[datetime] | date | datetime | Series | None) -> Series:
        '''
        Set values at the index locations.

        Parameters
        ----------
        idx
            Integers representing the index locations.
        value
            replacement values.

        Returns
        -------
        Series
            The mutated series.

        Notes
        -----
        Use of this function is frequently an anti-pattern, as it can
        block optimisation (predicate pushdown, etc). Consider using
        `pl.when(predicate).then(value).otherwise(self)` instead.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.set_at_idx(1, 10)
        shape: (3,)
        Series: \'a\' [i64]
        [
                1
                10
                3
        ]

        It is better to implement this as follows:

        >>> s.to_frame().with_row_count("row_nr").select(
        ...     pl.when(pl.col("row_nr") == 1).then(10).otherwise(pl.col("a"))
        ... )
        shape: (3, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ i64     │
        ╞═════════╡
        │ 1       │
        │ 10      │
        │ 3       │
        └─────────┘

        '''
    def clear(self, n: int = ...) -> Series:
        '''
        Create an empty copy of the current Series, with zero to \'n\' elements.

        The copy has an identical name/dtype, but no data.

        Parameters
        ----------
        n
            Number of (empty) elements to return in the cleared frame.

        See Also
        --------
        clone : Cheap deepcopy/clone.

        Examples
        --------
        >>> s = pl.Series("a", [None, True, False])
        >>> s.clear()
        shape: (0,)
        Series: \'a\' [bool]
        [
        ]

        >>> s.clear(n=2)
        shape: (2,)
        Series: \'a\' [bool]
        [
            null
            null
        ]

        '''
    def clone(self) -> Self:
        '''
        Very cheap deepcopy/clone.

        See Also
        --------
        clear : Create an empty copy of the current Series, with identical
            schema but no data.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.clone()
        shape: (3,)
        Series: \'a\' [i64]
        [
                1
                2
                3
        ]

        '''
    def fill_nan(self, value: int | float | Expr | None) -> Series:
        '''
        Fill floating point NaN value with a fill value.

        Parameters
        ----------
        value
            Value used to fill NaN values.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, float("nan")])
        >>> s.fill_nan(0)
        shape: (4,)
        Series: \'a\' [f64]
        [
                1.0
                2.0
                3.0
                0.0
        ]

        '''
    def fill_null(self, value: Any | None = ..., strategy: FillNullStrategy | None = ..., limit: int | None = ...) -> Series:
        '''
        Fill null values using the specified value or strategy.

        Parameters
        ----------
        value
            Value used to fill null values.
        strategy : {None, \'forward\', \'backward\', \'min\', \'max\', \'mean\', \'zero\', \'one\'}
            Strategy used to fill null values.
        limit
            Number of consecutive null values to fill when using the \'forward\' or
            \'backward\' strategy.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, None])
        >>> s.fill_null(strategy="forward")
        shape: (4,)
        Series: \'a\' [i64]
        [
            1
            2
            3
            3
        ]
        >>> s.fill_null(strategy="min")
        shape: (4,)
        Series: \'a\' [i64]
        [
            1
            2
            3
            1
        ]
        >>> s = pl.Series("b", ["x", None, "z"])
        >>> s.fill_null(pl.lit(""))
        shape: (3,)
        Series: \'b\' [str]
        [
            "x"
            ""
            "z"
        ]

        '''
    def floor(self) -> Series:
        '''
        Rounds down to the nearest integer value.

        Only works on floating point Series.

        Examples
        --------
        >>> s = pl.Series("a", [1.12345, 2.56789, 3.901234])
        >>> s.floor()
        shape: (3,)
        Series: \'a\' [f64]
        [
                1.0
                2.0
                3.0
        ]

        '''
    def ceil(self) -> Series:
        '''
        Rounds up to the nearest integer value.

        Only works on floating point Series.

        Examples
        --------
        >>> s = pl.Series("a", [1.12345, 2.56789, 3.901234])
        >>> s.ceil()
        shape: (3,)
        Series: \'a\' [f64]
        [
                2.0
                3.0
                4.0
        ]

        '''
    def round(self, decimals: int = ...) -> Series:
        '''
        Round underlying floating point data by `decimals` digits.

        Examples
        --------
        >>> s = pl.Series("a", [1.12345, 2.56789, 3.901234])
        >>> s.round(2)
        shape: (3,)
        Series: \'a\' [f64]
        [
                1.12
                2.57
                3.9
        ]

        Parameters
        ----------
        decimals
            number of decimals to round by.

        '''
    def dot(self, other: Series | ArrayLike) -> float | None:
        '''
        Compute the dot/inner product between two Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4.0, 5.0, 6.0])
        >>> s.dot(s2)
        32.0

        Parameters
        ----------
        other
            Series (or array) to compute dot product with.

        '''
    def mode(self) -> Series:
        '''
        Compute the most occurring value(s).

        Can return multiple Values.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.mode()
        shape: (1,)
        Series: \'a\' [i64]
        [
                2
        ]

        '''
    def sign(self) -> Series:
        '''
        Compute the element-wise indication of the sign.

        The returned values can be -1, 0, or 1:

        * -1 if x  < 0.
        *  0 if x == 0.
        *  1 if x  > 0.

        (null values are preserved as-is).

        Examples
        --------
        >>> s = pl.Series("a", [-9.0, -0.0, 0.0, 4.0, None])
        >>> s.sign()
        shape: (5,)
        Series: \'a\' [i64]
        [
                -1
                0
                0
                1
                null
        ]

        '''
    def sin(self) -> Series:
        '''
        Compute the element-wise value for the sine.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.sin()
        shape: (3,)
        Series: \'a\' [f64]
        [
            0.0
            1.0
            1.2246e-16
        ]

        '''
    def cos(self) -> Series:
        '''
        Compute the element-wise value for the cosine.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.cos()
        shape: (3,)
        Series: \'a\' [f64]
        [
            1.0
            6.1232e-17
            -1.0
        ]

        '''
    def tan(self) -> Series:
        '''
        Compute the element-wise value for the tangent.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.tan()
        shape: (3,)
        Series: \'a\' [f64]
        [
            0.0
            1.6331e16
            -1.2246e-16
        ]

        '''
    def arcsin(self) -> Series:
        '''
        Compute the element-wise value for the inverse sine.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arcsin()
        shape: (3,)
        Series: \'a\' [f64]
        [
            1.570796
            0.0
            -1.570796
        ]

        '''
    def arccos(self) -> Series:
        '''
        Compute the element-wise value for the inverse cosine.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arccos()
        shape: (3,)
        Series: \'a\' [f64]
        [
            0.0
            1.570796
            3.141593
        ]

        '''
    def arctan(self) -> Series:
        '''
        Compute the element-wise value for the inverse tangent.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arctan()
        shape: (3,)
        Series: \'a\' [f64]
        [
            0.785398
            0.0
            -0.785398
        ]

        '''
    def arcsinh(self) -> Series:
        '''
        Compute the element-wise value for the inverse hyperbolic sine.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.arcsinh()
        shape: (3,)
        Series: \'a\' [f64]
        [
            0.881374
            0.0
            -0.881374
        ]

        '''
    def arccosh(self) -> Series:
        '''
        Compute the element-wise value for the inverse hyperbolic cosine.

        Examples
        --------
        >>> s = pl.Series("a", [5.0, 1.0, 0.0, -1.0])
        >>> s.arccosh()
        shape: (4,)
        Series: \'a\' [f64]
        [
            2.292432
            0.0
            NaN
            NaN
        ]

        '''
    def arctanh(self) -> Series:
        '''
        Compute the element-wise value for the inverse hyperbolic tangent.

        Examples
        --------
        >>> s = pl.Series("a", [2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.1])
        >>> s.arctanh()
        shape: (7,)
        Series: \'a\' [f64]
        [
            NaN
            inf
            0.549306
            0.0
            -0.549306
            -inf
            NaN
        ]

        '''
    def sinh(self) -> Series:
        '''
        Compute the element-wise value for the hyperbolic sine.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.sinh()
        shape: (3,)
        Series: \'a\' [f64]
        [
            1.175201
            0.0
            -1.175201
        ]

        '''
    def cosh(self) -> Series:
        '''
        Compute the element-wise value for the hyperbolic cosine.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.cosh()
        shape: (3,)
        Series: \'a\' [f64]
        [
            1.543081
            1.0
            1.543081
        ]

        '''
    def tanh(self) -> Series:
        '''
        Compute the element-wise value for the hyperbolic tangent.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 0.0, -1.0])
        >>> s.tanh()
        shape: (3,)
        Series: \'a\' [f64]
        [
            0.761594
            0.0
            -0.761594
        ]

        '''
    def apply(self, function: Callable[[Any], Any], return_dtype: PolarsDataType | None = ...) -> Self:
        '''
        Apply a custom/user-defined function (UDF) over elements in this Series.

        .. warning::
            This method is much slower than the native expressions API.
            Only use it if you cannot implement your logic otherwise.

        If the function returns a different datatype, the return_dtype arg should
        be set, otherwise the method will fail.

        Implementing logic using a Python function is almost always _significantly_
        slower and more memory intensive than implementing the same logic using
        the native expression API because:

        - The native expression engine runs in Rust; UDFs run in Python.
        - Use of Python UDFs forces the DataFrame to be materialized in memory.
        - Polars-native expressions can be parallelised (UDFs typically cannot).
        - Polars-native expressions can be logically optimised (UDFs cannot).

        Wherever possible you should strongly prefer the native expression API
        to achieve the best performance.

        Parameters
        ----------
        function
            Custom function or lambda.
        return_dtype
            Output datatype. If none is given, the same datatype as this Series will be
            used.
        skip_nulls
            Nulls will be skipped and not passed to the python function.
            This is faster because python can be skipped and because we call
            more specialized functions.

        Warnings
        --------
        If ``return_dtype`` is not provided, this may lead to unexpected results.
        We allow this, but it is considered a bug in the user\'s query.

        Notes
        -----
        If your function is expensive and you don\'t want it to be called more than
        once for a given input, consider applying an ``@lru_cache`` decorator to it.
        With suitable data you may achieve order-of-magnitude speedups (or more).

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.apply(lambda x: x + 10)  # doctest: +SKIP
        shape: (3,)
        Series: \'a\' [i64]
        [
                11
                12
                13
        ]

        Returns
        -------
        Series

        '''
    def shift(self, periods: int = ...) -> Series:
        '''
        Shift the values by a given period.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.shift(periods=1)
        shape: (3,)
        Series: \'a\' [i64]
        [
                null
                1
                2
        ]
        >>> s.shift(periods=-1)
        shape: (3,)
        Series: \'a\' [i64]
        [
                2
                3
                null
        ]

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        '''
    def shift_and_fill(self, fill_value: int | Expr) -> Series:
        """
        Shift the values by a given period and fill the resulting null values.

        Parameters
        ----------
        fill_value
            Fill None values with the result of this expression.
        periods
            Number of places to shift (may be negative).

        """
    def zip_with(self, mask: Series, other: Series) -> Self:
        """
        Take values from self or other based on the given mask.

        Where mask evaluates true, take values from self. Where mask evaluates false,
        take values from other.

        Parameters
        ----------
        mask
            Boolean Series.
        other
            Series of same type.

        Returns
        -------
        Series

        Examples
        --------
        >>> s1 = pl.Series([1, 2, 3, 4, 5])
        >>> s2 = pl.Series([5, 4, 3, 2, 1])
        >>> s1.zip_with(s1 < s2, s2)
        shape: (5,)
        Series: '' [i64]
        [
                1
                2
                3
                2
                1
        ]
        >>> mask = pl.Series([True, False, True, False, True])
        >>> s1.zip_with(mask, s2)
        shape: (5,)
        Series: '' [i64]
        [
                1
                4
                3
                2
                5
        ]

        """
    def rolling_min(self, window_size: int, weights: list[float] | None = ..., min_periods: int | None = ...) -> Series:
        '''
        Apply a rolling min (moving min) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_min(window_size=3)
        shape: (5,)
        Series: \'a\' [i64]
        [
            null
            null
            100
            200
            300
        ]

        '''
    def rolling_max(self, window_size: int, weights: list[float] | None = ..., min_periods: int | None = ...) -> Series:
        '''
        Apply a rolling max (moving max) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_max(window_size=2)
        shape: (5,)
        Series: \'a\' [i64]
        [
            null
            200
            300
            400
            500
        ]

        '''
    def rolling_mean(self, window_size: int, weights: list[float] | None = ..., min_periods: int | None = ...) -> Series:
        '''
        Apply a rolling mean (moving mean) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [100, 200, 300, 400, 500])
        >>> s.rolling_mean(window_size=2)
        shape: (5,)
        Series: \'a\' [f64]
        [
            null
            150.0
            250.0
            350.0
            450.0
        ]

        '''
    def rolling_sum(self, window_size: int, weights: list[float] | None = ..., min_periods: int | None = ...) -> Series:
        '''
        Apply a rolling sum (moving sum) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.rolling_sum(window_size=2)
        shape: (5,)
        Series: \'a\' [i64]
        [
                null
                3
                5
                7
                9
        ]

        '''
    def rolling_std(self, window_size: int, weights: list[float] | None = ..., min_periods: int | None = ...) -> Series:
        '''
        Compute a rolling std dev.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        ddof
            "Delta Degrees of Freedom": The divisor for a length N window is N - ddof

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_std(window_size=3)
        shape: (6,)
        Series: \'a\' [f64]
        [
                null
                null
                1.0
                1.0
                1.527525
                2.0
        ]

        '''
    def rolling_var(self, window_size: int, weights: list[float] | None = ..., min_periods: int | None = ...) -> Series:
        '''
        Compute a rolling variance.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        ddof
            "Delta Degrees of Freedom": The divisor for a length N window is N - ddof

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_var(window_size=3)
        shape: (6,)
        Series: \'a\' [f64]
        [
                null
                null
                1.0
                1.0
                2.333333
                4.0
        ]

        '''
    def rolling_apply(self, function: Callable[[Series], Any], window_size: int, weights: list[float] | None = ..., min_periods: int | None = ...) -> Series:
        '''
        Apply a custom rolling window function.

        Prefer the specific rolling window functions over this one, as they are faster:

            * rolling_min
            * rolling_max
            * rolling_mean
            * rolling_sum

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        function
            Aggregation function
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("A", [11.0, 2.0, 9.0, float("nan"), 8.0])
        >>> print(s.rolling_apply(function=np.nanstd, window_size=3))
        shape: (5,)
        Series: \'A\' [f64]
        [
            null
            null
            3.858612
            3.5
            0.5
        ]

        '''
    def rolling_median(self, window_size: int, weights: list[float] | None = ..., min_periods: int | None = ...) -> Series:
        '''
        Compute a rolling median.

        Parameters
        ----------
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_median(window_size=3)
        shape: (6,)
        Series: \'a\' [f64]
        [
                null
                null
                2.0
                3.0
                4.0
                6.0
        ]

        '''
    def rolling_quantile(self, quantile: float, interpolation: RollingInterpolationMethod = ..., window_size: int = ..., weights: list[float] | None = ..., min_periods: int | None = ...) -> Series:
        '''
        Compute a rolling quantile.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {\'nearest\', \'higher\', \'lower\', \'midpoint\', \'linear\'}
            Interpolation method.
        window_size
            The length of the window.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        >>> s.rolling_quantile(quantile=0.33, window_size=3)
        shape: (6,)
        Series: \'a\' [f64]
        [
                null
                null
                1.0
                2.0
                3.0
                4.0
        ]
        >>> s.rolling_quantile(quantile=0.33, interpolation="linear", window_size=3)
        shape: (6,)
        Series: \'a\' [f64]
        [
                null
                null
                1.66
                2.66
                3.66
                5.32
        ]

        '''
    def rolling_skew(self, window_size: int) -> Series:
        """
        Compute a rolling skew.

        The window at a given row includes the row itself and the
        `window_size - 1` elements before it.

        Parameters
        ----------
        window_size
            Integer size of the rolling window.
        bias
            If False, the calculations are corrected for statistical bias.

        Examples
        --------
        >>> pl.Series([1, 4, 2, 9]).rolling_skew(3)
        shape: (4,)
        Series: '' [f64]
        [
            null
            null
            0.381802
            0.47033
        ]

        Note how the values match

        >>> pl.Series([1, 4, 2]).skew(), pl.Series([4, 2, 9]).skew()
        (0.38180177416060584, 0.47033046033698594)

        """
    def sample(self, *args, **kwargs) -> Series:
        '''
        Sample from this Series.

        Parameters
        ----------
        n
            Number of items to return. Cannot be used with `fraction`. Defaults to 1 if
            `fraction` is None.
        fraction
            Fraction of items to return. Cannot be used with `n`.
        with_replacement
            Allow values to be sampled more than once.
        shuffle
            Shuffle the order of sampled data points.
        seed
            Seed for the random number generator. If set to None (default), a random
            seed is generated using the ``random`` module.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.sample(2, seed=0)  # doctest: +IGNORE_RESULT
        shape: (2,)
        Series: \'a\' [i64]
        [
            1
            5
        ]

        '''
    def peak_max(self) -> Self:
        '''
        Get a boolean mask of the local maximum peaks.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.peak_max()
        shape: (5,)
        Series: \'\' [bool]
        [
                false
                false
                false
                false
                true
        ]

        '''
    def peak_min(self) -> Self:
        '''
        Get a boolean mask of the local minimum peaks.

        Examples
        --------
        >>> s = pl.Series("a", [4, 1, 3, 2, 5])
        >>> s.peak_min()
        shape: (5,)
        Series: \'\' [bool]
        [
            false
            true
            false
            true
            false
        ]

        '''
    def n_unique(self) -> int:
        '''
        Count the number of unique values in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.n_unique()
        3

        '''
    def shrink_to_fit(self) -> Series:
        """
        Shrink Series memory usage.

        Shrinks the underlying array capacity to exactly fit the actual data.
        (Note that this function does not change the Series data type).

        """
    def hash(self, seed: int = ..., seed_1: int | None = ..., seed_2: int | None = ..., seed_3: int | None = ...) -> Series:
        '''
        Hash the Series.

        The hash value is of type `UInt64`.

        Parameters
        ----------
        seed
            Random seed parameter. Defaults to 0.
        seed_1
            Random seed parameter. Defaults to `seed` if not set.
        seed_2
            Random seed parameter. Defaults to `seed` if not set.
        seed_3
            Random seed parameter. Defaults to `seed` if not set.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.hash(seed=42)  # doctest: +IGNORE_RESULT
        shape: (3,)
        Series: \'a\' [u64]
        [
            10734580197236529959
            3022416320763508302
            13756996518000038261
        ]

        '''
    def reinterpret(self) -> Series:
        """
        Reinterpret the underlying bits as a signed/unsigned integer.

        This operation is only allowed for 64bit integers. For lower bits integers,
        you can safely use that cast operation.

        Parameters
        ----------
        signed
            If True, reinterpret as `pl.Int64`. Otherwise, reinterpret as `pl.UInt64`.

        """
    def interpolate(self, method: InterpolationMethod = ...) -> Series:
        '''
        Fill null values using interpolation.

        Parameters
        ----------
        method : {\'linear\', \'nearest\'}
            Interpolation method.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, None, None, 5])
        >>> s.interpolate()
        shape: (5,)
        Series: \'a\' [i64]
        [
            1
            2
            3
            4
            5
        ]

        '''
    def abs(self) -> Series:
        """
        Compute absolute values.

        Same as `abs(series)`.
        """
    def rank(self, method: RankMethod = ...) -> Series:
        '''
        Assign ranks to data, dealing with ties appropriately.

        Parameters
        ----------
        method : {\'average\', \'min\', \'max\', \'dense\', \'ordinal\', \'random\'}
            The method used to assign ranks to tied elements.
            The following methods are available (default is \'average\'):

            - \'average\' : The average of the ranks that would have been assigned to
              all the tied values is assigned to each value.
            - \'min\' : The minimum of the ranks that would have been assigned to all
              the tied values is assigned to each value. (This is also referred to
              as "competition" ranking.)
            - \'max\' : The maximum of the ranks that would have been assigned to all
              the tied values is assigned to each value.
            - \'dense\' : Like \'min\', but the rank of the next highest element is
              assigned the rank immediately after those assigned to the tied
              elements.
            - \'ordinal\' : All values are given a distinct rank, corresponding to
              the order that the values occur in the Series.
            - \'random\' : Like \'ordinal\', but the rank for ties is not dependent
              on the order that the values occur in the Series.
        descending
            Rank in descending order.
        seed
            If `method="random"`, use this as seed.

        Examples
        --------
        The \'average\' method:

        >>> s = pl.Series("a", [3, 6, 1, 1, 6])
        >>> s.rank()
        shape: (5,)
        Series: \'a\' [f32]
        [
            3.0
            4.5
            1.5
            1.5
            4.5
        ]

        The \'ordinal\' method:

        >>> s = pl.Series("a", [3, 6, 1, 1, 6])
        >>> s.rank("ordinal")
        shape: (5,)
        Series: \'a\' [u32]
        [
            3
            4
            1
            2
            5
        ]

        '''
    def diff(self, n: int = ..., null_behavior: NullBehavior = ...) -> Series:
        '''
        Calculate the n-th discrete difference.

        Parameters
        ----------
        n
            Number of slots to shift.
        null_behavior : {\'ignore\', \'drop\'}
            How to handle null values.

        Examples
        --------
        >>> s = pl.Series("s", values=[20, 10, 30, 25, 35], dtype=pl.Int8)
        >>> s.diff()
        shape: (5,)
        Series: \'s\' [i8]
        [
            null
            -10
            20
            -5
            10
        ]

        >>> s.diff(n=2)
        shape: (5,)
        Series: \'s\' [i8]
        [
            null
            null
            10
            15
            5
        ]

        >>> s.diff(n=2, null_behavior="drop")
        shape: (3,)
        Series: \'s\' [i8]
        [
            10
            15
            5
        ]

        '''
    def pct_change(self, n: int = ...) -> Series:
        """
        Computes percentage change between values.

        Percentage change (as fraction) between current element and most-recent
        non-null element at least ``n`` period(s) before the current element.

        Computes the change from the previous row by default.

        Parameters
        ----------
        n
            periods to shift for forming percent change.

        Examples
        --------
        >>> pl.Series(range(10)).pct_change()
        shape: (10,)
        Series: '' [f64]
        [
            null
            inf
            1.0
            0.5
            0.333333
            0.25
            0.2
            0.166667
            0.142857
            0.125
        ]

        >>> pl.Series([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]).pct_change(2)
        shape: (10,)
        Series: '' [f64]
        [
            null
            null
            3.0
            3.0
            3.0
            3.0
            3.0
            3.0
            3.0
            3.0
        ]

        """
    def skew(self) -> float | None:
        """
        Compute the sample skewness of a data set.

        For normally distributed data, the skewness should be about zero. For
        unimodal continuous distributions, a skewness value greater than zero means
        that there is more weight in the right tail of the distribution. The
        function `skewtest` can be used to determine if the skewness value
        is close enough to zero, statistically speaking.


        See scipy.stats for more information.

        Parameters
        ----------
        bias : bool, optional
            If False, the calculations are corrected for statistical bias.

        Notes
        -----
        The sample skewness is computed as the Fisher-Pearson coefficient
        of skewness, i.e.

        .. math:: g_1=\\frac{m_3}{m_2^{3/2}}

        where

        .. math:: m_i=\\frac{1}{N}\\sum_{n=1}^N(x[n]-\\bar{x})^i

        is the biased sample :math:`i\\texttt{th}` central moment, and
        :math:`\\bar{x}` is
        the sample mean.  If ``bias`` is False, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.

        .. math::
            G_1 = \\frac{k_3}{k_2^{3/2}} = \\frac{\\sqrt{N(N-1)}}{N-2}\\frac{m_3}{m_2^{3/2}}

        """
    def kurtosis(self) -> float | None:
        """
        Compute the kurtosis (Fisher or Pearson) of a dataset.

        Kurtosis is the fourth central moment divided by the square of the
        variance. If Fisher's definition is used, then 3.0 is subtracted from
        the result to give 0.0 for a normal distribution.
        If bias is False then the kurtosis is calculated using k statistics to
        eliminate bias coming from biased moment estimators

        See scipy.stats for more information

        Parameters
        ----------
        fisher : bool, optional
            If True, Fisher's definition is used (normal ==> 0.0). If False,
            Pearson's definition is used (normal ==> 3.0).
        bias : bool, optional
            If False, the calculations are corrected for statistical bias.

        """
    def clip(self, lower_bound: int | float, upper_bound: int | float) -> Series:
        '''
        Clip (limit) the values in an array to a `min` and `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        lower_bound
            Minimum value.
        upper_bound
            Maximum value.

        Examples
        --------
        >>> s = pl.Series("foo", [-50, 5, None, 50])
        >>> s.clip(1, 10)
        shape: (4,)
        Series: \'foo\' [i64]
        [
            1
            5
            null
            10
        ]

        '''
    def clip_min(self, lower_bound: int | float) -> Series:
        '''
        Clip (limit) the values in an array to a `min` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        lower_bound
            Lower bound.

        '''
    def clip_max(self, upper_bound: int | float) -> Series:
        '''
        Clip (limit) the values in an array to a `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        upper_bound
            Upper bound.

        '''
    def lower_bound(self) -> Self:
        '''
        Return the lower bound of this Series\' dtype as a unit Series.

        See Also
        --------
        upper_bound : return the upper bound of the given Series\' dtype.

        Examples
        --------
        >>> s = pl.Series("s", [-1, 0, 1], dtype=pl.Int32)
        >>> s.lower_bound()
        shape: (1,)
        Series: \'s\' [i32]
        [
            -2147483648
        ]

        >>> s = pl.Series("s", [1.0, 2.5, 3.0], dtype=pl.Float32)
        >>> s.lower_bound()
        shape: (1,)
        Series: \'s\' [f32]
        [
            -inf
        ]

        '''
    def upper_bound(self) -> Self:
        '''
        Return the upper bound of this Series\' dtype as a unit Series.

        See Also
        --------
        lower_bound : return the lower bound of the given Series\' dtype.

        Examples
        --------
        >>> s = pl.Series("s", [-1, 0, 1], dtype=pl.Int8)
        >>> s.upper_bound()
        shape: (1,)
        Series: \'s\' [i8]
        [
            127
        ]

        >>> s = pl.Series("s", [1.0, 2.5, 3.0], dtype=pl.Float64)
        >>> s.upper_bound()
        shape: (1,)
        Series: \'s\' [f64]
        [
            inf
        ]

        '''
    def map_dict(self, remapping: dict[Any, Any]) -> Self:
        '''
        Replace values in the Series using a remapping dictionary.

        Parameters
        ----------
        remapping
            Dictionary containing the before/after values to map.
        default
            Value to use when the remapping dict does not contain the lookup value.
            Use ``pl.first()``, to keep the original value.
        return_dtype
            Set return dtype to override automatic return dtype determination.

        Examples
        --------
        >>> s = pl.Series("iso3166", ["TUR", "???", "JPN", "NLD"])
        >>> country_lookup = {
        ...     "JPN": "Japan",
        ...     "TUR": "Türkiye",
        ...     "NLD": "Netherlands",
        ... }

        Remap, setting a default for unrecognised values...

        >>> s.map_dict(country_lookup, default="Unspecified").alias("country_name")
        shape: (4,)
        Series: \'country_name\' [str]
        [
            "Türkiye"
            "Unspecified"
            "Japan"
            "Netherlands"
        ]

        ...or keep the original value, by making use of ``pl.first()``:

        >>> s.map_dict(country_lookup, default=pl.first()).alias("country_name")
        shape: (4,)
        Series: \'country_name\' [str]
        [
            "Türkiye"
            "???"
            "Japan"
            "Netherlands"
        ]

        ...or keep the original value, by assigning the input series:

        >>> s.map_dict(country_lookup, default=s).alias("country_name")
        shape: (4,)
        Series: \'country_name\' [str]
        [
            "Türkiye"
            "???"
            "Japan"
            "Netherlands"
        ]

        Override return dtype:

        >>> s = pl.Series("int8", [5, 2, 3], dtype=pl.Int8)
        >>> s.map_dict({2: 7}, default=pl.first(), return_dtype=pl.Int16)
        shape: (3,)
        Series: \'int8\' [i16]
        [
            5
            7
            3
        ]

        '''
    def reshape(self, dimensions: tuple[int, ...]) -> Series:
        '''
        Reshape this Series to a flat Series or a Series of Lists.

        Parameters
        ----------
        dimensions
            Tuple of the dimension sizes. If a -1 is used in any of the dimensions, that
            dimension is inferred.

        Returns
        -------
        Series
            If a single dimension is given, results in a Series of the original
            data type.
            If a multiple dimensions are given, results in a Series of data type
            :class:`List` with shape (rows, cols).

        See Also
        --------
        Series.list.explode : Explode a list column.

        Examples
        --------
        >>> s = pl.Series("foo", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> s.reshape((3, 3))
        shape: (3,)
        Series: \'foo\' [list[i64]]
        [
                [1, 2, 3]
                [4, 5, 6]
                [7, 8, 9]
        ]

        '''
    def shuffle(self, seed: int | None = ...) -> Series:
        '''
        Shuffle the contents of this Series.

        Parameters
        ----------
        seed
            Seed for the random number generator. If set to None (default), a random
            seed is generated using the ``random`` module.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.shuffle(seed=1)
        shape: (3,)
        Series: \'a\' [i64]
        [
                2
                1
                3
        ]

        '''
    def ewm_mean(self, com: float | None = ..., span: float | None = ..., half_life: float | None = ..., alpha: float | None = ...) -> Series:
        """
        Exponentially-weighted moving average.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\\gamma`, with

                .. math::
                    \\alpha = \\frac{1}{1 + \\gamma} \\; \\forall \\; \\gamma \\geq 0
        span
            Specify decay in terms of span, :math:`\\theta`, with

                .. math::
                    \\alpha = \\frac{2}{\\theta + 1} \\; \\forall \\; \\theta \\geq 1
        half_life
            Specify decay in terms of half-life, :math:`\\lambda`, with

                .. math::
                    \\alpha = 1 - \\exp \\left\\{ \\frac{ -\\ln(2) }{ \\lambda } \\right\\} \\;
                    \\forall \\; \\lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \\alpha \\leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \\alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\\\\n                    y_t &= (1 - \\alpha)y_{t - 1} + \\alpha x_t
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When ``ignore_nulls=False`` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\\alpha)^2` and :math:`1` if ``adjust=True``, and
                  :math:`(1-\\alpha)^2` and :math:`\\alpha` if ``adjust=False``.

                - When ``ignore_nulls=True``, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\\alpha` and :math:`1` if ``adjust=True``,
                  and :math:`1-\\alpha` and :math:`\\alpha` if ``adjust=False``.

        """
    def ewm_std(self, com: float | None = ..., span: float | None = ..., half_life: float | None = ..., alpha: float | None = ...) -> Series:
        '''
        Exponentially-weighted moving standard deviation.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\\gamma`, with

                .. math::
                    \\alpha = \\frac{1}{1 + \\gamma} \\; \\forall \\; \\gamma \\geq 0
        span
            Specify decay in terms of span, :math:`\\theta`, with

                .. math::
                    \\alpha = \\frac{2}{\\theta + 1} \\; \\forall \\; \\theta \\geq 1
        half_life
            Specify decay in terms of half-life, :math:`\\lambda`, with

                .. math::
                    \\alpha = 1 - \\exp \\left\\{ \\frac{ -\\ln(2) }{ \\lambda } \\right\\} \\;
                    \\forall \\; \\lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \\alpha \\leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \\alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\\\\n                    y_t &= (1 - \\alpha)y_{t - 1} + \\alpha x_t
        bias
            When ``bias=False``, apply a correction to make the estimate statistically
            unbiased.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When ``ignore_nulls=False`` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\\alpha)^2` and :math:`1` if ``adjust=True``, and
                  :math:`(1-\\alpha)^2` and :math:`\\alpha` if ``adjust=False``.

                - When ``ignore_nulls=True``, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\\alpha` and :math:`1` if ``adjust=True``,
                  and :math:`1-\\alpha` and :math:`\\alpha` if ``adjust=False``.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.ewm_std(com=1)
        shape: (3,)
        Series: \'a\' [f64]
        [
            0.0
            0.707107
            0.963624
        ]

        '''
    def ewm_var(self, com: float | None = ..., span: float | None = ..., half_life: float | None = ..., alpha: float | None = ...) -> Series:
        '''
        Exponentially-weighted moving variance.

        Parameters
        ----------
        com
            Specify decay in terms of center of mass, :math:`\\gamma`, with

                .. math::
                    \\alpha = \\frac{1}{1 + \\gamma} \\; \\forall \\; \\gamma \\geq 0
        span
            Specify decay in terms of span, :math:`\\theta`, with

                .. math::
                    \\alpha = \\frac{2}{\\theta + 1} \\; \\forall \\; \\theta \\geq 1
        half_life
            Specify decay in terms of half-life, :math:`\\lambda`, with

                .. math::
                    \\alpha = 1 - \\exp \\left\\{ \\frac{ -\\ln(2) }{ \\lambda } \\right\\} \\;
                    \\forall \\; \\lambda > 0
        alpha
            Specify smoothing factor alpha directly, :math:`0 < \\alpha \\leq 1`.
        adjust
            Divide by decaying adjustment factor in beginning periods to account for
            imbalance in relative weightings

                - When ``adjust=True`` the EW function is calculated
                  using weights :math:`w_i = (1 - \\alpha)^i`
                - When ``adjust=False`` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\\\\n                    y_t &= (1 - \\alpha)y_{t - 1} + \\alpha x_t
        bias
            When ``bias=False``, apply a correction to make the estimate statistically
            unbiased.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When ``ignore_nulls=False`` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\\alpha)^2` and :math:`1` if ``adjust=True``, and
                  :math:`(1-\\alpha)^2` and :math:`\\alpha` if ``adjust=False``.

                - When ``ignore_nulls=True``, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\\alpha` and :math:`1` if ``adjust=True``,
                  and :math:`1-\\alpha` and :math:`\\alpha` if ``adjust=False``.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.ewm_var(com=1)
        shape: (3,)
        Series: \'a\' [f64]
        [
            0.0
            0.5
            0.928571
        ]

        '''
    def extend_constant(self, value: PythonLiteral | None, n: int) -> Series:
        """
        Extremely fast method for extending the Series with 'n' copies of a value.

        Parameters
        ----------
        value
            A constant literal value (not an expression) with which to extend
            the Series; can pass None to extend with nulls.
        n
            The number of additional values that will be added.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.extend_constant(99, n=2)
        shape: (5,)
        Series: '' [i64]
        [
                1
                2
                3
                99
                99
        ]

        """
    def set_sorted(self) -> Self:
        '''
        Flags the Series as \'sorted\'.

        Enables downstream code to user fast paths for sorted arrays.

        Parameters
        ----------
        descending
            If the `Series` order is descending.

        Warnings
        --------
        This can lead to incorrect results if this `Series` is not sorted!!
        Use with care!

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.set_sorted().max()
        3

        '''
    def new_from_index(self, index: int, length: int) -> Self:
        """Create a new Series filled with values from the given index."""
    def shrink_dtype(self) -> Series:
        """
        Shrink numeric columns to the minimal required datatype.

        Shrink to the dtype needed to fit the extrema of this [`Series`].
        This can be used to reduce memory pressure.
        """
    def get_chunks(self) -> list[Series]:
        """Get the chunks of this Series as a list of Series."""
    def implode(self) -> Self:
        """Aggregate values into a list."""
    @property
    def dtype(self): ...
    @property
    def flags(self): ...
    @property
    def inner_dtype(self): ...
    @property
    def name(self): ...
    @property
    def shape(self): ...
    @property
    def time_unit(self): ...
    @property
    def bin(self): ...
    @property
    def cat(self): ...
    @property
    def dt(self): ...
    @property
    def list(self): ...
    @property
    def arr(self): ...
    @property
    def str(self): ...
    @property
    def struct(self): ...
def _resolve_datetime_dtype(dtype: PolarsDataType | None, ndtype: np.datetime64) -> PolarsDataType | None:
    """Given polars/numpy datetime dtypes, resolve to an explicit unit."""

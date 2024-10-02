#: version 1.3.0
import jax
import np as np
import npt
import pa as pa
import pd as pd
import torch
from polars.polars import PySeries
from datetime import date, datetime, timedelta
from polars._utils.construction.series import (
    arrow_to_pyseries as arrow_to_pyseries,
    dataframe_to_pyseries as dataframe_to_pyseries,
    iterable_to_pyseries as iterable_to_pyseries,
    numpy_to_pyseries as numpy_to_pyseries,
    pandas_to_pyseries as pandas_to_pyseries,
    sequence_to_pyseries as sequence_to_pyseries,
    series_to_pyseries as series_to_pyseries,
)
from polars._utils.convert import (
    date_to_int as date_to_int,
    datetime_to_int as datetime_to_int,
    time_to_int as time_to_int,
    timedelta_to_int as timedelta_to_int,
)
from polars._utils.deprecation import (
    deprecate_function as deprecate_function,
    deprecate_renamed_parameter as deprecate_renamed_parameter,
    issue_deprecation_warning as issue_deprecation_warning,
)
from polars._utils.getitem import get_series_item_by_key as get_series_item_by_key
from polars._utils.unstable import unstable as unstable
from polars._utils.various import (
    _is_generator as _is_generator,
    no_default as no_default,
    parse_version as parse_version,
    scale_bytes as scale_bytes,
    sphinx_accessor as sphinx_accessor,
    warn_null_comparison as warn_null_comparison,
)
from polars._utils.wrap import wrap_df as wrap_df
from polars.datatypes._parse import parse_into_dtype as parse_into_dtype
from polars.datatypes._utils import dtype_to_init_repr as dtype_to_init_repr
from polars.datatypes.classes import (
    Array as Array,
    Boolean as Boolean,
    Categorical as Categorical,
    Date as Date,
    Datetime as Datetime,
    Decimal as Decimal,
    Duration as Duration,
    Enum as Enum,
    Float32 as Float32,
    Float64 as Float64,
    Int32 as Int32,
    Int64 as Int64,
    List as List,
    Null as Null,
    Object as Object,
    String as String,
    Time as Time,
    UInt16 as UInt16,
    UInt32 as UInt32,
    UInt64 as UInt64,
    Unknown as Unknown,
)
from polars.datatypes.convert import (
    is_polars_dtype as is_polars_dtype,
    maybe_cast as maybe_cast,
    numpy_char_code_to_dtype as numpy_char_code_to_dtype,
    supported_numpy_char_code as supported_numpy_char_code,
)
from polars.dependencies import (
    _check_for_numpy as _check_for_numpy,
    _check_for_pandas as _check_for_pandas,
    _check_for_pyarrow as _check_for_pyarrow,
    hvplot as hvplot,
    import_optional as import_optional,
)
from polars.exceptions import (
    ComputeError as ComputeError,
    ModuleUpgradeRequiredError as ModuleUpgradeRequiredError,
    ShapeError as ShapeError,
)
from polars.interchange.protocol import CompatLevel as CompatLevel
from polars.series.array import ArrayNameSpace as ArrayNameSpace
from polars.series.binary import BinaryNameSpace as BinaryNameSpace
from polars.series.categorical import CatNameSpace as CatNameSpace
from polars.series.datetime import DateTimeNameSpace as DateTimeNameSpace
from polars.series.list import ListNameSpace as ListNameSpace
from polars.series.string import StringNameSpace as StringNameSpace
from polars.series.struct import StructNameSpace as StructNameSpace
from polars.series.utils import expr_dispatch as expr_dispatch, get_ffi_func as get_ffi_func
from typing import (
    Any,
    ArrayLike,
    Callable,
    ClassVar as _ClassVar,
    Collection,
    Generator,
    Iterable,
    Mapping,
    NoReturn,
    Sequence,
)

TYPE_CHECKING: bool
BUILDING_SPHINX_DOCS: None
_HVPLOT_AVAILABLE: bool
_PYARROW_AVAILABLE: bool

class Series:
    _s: _ClassVar[None] = ...
    _accessors: _ClassVar[set[str]] = ...
    def __init__(
        self,
        name: str | ArrayLike | None = ...,
        values: ArrayLike | None = ...,
        dtype: PolarsDataType | None = ...,
    ) -> None: ...
    @classmethod
    def _from_pyseries(cls, pyseries: PySeries) -> Self: ...
    @classmethod
    def _import_from_c(cls, name: str, pointers: list[tuple[int, int]]) -> Self: ...
    @classmethod
    def _import_arrow_from_c(cls, name: str, pointers: list[tuple[int, int]]) -> Self:
        """
        Construct a Series from Arrows C interface.

        Parameters
        ----------
        name
            The name that should be given to the `Series`.
        pointers
            A list with tuples containing two entries:
             - The raw pointer to a C ArrowArray struct
             - The raw pointer to a C ArrowSchema struct

        Warning
        -------
        This will read the `array` pointer without moving it. The host process should
        garbage collect the heap pointer, but not its contents.
        """
    def _export_arrow_to_c(self, out_ptr: int, out_schema_ptr: int) -> None:
        """
        Export to a C ArrowArray and C ArrowSchema struct, given their pointers.

        Parameters
        ----------
        out_ptr: int
            The raw pointer to a C ArrowArray struct.
        out_schema_ptr: int (optional)
            The raw pointer to a C ArrowSchema struct.

        Notes
        -----
        The series should only contain a single chunk. If you want to export all chunks,
        first call `Series.get_chunks` to give you a list of chunks.

        Warning
        -------
        Safety
        This function will write to the pointers given in `out_ptr` and `out_schema_ptr`
        and thus is highly unsafe.

        Leaking
        If you don't pass the ArrowArray struct to a consumer,
        array memory will leak.  This is a low-level function intended for
        expert users.
        """
    def _get_buffer_info(self) -> BufferInfo:
        """
        Return pointer, offset, and length information about the underlying buffer.

        Returns
        -------
        tuple of ints
            Tuple of the form (pointer, offset, length)

        Raises
        ------
        TypeError
            If the `Series` data type is not physical.
        ComputeError
            If the `Series` contains multiple chunks.

        Notes
        -----
        This method is mainly intended for use with the dataframe interchange protocol.
        """
    def _get_buffers(self) -> SeriesBuffers:
        """
        Return the underlying values, validity, and offsets buffers as Series.

        The values buffer always exists.
        The validity buffer may not exist if the column contains no null values.
        The offsets buffer only exists for Series of data type `String` and `List`.

        Returns
        -------
        dict
            Dictionary with `"values"`, `"validity"`, and `"offsets"` keys mapping
            to the corresponding buffer or `None` if the buffer doesn\'t exist.

        Warnings
        --------
        The underlying buffers for `String` Series cannot be represented in this
        format. Instead, the buffers are converted to a values and offsets buffer.

        Notes
        -----
        This method is mainly intended for use with the dataframe interchange protocol.
        """
    @classmethod
    def _from_buffer(cls, dtype: PolarsDataType, buffer_info: BufferInfo, owner: Any) -> Self:
        """
        Construct a Series from information about its underlying buffer.

        Parameters
        ----------
        dtype
            The data type of the buffer.
            Must be a physical type (integer, float, or boolean).
        buffer_info
            Tuple containing buffer information in the form `(pointer, offset, length)`.
        owner
            The object owning the buffer.

        Returns
        -------
        Series

        Raises
        ------
        TypeError
            When the given `dtype` is not supported.

        Notes
        -----
        This method is mainly intended for use with the dataframe interchange protocol.
        """
    @classmethod
    def _from_buffers(
        cls, dtype: PolarsDataType, data: Series | Sequence[Series], validity: Series | None = ...
    ) -> Self:
        """
        Construct a Series from information about its underlying buffers.

        Parameters
        ----------
        dtype
            The data type of the resulting Series.
        data
            Buffers describing the data. For most data types, this is a single Series of
            the physical data type of `dtype`. Some data types require multiple buffers:

            - `String`: A data buffer of type `UInt8` and an offsets buffer
              of type `Int64`. Note that this does not match how the data
              is represented internally and data copy is required to construct
              the Series.
        validity
            Validity buffer. If specified, must be a Series of data type `Boolean`.

        Returns
        -------
        Series

        Raises
        ------
        TypeError
            When the given `dtype` is not supported or the other inputs do not match
            the requirements for constructing a Series of the given `dtype`.

        Warnings
        --------
        Constructing a `String` Series requires specifying a values and offsets buffer,
        which does not match the actual underlying buffers. The values and offsets
        buffer are converted into the actual buffers, which copies data.

        Notes
        -----
        This method is mainly intended for use with the dataframe interchange protocol.
        """
    @staticmethod
    def _newest_compat_level() -> int:
        """
        Get the newest supported compat level.

        This is for pyo3-polars.
        """
    def __bool__(self) -> NoReturn: ...
    def __len__(self) -> int: ...
    def __and__(self, other: Any) -> Self: ...
    def __rand__(self, other: Any) -> Series: ...
    def __or__(self, other: Any) -> Self: ...
    def __ror__(self, other: Any) -> Series: ...
    def __xor__(self, other: Any) -> Self: ...
    def __rxor__(self, other: Any) -> Series: ...
    def _comp(self, other: Any, op: ComparisonOperator) -> Series: ...
    def __eq__(self, other: Any) -> Series | Expr: ...
    def __ne__(self, other: Any) -> Series | Expr: ...
    def __gt__(self, other: Any) -> Series | Expr: ...
    def __lt__(self, other: Any) -> Series | Expr: ...
    def __ge__(self, other: Any) -> Series | Expr: ...
    def __le__(self, other: Any) -> Series | Expr: ...
    def le(self, other: Any) -> Series | Expr:
        """Method equivalent of operator expression `series <= other`."""
    def lt(self, other: Any) -> Series | Expr:
        """Method equivalent of operator expression `series < other`."""
    def eq(self, other: Any) -> Series | Expr:
        """Method equivalent of operator expression `series == other`."""
    def eq_missing(self, other: Any) -> Series | Expr:
        """
        Method equivalent of equality operator `series == other` where `None == None`.

        This differs from the standard `ne` where null values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        See Also
        --------
        ne_missing
        eq

        Examples
        --------
        >>> s1 = pl.Series("a", [333, 200, None])
        >>> s2 = pl.Series("a", [100, 200, None])
        >>> s1.eq(s2)
        shape: (3,)
        Series: \'a\' [bool]
        [
            false
            true
            null
        ]
        >>> s1.eq_missing(s2)
        shape: (3,)
        Series: \'a\' [bool]
        [
            false
            true
            true
        ]
        """
    def ne(self, other: Any) -> Series | Expr:
        """Method equivalent of operator expression `series != other`."""
    def ne_missing(self, other: Any) -> Series | Expr:
        """
        Method equivalent of equality operator `series != other` where `None == None`.

        This differs from the standard `ne` where null values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        See Also
        --------
        eq_missing
        ne

        Examples
        --------
        >>> s1 = pl.Series("a", [333, 200, None])
        >>> s2 = pl.Series("a", [100, 200, None])
        >>> s1.ne(s2)
        shape: (3,)
        Series: \'a\' [bool]
        [
            true
            false
            null
        ]
        >>> s1.ne_missing(s2)
        shape: (3,)
        Series: \'a\' [bool]
        [
            true
            false
            false
        ]
        """
    def ge(self, other: Any) -> Series | Expr:
        """Method equivalent of operator expression `series >= other`."""
    def gt(self, other: Any) -> Series | Expr:
        """Method equivalent of operator expression `series > other`."""
    def _arithmetic(self, other: Any, op_s: str, op_ffi: str) -> Self: ...
    def __add__(self, other: Any) -> Self | DataFrame | Expr: ...
    def __sub__(self, other: Any) -> Self | Expr: ...
    def __truediv__(self, other: Any) -> Series | Expr: ...
    def __floordiv__(self, other: Any) -> Series | Expr: ...
    def __invert__(self) -> Series: ...
    def __mul__(self, other: Any) -> Series | DataFrame | Expr: ...
    def __mod__(self, other: Any) -> Series | Expr: ...
    def __rmod__(self, other: Any) -> Series: ...
    def __radd__(self, other: Any) -> Series: ...
    def __rsub__(self, other: Any) -> Series: ...
    def __rtruediv__(self, other: Any) -> Series: ...
    def __rfloordiv__(self, other: Any) -> Series: ...
    def __rmul__(self, other: Any) -> Series: ...
    def __pow__(self, exponent: int | float | Series) -> Series: ...
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
    def __getitem__(self, key: SingleIndexSelector | MultiIndexSelector) -> Any | Series:
        """Get part of the Series as a new Series or scalar."""
    def __setitem__(
        self,
        key: int | Series | np.ndarray[Any, Any] | Sequence[object] | tuple[object],
        value: Any,
    ) -> None: ...
    def __array__(
        self, dtype: npt.DTypeLike | None = ..., copy: bool | None = ...
    ) -> np.ndarray[Any, Any]:
        """
        Return a NumPy ndarray with the given data type.

        This method ensures a Polars Series can be treated as a NumPy ndarray.
        It enables `np.asarray` and NumPy universal functions.

        See the NumPy documentation for more information:
        https://numpy.org/doc/stable/user/basics.interoperability.html#the-array-method

        See Also
        --------
        __array_ufunc__
        """
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Series:
        """Numpy universal functions."""
    def __arrow_c_stream__(self, requested_schema: object) -> object:
        """
        Export a Series via the Arrow PyCapsule Interface.

        https://arrow.apache.org/docs/dev/format/CDataInterface/PyCapsuleInterface.html
        """
    def _repr_html_(self) -> str:
        """Format output data in HTML for display in Jupyter Notebooks."""
    def item(self, index: int | None = ...) -> Any:
        """
        Return the Series as a scalar, or return the element at the given index.

        If no index is provided, this is equivalent to `s[0]`, with a check
        that the shape is (1,). With an index, this is equivalent to `s[index]`.

        Examples
        --------
        >>> s1 = pl.Series("a", [1])
        >>> s1.item()
        1
        >>> s2 = pl.Series("a", [9, 8, 7])
        >>> s2.cum_sum().item(-1)
        24
        """
    def estimated_size(self, unit: SizeUnit = ...) -> int | float:
        """
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
        """
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

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.sqrt()
        shape: (3,)
        Series: '' [f64]
        [
            1.0
            1.414214
            1.732051
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

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.cbrt()
        shape: (3,)
        Series: '' [f64]
        [
            1.0
            1.259921
            1.44225
        ]
        """
    def any(self) -> bool | None:
        """
        Return whether any of the values in the column are `True`.

        Only works on columns of data type :class:`Boolean`.

        Parameters
        ----------
        ignore_nulls
            Ignore null values (default).

            If set to `False`, `Kleene logic`_ is used to deal with nulls:
            if the column contains any null values and no `True` values,
            the output is `None`.

            .. _Kleene logic: https://en.wikipedia.org/wiki/Three-valued_logic

        Returns
        -------
        bool or None

        Examples
        --------
        >>> pl.Series([True, False]).any()
        True
        >>> pl.Series([False, False]).any()
        False
        >>> pl.Series([None, False]).any()
        False

        Enable Kleene logic by setting `ignore_nulls=False`.

        >>> pl.Series([None, False]).any(ignore_nulls=False)  # Returns None
        """
    def all(self) -> bool | None:
        """
        Return whether all values in the column are `True`.

        Only works on columns of data type :class:`Boolean`.

        Parameters
        ----------
        ignore_nulls
            Ignore null values (default).

            If set to `False`, `Kleene logic`_ is used to deal with nulls:
            if the column contains any null values and no `False` values,
            the output is `None`.

            .. _Kleene logic: https://en.wikipedia.org/wiki/Three-valued_logic

        Returns
        -------
        bool or None

        Examples
        --------
        >>> pl.Series([True, True]).all()
        True
        >>> pl.Series([False, True]).all()
        False
        >>> pl.Series([None, True]).all()
        True

        Enable Kleene logic by setting `ignore_nulls=False`.

        >>> pl.Series([None, True]).all(ignore_nulls=False)  # Returns None
        """
    def log(self, base: float = ...) -> Series:
        """
        Compute the logarithm to a given base.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.log()
        shape: (3,)
        Series: '' [f64]
        [
            0.0
            0.693147
            1.098612
        ]
        """
    def log1p(self) -> Series:
        """
        Compute the natural logarithm of the input array plus one, element-wise.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.log1p()
        shape: (3,)
        Series: '' [f64]
        [
            0.693147
            1.098612
            1.386294
        ]
        """
    def log10(self) -> Series:
        """
        Compute the base 10 logarithm of the input array, element-wise.

        Examples
        --------
        >>> s = pl.Series([10, 100, 1000])
        >>> s.log10()
        shape: (3,)
        Series: '' [f64]
        [
            1.0
            2.0
            3.0
        ]
        """
    def exp(self) -> Series:
        """
        Compute the exponential, element-wise.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.exp()
        shape: (3,)
        Series: '' [f64]
        [
            2.718282
            7.389056
            20.085537
        ]
        """
    def drop_nulls(self) -> Series:
        """
        Drop all null values.

        The original order of the remaining elements is preserved.

        See Also
        --------
        drop_nans

        Notes
        -----
        A null value is not the same as a NaN value.
        To drop NaN values, use :func:`drop_nans`.

        Examples
        --------
        >>> s = pl.Series([1.0, None, 3.0, float("nan")])
        >>> s.drop_nulls()
        shape: (3,)
        Series: \'\' [f64]
        [
                1.0
                3.0
                NaN
        ]
        """
    def drop_nans(self) -> Series:
        """
        Drop all floating point NaN values.

        The original order of the remaining elements is preserved.

        See Also
        --------
        drop_nulls

        Notes
        -----
        A NaN value is not the same as a null value.
        To drop null values, use :func:`drop_nulls`.

        Examples
        --------
        >>> s = pl.Series([1.0, None, 3.0, float("nan")])
        >>> s.drop_nans()
        shape: (3,)
        Series: \'\' [f64]
        [
                1.0
                null
                3.0
        ]
        """
    def to_frame(self, name: str | None = ...) -> DataFrame:
        """
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
        """
    def describe(
        self,
        percentiles: Sequence[float] | float | None = ...,
        interpolation: RollingInterpolationMethod = ...,
    ) -> DataFrame:
        """
        Quick summary statistics of a Series.

        Series with mixed datatypes will return summary statistics for the datatype of
        the first value.

        Parameters
        ----------
        percentiles
            One or more percentiles to include in the summary statistics (if the
            Series has a numeric dtype). All values must be in the range `[0, 1]`.
        interpolation : {\'nearest\', \'higher\', \'lower\', \'midpoint\', \'linear\'}
            Interpolation method used when calculating percentiles.

        Notes
        -----
        The median is included by default as the 50% percentile.

        Returns
        -------
        DataFrame
            Mapping with summary statistics of a Series.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3, 4, 5])
        >>> s.describe()
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
        │ 25%        ┆ 2.0      │
        │ 50%        ┆ 3.0      │
        │ 75%        ┆ 4.0      │
        │ max        ┆ 5.0      │
        └────────────┴──────────┘

        Non-numeric data types may not have all statistics available.

        >>> s = pl.Series(["aa", "aa", None, "bb", "cc"])
        >>> s.describe()
        shape: (4, 2)
        ┌────────────┬───────┐
        │ statistic  ┆ value │
        │ ---        ┆ ---   │
        │ str        ┆ str   │
        ╞════════════╪═══════╡
        │ count      ┆ 4     │
        │ null_count ┆ 1     │
        │ min        ┆ aa    │
        │ max        ┆ cc    │
        └────────────┴───────┘
        """
    def sum(self) -> int | float:
        """
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
        """
    def mean(self) -> PythonLiteral | None:
        """
        Reduce this Series to the mean value.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.mean()
        2.0
        """
    def product(self) -> int | float:
        """
        Reduce this Series to the product value.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.product()
        6
        """
    def pow(self, exponent: int | float | Series) -> Series:
        """
        Raise to the power of the given exponent.

        If the exponent is float, the result follows the dtype of exponent.
        Otherwise, it follows dtype of base.

        Parameters
        ----------
        exponent
            The exponent. Accepts Series input.

        Examples
        --------
        Raising integers to positive integers results in integers:

        >>> s = pl.Series("foo", [1, 2, 3, 4])
        >>> s.pow(3)
        shape: (4,)
        Series: \'foo\' [i64]
        [
            1
            8
            27
            64
        ]

        In order to raise integers to negative integers, you can cast either the
        base or the exponent to float:

        >>> s.pow(-3.0)
        shape: (4,)
        Series: \'foo\' [f64]
        [
                1.0
                0.125
                0.037037
                0.015625
        ]
        """
    def min(self) -> PythonLiteral | None:
        """
        Get the minimal value in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.min()
        1
        """
    def max(self) -> PythonLiteral | None:
        """
        Get the maximum value in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.max()
        3
        """
    def nan_max(self) -> int | float | date | datetime | timedelta | str:
        """
        Get maximum value, but propagate/poison encountered NaN values.

        This differs from numpy\'s `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        Examples
        --------
        >>> s = pl.Series("a", [1, 3, 4])
        >>> s.nan_max()
        4

        >>> s = pl.Series("a", [1.0, float("nan"), 4.0])
        >>> s.nan_max()
        nan
        """
    def nan_min(self) -> int | float | date | datetime | timedelta | str:
        """
        Get minimum value, but propagate/poison encountered NaN values.

        This differs from numpy\'s `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        Examples
        --------
        >>> s = pl.Series("a", [1, 3, 4])
        >>> s.nan_min()
        1

        >>> s = pl.Series("a", [1.0, float("nan"), 4.0])
        >>> s.nan_min()
        nan
        """
    def std(self, ddof: int = ...) -> float | timedelta | None:
        """
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
        """
    def var(self, ddof: int = ...) -> float | timedelta | None:
        """
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
        """
    def median(self) -> PythonLiteral | None:
        """
        Get the median of this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.median()
        2.0
        """
    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = ...
    ) -> float | None:
        """
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
        """
    def to_dummies(self) -> DataFrame:
        """
        Get dummy/indicator variables.

        Parameters
        ----------
        separator
            Separator/delimiter used when generating column names.
        drop_first
            Remove the first category from the variable being encoded.

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

        >>> s.to_dummies(drop_first=True)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a_2 ┆ a_3 │
        │ --- ┆ --- │
        │ u8  ┆ u8  │
        ╞═════╪═════╡
        │ 0   ┆ 0   │
        │ 1   ┆ 0   │
        │ 0   ┆ 1   │
        └─────┴─────┘
        """
    def cut(self, breaks: Sequence[float]) -> Series:
        """
        Bin continuous values into discrete categories.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        breaks
            List of unique cut points.
        labels
            Names of the categories. The number of labels must be equal to the number
            of cut points plus one.
        left_closed
            Set the intervals to be left-closed instead of right-closed.
        include_breaks
            Include a column with the right endpoint of the bin each observation falls
            in. This will change the data type of the output from a
            :class:`Categorical` to a :class:`Struct`.

        Returns
        -------
        Series
            Series of data type :class:`Categorical` if `include_breaks` is set to
            `False` (default), otherwise a Series of data type :class:`Struct`.

        See Also
        --------
        qcut

        Examples
        --------
        Divide the column into three categories.

        >>> s = pl.Series("foo", [-2, -1, 0, 1, 2])
        >>> s.cut([-1, 1], labels=["a", "b", "c"])
        shape: (5,)
        Series: \'foo\' [cat]
        [
                "a"
                "a"
                "b"
                "b"
                "c"
        ]

        Create a DataFrame with the breakpoint and category for each value.

        >>> cut = s.cut([-1, 1], include_breaks=True).alias("cut")
        >>> s.to_frame().with_columns(cut).unnest("cut")
        shape: (5, 3)
        ┌─────┬────────────┬────────────┐
        │ foo ┆ breakpoint ┆ category   │
        │ --- ┆ ---        ┆ ---        │
        │ i64 ┆ f64        ┆ cat        │
        ╞═════╪════════════╪════════════╡
        │ -2  ┆ -1.0       ┆ (-inf, -1] │
        │ -1  ┆ -1.0       ┆ (-inf, -1] │
        │ 0   ┆ 1.0        ┆ (-1, 1]    │
        │ 1   ┆ 1.0        ┆ (-1, 1]    │
        │ 2   ┆ inf        ┆ (1, inf]   │
        └─────┴────────────┴────────────┘
        """
    def qcut(self, quantiles: Sequence[float] | int) -> Series:
        """
        Bin continuous values into discrete categories based on their quantiles.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        quantiles
            Either a list of quantile probabilities between 0 and 1 or a positive
            integer determining the number of bins with uniform probability.
        labels
            Names of the categories. The number of labels must be equal to the number
            of cut points plus one.
        left_closed
            Set the intervals to be left-closed instead of right-closed.
        allow_duplicates
            If set to `True`, duplicates in the resulting quantiles are dropped,
            rather than raising a `DuplicateError`. This can happen even with unique
            probabilities, depending on the data.
        include_breaks
            Include a column with the right endpoint of the bin each observation falls
            in. This will change the data type of the output from a
            :class:`Categorical` to a :class:`Struct`.

        Returns
        -------
        Series
            Series of data type :class:`Categorical` if `include_breaks` is set to
            `False` (default), otherwise a Series of data type :class:`Struct`.

        See Also
        --------
        cut

        Examples
        --------
        Divide a column into three categories according to pre-defined quantile
        probabilities.

        >>> s = pl.Series("foo", [-2, -1, 0, 1, 2])
        >>> s.qcut([0.25, 0.75], labels=["a", "b", "c"])
        shape: (5,)
        Series: \'foo\' [cat]
        [
                "a"
                "a"
                "b"
                "b"
                "c"
        ]

        Divide a column into two categories using uniform quantile probabilities.

        >>> s.qcut(2, labels=["low", "high"], left_closed=True)
        shape: (5,)
        Series: \'foo\' [cat]
        [
                "low"
                "low"
                "high"
                "high"
                "high"
        ]

        Create a DataFrame with the breakpoint and category for each value.

        >>> cut = s.qcut([0.25, 0.75], include_breaks=True).alias("cut")
        >>> s.to_frame().with_columns(cut).unnest("cut")
        shape: (5, 3)
        ┌─────┬────────────┬────────────┐
        │ foo ┆ breakpoint ┆ category   │
        │ --- ┆ ---        ┆ ---        │
        │ i64 ┆ f64        ┆ cat        │
        ╞═════╪════════════╪════════════╡
        │ -2  ┆ -1.0       ┆ (-inf, -1] │
        │ -1  ┆ -1.0       ┆ (-inf, -1] │
        │ 0   ┆ 1.0        ┆ (-1, 1]    │
        │ 1   ┆ 1.0        ┆ (-1, 1]    │
        │ 2   ┆ inf        ┆ (1, inf]   │
        └─────┴────────────┴────────────┘
        """
    def rle(self) -> Series:
        """
        Compress the Series data using run-length encoding.

        Run-length encoding (RLE) encodes data by storing each *run* of identical values
        as a single value and its length.

        Returns
        -------
        Series
            Series of data type `Struct` with fields `len` of data type `UInt32`
            and `value` of the original data type.

        Examples
        --------
        >>> s = pl.Series("s", [1, 1, 2, 1, None, 1, 3, 3])
        >>> s.rle().struct.unnest()
        shape: (6, 2)
        ┌─────┬───────┐
        │ len ┆ value │
        │ --- ┆ ---   │
        │ u32 ┆ i64   │
        ╞═════╪═══════╡
        │ 2   ┆ 1     │
        │ 1   ┆ 2     │
        │ 1   ┆ 1     │
        │ 1   ┆ null  │
        │ 1   ┆ 1     │
        │ 2   ┆ 3     │
        └─────┴───────┘
        """
    def rle_id(self) -> Series:
        """
        Get a distinct integer ID for each run of identical values.

        The ID starts at 0 and increases by one each time the value of the column
        changes.

        Returns
        -------
        Series
            Series of data type `UInt32`.

        See Also
        --------
        rle

        Notes
        -----
        This functionality is especially useful for defining a new group for every time
        a column\'s value changes, rather than for every distinct value of that column.

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
        """
    def hist(self, bins: list[float] | None = ...) -> DataFrame:
        """
        Bin values into buckets and count their occurrences.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        bins
            Discretizations to make.
            If None given, we determine the boundaries based on the data.
        bin_count
            If no bins provided, this will be used to determine
            the distance of the bins
        include_breakpoint
            Include a column that indicates the upper breakpoint.
        include_category
            Include a column that shows the intervals as categories.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> a = pl.Series("a", [1, 3, 8, 8, 2, 1, 3])
        >>> a.hist(bin_count=4)
        shape: (5, 3)
        ┌────────────┬─────────────┬───────┐
        │ breakpoint ┆ category    ┆ count │
        │ ---        ┆ ---         ┆ ---   │
        │ f64        ┆ cat         ┆ u32   │
        ╞════════════╪═════════════╪═══════╡
        │ 0.0        ┆ (-inf, 0.0] ┆ 0     │
        │ 2.25       ┆ (0.0, 2.25] ┆ 3     │
        │ 4.5        ┆ (2.25, 4.5] ┆ 2     │
        │ 6.75       ┆ (4.5, 6.75] ┆ 0     │
        │ inf        ┆ (6.75, inf] ┆ 2     │
        └────────────┴─────────────┴───────┘
        """
    def value_counts(self) -> DataFrame:
        """
        Count the occurrences of unique values.

        Parameters
        ----------
        sort
            Sort the output by count in descending order.
            If set to `False` (default), the order of the output is random.
        parallel
            Execute the computation in parallel.

            .. note::
                This option should likely not be enabled in a group by context,
                as the computation is already parallelized per group.
        name
            Give the resulting count column a specific name;
            if `normalize` is True defaults to "proportion",
            otherwise defaults to "count".
        normalize
            If true gives relative frequencies of the unique values

        Returns
        -------
        DataFrame
            Mapping of unique values to their count.

        Examples
        --------
        >>> s = pl.Series("color", ["red", "blue", "red", "green", "blue", "blue"])
        >>> s.value_counts()  # doctest: +IGNORE_RESULT
        shape: (3, 2)
        ┌───────┬───────┐
        │ color ┆ count │
        │ ---   ┆ ---   │
        │ str   ┆ u32   │
        ╞═══════╪═══════╡
        │ red   ┆ 2     │
        │ green ┆ 1     │
        │ blue  ┆ 3     │
        └───────┴───────┘

        Sort the output by count and customize the count column name.

        >>> s.value_counts(sort=True, name="n")
        shape: (3, 2)
        ┌───────┬─────┐
        │ color ┆ n   │
        │ ---   ┆ --- │
        │ str   ┆ u32 │
        ╞═══════╪═════╡
        │ blue  ┆ 3   │
        │ red   ┆ 2   │
        │ green ┆ 1   │
        └───────┴─────┘
        """
    def unique_counts(self) -> Series:
        """
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
        """
    def entropy(self, base: float = ...) -> float | None:
        """
        Computes the entropy.

        Uses the formula `-sum(pk * log(pk)` where `pk` are discrete probabilities.

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
    def cumulative_eval(self, expr: Expr) -> Series:
        """
        Run an expression over a sliding window that increases `1` slot every iteration.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        expr
            Expression to evaluate
        min_periods
            Number of valid values there should be in the window before the expression
            is evaluated. valid values = `length - null_count`
        parallel
            Run in parallel. Don\'t do this in a group by or another operation that
            already has much parallelization.

        Warnings
        --------
        This can be really slow as it can have `O(n^2)` complexity. Don\'t use this
        for operations that visit all elements.

        Examples
        --------
        >>> s = pl.Series("values", [1, 2, 3, 4, 5])
        >>> s.cumulative_eval(pl.element().first() - pl.element().last() ** 2)
        shape: (5,)
        Series: \'values\' [i64]
        [
            0
            -3
            -8
            -15
            -24
        ]
        """
    def alias(self, name: str) -> Series:
        """
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
        """
    def rename(self, name: str) -> Series:
        """
        Rename this Series.

        Alias for :func:`Series.alias`.

        Parameters
        ----------
        name
            New name.

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
        """
    def chunk_lengths(self) -> list[int]:
        """
        Get the length of each individual chunk.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("a", [4, 5, 6])

        Concatenate Series with rechunk = True

        >>> pl.concat([s, s2], rechunk=True).chunk_lengths()
        [6]

        Concatenate Series with rechunk = False

        >>> pl.concat([s, s2], rechunk=False).chunk_lengths()
        [3, 3]
        """
    def n_chunks(self) -> int:
        """
        Get the number of chunks that this Series contains.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.n_chunks()
        1
        >>> s2 = pl.Series("a", [4, 5, 6])

        Concatenate Series with rechunk = True

        >>> pl.concat([s, s2], rechunk=True).n_chunks()
        1

        Concatenate Series with rechunk = False

        >>> pl.concat([s, s2], rechunk=False).n_chunks()
        2
        """
    def cum_max(self) -> Series:
        """
        Get an array with the cumulative max computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Examples
        --------
        >>> s = pl.Series("s", [3, 5, 1])
        >>> s.cum_max()
        shape: (3,)
        Series: \'s\' [i64]
        [
            3
            5
            5
        ]
        """
    def cum_min(self) -> Series:
        """
        Get an array with the cumulative min computed at every element.

        Parameters
        ----------
        reverse
            reverse the operation.

        Examples
        --------
        >>> s = pl.Series("s", [1, 2, 3])
        >>> s.cum_min()
        shape: (3,)
        Series: \'s\' [i64]
        [
            1
            1
            1
        ]
        """
    def cum_prod(self) -> Series:
        """
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
        >>> s.cum_prod()
        shape: (3,)
        Series: \'a\' [i64]
        [
            1
            2
            6
        ]
        """
    def cum_sum(self) -> Series:
        """
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
        >>> s.cum_sum()
        shape: (3,)
        Series: \'a\' [i64]
        [
            1
            3
            6
        ]
        """
    def cum_count(self) -> Self:
        """
        Return the cumulative count of the non-null values in the column.

        Parameters
        ----------
        reverse
            Reverse the operation.

        Examples
        --------
        >>> s = pl.Series(["x", "k", None, "d"])
        >>> s.cum_count()
        shape: (4,)
        Series: \'\' [u32]
        [
                1
                2
                2
                3
        ]
        """
    def slice(self, offset: int, length: int | None = ...) -> Series:
        """
        Get a slice of this Series.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to `None`, all rows starting at the offset
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
        """
    def append(self, other: Series) -> Self:
        """
        Append a Series to this one.

        The resulting series will consist of multiple chunks.

        Parameters
        ----------
        other
            Series to append.

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
        """
    def extend(self, other: Series) -> Self:
        """
        Extend the memory backed by this Series with the values from another.

        Different from `append`, which adds the chunks from `other` to the chunks of
        this series, `extend` appends the data from `other` to the underlying memory
        locations and thus may cause a reallocation (which is expensive).

        If this does `not` cause a reallocation, the resulting data structure will not
        have any extra chunks and thus will yield faster queries.

        Prefer `extend` over `append` when you want to do a query after a single
        append. For instance, during online operations where you add `n` rows
        and rerun a query.

        Prefer `append` over `extend` when you want to append many times
        before doing a query. For instance, when you read in multiple files and want
        to store them in a single `Series`. In the latter case, finish the sequence
        of `append` operations with a `rechunk`.

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
        """
    def filter(self, predicate: Series | list[bool]) -> Self:
        """
        Filter elements by a boolean mask.

        The original order of the remaining elements is preserved.

        Elements where the filter does not evaluate to True are discarded, including
        nulls.

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
        """
    def head(self, n: int = ...) -> Series:
        """
        Get the first `n` elements.

        Parameters
        ----------
        n
            Number of elements to return. If a negative value is passed, return all
            elements except the last `abs(n)`.

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

        Pass a negative value to get all rows `except` the last `abs(n)`.

        >>> s.head(-3)
        shape: (2,)
        Series: \'a\' [i64]
        [
                1
                2
        ]
        """
    def tail(self, n: int = ...) -> Series:
        """
        Get the last `n` elements.

        Parameters
        ----------
        n
            Number of elements to return. If a negative value is passed, return all
            elements except the first `abs(n)`.

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

        Pass a negative value to get all rows `except` the first `abs(n)`.

        >>> s.tail(-3)
        shape: (2,)
        Series: \'a\' [i64]
        [
                4
                5
        ]
        """
    def limit(self, n: int = ...) -> Series:
        """
        Get the first `n` elements.

        Alias for :func:`Series.head`.

        Parameters
        ----------
        n
            Number of elements to return. If a negative value is passed, return all
            elements except the last `abs(n)`.

        See Also
        --------
        head

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.limit(3)
        shape: (3,)
        Series: \'a\' [i64]
        [
            1
            2
            3
        ]

        Pass a negative value to get all rows `except` the last `abs(n)`.

        >>> s.limit(-3)
        shape: (2,)
        Series: \'a\' [i64]
        [
                1
                2
        ]
        """
    def gather_every(self, n: int, offset: int = ...) -> Series:
        """
        Take every nth value in the Series and return as new Series.

        Parameters
        ----------
        n
            Gather every *n*-th row.
        offset
            Start the row index at this offset.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.gather_every(2)
        shape: (2,)
        Series: \'a\' [i64]
        [
            1
            3
        ]
        >>> s.gather_every(2, offset=1)
        shape: (2,)
        Series: \'a\' [i64]
        [
            2
            4
        ]
        """
    def sort(self) -> Self:
        """
        Sort this Series.

        Parameters
        ----------
        descending
            Sort in descending order.
        nulls_last
            Place null values last instead of first.
        multithreaded
            Sort using multiple threads.
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
        """
    def top_k(self, k: int = ...) -> Series:
        """
        Return the `k` largest elements.

        Non-null elements are always preferred over null elements. The output is
        not guaranteed to be in any particular order, call :func:`sort` after
        this function if you wish the output to be sorted.

        This has time complexity:

        .. math:: O(n)

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
        """
    def bottom_k(self, k: int = ...) -> Series:
        """
        Return the `k` smallest elements.

        Non-null elements are always preferred over null elements. The output is
        not guaranteed to be in any particular order, call :func:`sort` after
        this function if you wish the output to be sorted. This has time
        complexity:

        This has time complexity:

        .. math:: O(n)

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
        """
    def arg_sort(self) -> Series:
        """
        Get the index values that would sort this Series.

        Parameters
        ----------
        descending
            Sort in descending order.
        nulls_last
            Place null values last instead of first.

        See Also
        --------
        Series.gather: Take values by index.
        Series.rank : Get the rank of each row.

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
        """
    def arg_unique(self) -> Series:
        """
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
        """
    def arg_min(self) -> int | None:
        """
        Get the index of the minimal value.

        Returns
        -------
        int

        Examples
        --------
        >>> s = pl.Series("a", [3, 2, 1])
        >>> s.arg_min()
        2
        """
    def arg_max(self) -> int | None:
        """
        Get the index of the maximal value.

        Returns
        -------
        int

        Examples
        --------
        >>> s = pl.Series("a", [3, 2, 1])
        >>> s.arg_max()
        0
        """
    def search_sorted(
        self, element: IntoExpr | np.ndarray[Any, Any] | None, side: SearchSortedSide = ...
    ) -> int | Series:
        """
        Find indices where elements should be inserted to maintain order.

        .. math:: a[i-1] < v <= a[i]

        Parameters
        ----------
        element
            Expression or scalar value.
        side : {\'any\', \'left\', \'right\'}
            If \'any\', the index of the first suitable location found is given.
            If \'left\', the index of the leftmost suitable location found is given.
            If \'right\', return the rightmost suitable location found is given.

        Examples
        --------
        >>> s = pl.Series("set", [1, 2, 3, 4, 4, 5, 6, 7])
        >>> s.search_sorted(4)
        3
        >>> s.search_sorted(4, "left")
        3
        >>> s.search_sorted(4, "right")
        5
        >>> s.search_sorted([1, 4, 5])
        shape: (3,)
        Series: \'set\' [u32]
        [
                0
                3
                5
        ]
        >>> s.search_sorted([1, 4, 5], "left")
        shape: (3,)
        Series: \'set\' [u32]
        [
                0
                3
                5
        ]
        >>> s.search_sorted([1, 4, 5], "right")
        shape: (3,)
        Series: \'set\' [u32]
        [
                1
                5
                6
        ]
        """
    def unique(self) -> Series:
        """
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
        """
    def gather(self, indices: int | list[int] | Expr | Series | np.ndarray[Any, Any]) -> Series:
        """
        Take values by index.

        Parameters
        ----------
        indices
            Index location used for selection.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4])
        >>> s.gather([1, 3])
        shape: (2,)
        Series: \'a\' [i64]
        [
                2
                4
        ]
        """
    def null_count(self) -> int:
        """
        Count the null values in this Series.

        Examples
        --------
        >>> s = pl.Series([1, None, None])
        >>> s.null_count()
        2
        """
    def has_nulls(self) -> bool:
        """
        Check whether the Series contains one or more null values.

        Examples
        --------
        >>> s = pl.Series([1, 2, None])
        >>> s.has_nulls()
        True
        >>> s[:2].has_nulls()
        False
        """
    def has_validity(self) -> bool:
        """
        Check whether the Series contains one or more null values.

        .. deprecated:: 0.20.30
            Use :meth:`has_nulls` instead.
        """
    def is_empty(self) -> bool:
        """
        Check if the Series is empty.

        Examples
        --------
        >>> s = pl.Series("a", [], dtype=pl.Float32)
        >>> s.is_empty()
        True
        """
    def is_sorted(self) -> bool:
        """
        Check if the Series is sorted.

        Parameters
        ----------
        descending
            Check if the Series is sorted in descending order
        nulls_last
            Set nulls at the end of the Series in sorted check.

        Examples
        --------
        >>> s = pl.Series([1, 3, 2])
        >>> s.is_sorted()
        False

        >>> s = pl.Series([3, 2, 1])
        >>> s.is_sorted(descending=True)
        True
        """
    def not_(self) -> Series:
        """
        Negate a boolean Series.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series("a", [True, False, False])
        >>> s.not_()
        shape: (3,)
        Series: \'a\' [bool]
        [
            false
            true
            true
        ]
        """
    def is_null(self) -> Series:
        """
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
        """
    def is_not_null(self) -> Series:
        """
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
        """
    def is_finite(self) -> Series:
        """
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
        """
    def is_infinite(self) -> Series:
        """
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
        """
    def is_nan(self) -> Series:
        """
        Returns a boolean Series indicating which values are not NaN.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.nan])
        >>> s.is_nan()
        shape: (4,)
        Series: \'a\' [bool]
        [
                false
                false
                false
                true
        ]
        """
    def is_not_nan(self) -> Series:
        """
        Returns a boolean Series indicating which values are not NaN.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> import numpy as np
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, np.nan])
        >>> s.is_not_nan()
        shape: (4,)
        Series: \'a\' [bool]
        [
                true
                true
                true
                false
        ]
        """
    def is_in(self, other: Series | Collection[Any]) -> Series:
        """
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
        """
    def arg_true(self) -> Series:
        """
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
        """
    def is_unique(self) -> Series:
        """
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
        """
    def is_first_distinct(self) -> Series:
        """
        Return a boolean mask indicating the first occurrence of each distinct value.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series([1, 1, 2, 3, 2])
        >>> s.is_first_distinct()
        shape: (5,)
        Series: '' [bool]
        [
                true
                false
                true
                true
                false
        ]
        """
    def is_last_distinct(self) -> Series:
        """
        Return a boolean mask indicating the last occurrence of each distinct value.

        Returns
        -------
        Series
            Series of data type :class:`Boolean`.

        Examples
        --------
        >>> s = pl.Series([1, 1, 2, 3, 2])
        >>> s.is_last_distinct()
        shape: (5,)
        Series: '' [bool]
        [
                false
                true
                false
                true
                true
        ]
        """
    def is_duplicated(self) -> Series:
        """
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
        """
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

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3], [4, 5, 6]])
        >>> s
        shape: (2,)
        Series: \'a\' [list[i64]]
        [
                [1, 2, 3]
                [4, 5, 6]
        ]
        >>> s.explode()
        shape: (6,)
        Series: \'a\' [i64]
        [
                1
                2
                3
                4
                5
                6
        ]
        """
    def equals(self, other: Series) -> bool:
        """
        Check whether the Series is equal to another Series.

        Parameters
        ----------
        other
            Series to compare with.
        check_dtypes
            Require data types to match.
        check_names
            Require names to match.
        null_equal
            Consider null values as equal.

        See Also
        --------
        assert_series_equal

        Examples
        --------
        >>> s1 = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("b", [4, 5, 6])
        >>> s1.equals(s1)
        True
        >>> s1.equals(s2)
        False
        """
    def cast(
        self, dtype: PolarsDataType | type[int] | type[float] | type[str] | type[bool]
    ) -> Self:
        """
        Cast between data types.

        Parameters
        ----------
        dtype
            DataType to cast to.
        strict
            If True invalid casts generate exceptions instead of `null`\\s.
        wrap_numerical
            If True numeric casts wrap overflowing values instead of
            marking the cast as invalid.

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
        """
    def to_physical(self) -> Series:
        """
        Cast to physical representation of the logical dtype.

        - :func:`polars.datatypes.Date` -> :func:`polars.datatypes.Int32`
        - :func:`polars.datatypes.Datetime` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Time` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Duration` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Categorical` -> :func:`polars.datatypes.UInt32`
        - `List(inner)` -> `List(physical of inner)`
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
        """
    def to_list(self) -> list[Any]:
        """
        Convert this Series to a Python list.

        This operation copies data.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_list()
        [1, 2, 3]
        >>> type(s.to_list())
        <class \'list\'>
        """
    def rechunk(self) -> Self:
        """
        Create a single chunk of memory for this Series.

        Parameters
        ----------
        in_place
            In place or not.

        Examples
        --------
        >>> s1 = pl.Series("a", [1, 2, 3])
        >>> s1.n_chunks()
        1
        >>> s2 = pl.Series("a", [4, 5, 6])
        >>> s = pl.concat([s1, s2], rechunk=False)
        >>> s.n_chunks()
        2
        >>> s.rechunk(in_place=True)
        shape: (6,)
        Series: \'a\' [i64]
        [
                1
                2
                3
                4
                5
                6
        ]
        >>> s.n_chunks()
        1
        """
    def reverse(self) -> Series:
        """
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
        """
    def is_between(
        self, lower_bound: IntoExpr, upper_bound: IntoExpr, closed: ClosedInterval = ...
    ) -> Series:
        """
        Get a boolean mask of the values that are between the given lower/upper bounds.

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

        Notes
        -----
        If the value of the `lower_bound` is greater than that of the `upper_bound`
        then the result will be False, as no value can satisfy the condition.

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

        Use the `closed` argument to include or exclude the values at the bounds:

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
        """
    def to_numpy(self) -> np.ndarray[Any, Any]:
        """
        Convert this Series to a NumPy ndarray.

        This operation copies data only when necessary. The conversion is zero copy when
        all of the following hold:

        - The data type is an integer, float, `Datetime`, `Duration`, or `Array`.
        - The Series contains no null values.
        - The Series consists of a single chunk.
        - The `writable` parameter is set to `False` (default).

        Parameters
        ----------
        writable
            Ensure the resulting array is writable. This will force a copy of the data
            if the array was created without copy as the underlying Arrow data is
            immutable.
        allow_copy
            Allow memory to be copied to perform the conversion. If set to `False`,
            causes conversions that are not zero-copy to fail.

        use_pyarrow
            First convert to PyArrow, then call `pyarrow.Array.to_numpy
            <https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.to_numpy>`_
            to convert to NumPy. If set to `False`, Polars' own conversion logic is
            used.

            .. deprecated:: 0.20.28
                Polars now uses its native engine by default for conversion to NumPy.
                To use PyArrow's engine, call `.to_arrow().to_numpy()` instead.

        zero_copy_only
            Raise an exception if the conversion to a NumPy would require copying
            the underlying data. Data copy occurs, for example, when the Series contains
            nulls or non-numeric types.

            .. deprecated:: 0.20.10
                Use the `allow_copy` parameter instead, which is the inverse of this
                one.

        Examples
        --------
        Numeric data without nulls can be converted without copying data.
        The resulting array will not be writable.

        >>> s = pl.Series([1, 2, 3], dtype=pl.Int8)
        >>> arr = s.to_numpy()
        >>> arr
        array([1, 2, 3], dtype=int8)
        >>> arr.flags.writeable
        False

        Set `writable=True` to force data copy to make the array writable.

        >>> s.to_numpy(writable=True).flags.writeable
        True

        Integer Series containing nulls will be cast to a float type with `nan`
        representing a null value. This requires data to be copied.

        >>> s = pl.Series([1, 2, None], dtype=pl.UInt16)
        >>> s.to_numpy()
        array([ 1.,  2., nan], dtype=float32)

        Set `allow_copy=False` to raise an error if data would be copied.

        >>> s.to_numpy(allow_copy=False)  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        RuntimeError: copy not allowed: cannot convert to a NumPy array without copying data

        Series of data type `Array` and `Struct` will result in an array with more than
        one dimension.

        >>> s = pl.Series([[1, 2, 3], [4, 5, 6]], dtype=pl.Array(pl.Int64, 3))
        >>> s.to_numpy()
        array([[1, 2, 3],
               [4, 5, 6]])
        """
    def to_jax(self, device: jax.Device | str | None = ...) -> jax.Array:
        """
        Convert this Series to a Jax Array.

        .. versionadded:: 0.20.27

        .. warning::
            This functionality is currently considered **unstable**. It may be
            changed at any point without it being considered a breaking change.

        Parameters
        ----------
        device
            Specify the jax `Device` on which the array will be created; can provide
            a string (such as "cpu", "gpu", or "tpu") in which case the device is
            retrieved as `jax.devices(string)[0]`. For more specific control you
            can supply the instantiated `Device` directly. If None, arrays are
            created on the default device.

        Examples
        --------
        >>> s = pl.Series("x", [10.5, 0.0, -10.0, 5.5])
        >>> s.to_jax()
        Array([ 10.5,   0. , -10. ,   5.5], dtype=float32)
        """
    def to_torch(self) -> torch.Tensor:
        """
        Convert this Series to a PyTorch Tensor.

        .. versionadded:: 0.20.23

        .. warning::
            This functionality is currently considered **unstable**. It may be
            changed at any point without it being considered a breaking change.

        Notes
        -----
        PyTorch tensors do not support UInt16, UInt32, or UInt64; these dtypes
        will be automatically cast to Int32, Int64, and Int64, respectively.

        Examples
        --------
        >>> s = pl.Series("x", [1, 0, 1, 2, 0], dtype=pl.UInt8)
        >>> s.to_torch()
        tensor([1, 0, 1, 2, 0], dtype=torch.uint8)
        >>> s = pl.Series("x", [5.5, -10.0, 2.5], dtype=pl.Float32)
        >>> s.to_torch()
        tensor([  5.5000, -10.0000,   2.5000])
        """
    def to_arrow(self) -> pa.Array:
        """
        Return the underlying Arrow array.

        If the Series contains only a single chunk this operation is zero copy.

        Parameters
        ----------
        compat_level
            Use a specific compatibility level
            when exporting Polars\' internal data structures.

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
        """
    def to_pandas(self, **kwargs: Any) -> pd.Series[Any]:
        """
        Convert this Series to a pandas Series.

        This operation copies data if `use_pyarrow_extension_array` is not enabled.

        Parameters
        ----------
        use_pyarrow_extension_array
            Use a PyArrow-backed extension array instead of a NumPy array for the pandas
            Series. This allows zero copy operations and preservation of null values.
            Subsequent operations on the resulting pandas Series may trigger conversion
            to NumPy if those operations are not supported by PyArrow compute functions.
        **kwargs
            Additional keyword arguments to be passed to
            :meth:`pyarrow.Array.to_pandas`.

        Returns
        -------
        :class:`pandas.Series`

        Notes
        -----
        This operation requires that both :mod:`pandas` and :mod:`pyarrow` are
        installed.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.to_pandas()
        0    1
        1    2
        2    3
        Name: a, dtype: int64

        Null values are converted to `NaN`.

        >>> s = pl.Series("b", [1, 2, None])
        >>> s.to_pandas()
        0    1.0
        1    2.0
        2    NaN
        Name: b, dtype: float64

        Pass `use_pyarrow_extension_array=True` to get a pandas Series backed by a
        PyArrow extension array. This will preserve null values.

        >>> s.to_pandas(use_pyarrow_extension_array=True)
        0       1
        1       2
        2    <NA>
        Name: b, dtype: int64[pyarrow]
        """
    def to_init_repr(self, n: int = ...) -> str:
        """
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
        pl.Series(\'a\', [1, 2, None, 4], dtype=pl.Int16)
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
        """
    def count(self) -> int:
        """
        Return the number of non-null elements in the column.

        See Also
        --------
        len

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, None])
        >>> s.count()
        2
        """
    def len(self) -> int:
        """
        Return the number of elements in the Series.

        Null values count towards the total.

        See Also
        --------
        count

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, None])
        >>> s.len()
        3
        """
    def set(self, filter: Series, value: int | float | str | bool | None) -> Series:
        """
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
        """
    def scatter(
        self,
        indices: Series | Iterable[int] | int | np.ndarray[Any, Any],
        values: Series | Iterable[PythonLiteral] | PythonLiteral | None,
    ) -> Series:
        """
        Set values at the index locations.

        Parameters
        ----------
        indices
            Integers representing the index locations.
        values
            Replacement values.

        Notes
        -----
        Use of this function is frequently an anti-pattern, as it can
        block optimization (predicate pushdown, etc). Consider using
        `pl.when(predicate).then(value).otherwise(self)` instead.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.scatter(1, 10)
        shape: (3,)
        Series: \'a\' [i64]
        [
                1
                10
                3
        ]

        It is better to implement this as follows:

        >>> s.to_frame().with_row_index().select(
        ...     pl.when(pl.col("index") == 1).then(10).otherwise(pl.col("a"))
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
        """
    def clear(self, n: int = ...) -> Series:
        """
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
        """
    def clone(self) -> Self:
        """
        Create a copy of this Series.

        This is a cheap operation that does not copy data.

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
        """
    def fill_nan(self, value: int | float | Expr | None) -> Series:
        """
        Fill floating point NaN value with a fill value.

        Parameters
        ----------
        value
            Value used to fill NaN values.

        Warnings
        --------
        Note that floating point NaNs (Not a Number) are not missing values.
        To replace missing values, use :func:`fill_null`.

        See Also
        --------
        fill_null

        Examples
        --------
        >>> s = pl.Series("a", [1.0, 2.0, 3.0, float("nan")])
        >>> s.fill_nan(0)
        shape: (4,)
        Series: \'a\' [f64]
        [
                1.0
                2.0
                3.0
                0.0
        ]
        """
    def fill_null(
        self,
        value: Any | Expr | None = ...,
        strategy: FillNullStrategy | None = ...,
        limit: int | None = ...,
    ) -> Series:
        """
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

        See Also
        --------
        fill_nan

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
        """
    def floor(self) -> Series:
        """
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
        """
    def ceil(self) -> Series:
        """
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
        """
    def round(self, decimals: int = ...) -> Series:
        """
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
        """
    def round_sig_figs(self, digits: int) -> Series:
        """
        Round to a number of significant figures.

        Parameters
        ----------
        digits
            Number of significant figures to round to.

        Examples
        --------
        >>> s = pl.Series([0.01234, 3.333, 1234.0])
        >>> s.round_sig_figs(2)
        shape: (3,)
        Series: '' [f64]
        [
                0.012
                3.3
                1200.0
        ]
        """
    def dot(self, other: Series | ArrayLike) -> int | float | None:
        """
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
        """
    def mode(self) -> Series:
        """
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
        """
    def sign(self) -> Series:
        """
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
        """
    def sin(self) -> Series:
        """
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
        """
    def cos(self) -> Series:
        """
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
        """
    def tan(self) -> Series:
        """
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
        """
    def cot(self) -> Series:
        """
        Compute the element-wise value for the cotangent.

        Examples
        --------
        >>> import math
        >>> s = pl.Series("a", [0.0, math.pi / 2.0, math.pi])
        >>> s.cot()
        shape: (3,)
        Series: \'a\' [f64]
        [
            inf
            6.1232e-17
            -8.1656e15
        ]
        """
    def arcsin(self) -> Series:
        """
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
        """
    def arccos(self) -> Series:
        """
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
        """
    def arctan(self) -> Series:
        """
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
        """
    def arcsinh(self) -> Series:
        """
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
        """
    def arccosh(self) -> Series:
        """
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
        """
    def arctanh(self) -> Series:
        """
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
        """
    def sinh(self) -> Series:
        """
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
        """
    def cosh(self) -> Series:
        """
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
        """
    def tanh(self) -> Series:
        """
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
        """
    def map_elements(
        self, function: Callable[[Any], Any], return_dtype: PolarsDataType | None = ...
    ) -> Self:
        """
        Map a custom/user-defined function (UDF) over elements in this Series.

        .. warning::
            This method is much slower than the native expressions API.
            Only use it if you cannot implement your logic otherwise.

            Suppose that the function is: `x ↦ sqrt(x)`:

            - For mapping elements of a series, consider: `s.sqrt()`.
            - For mapping inner elements of lists, consider:
              `s.list.eval(pl.element().sqrt())`.
            - For mapping elements of struct fields, consider:
              `s.struct.field("field_name").sqrt()`.

        If the function returns a different datatype, the return_dtype arg should
        be set, otherwise the method will fail.

        Implementing logic using a Python function is almost always *significantly*
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
            Output datatype.
            If not set, the dtype will be inferred based on the first non-null value
            that is returned by the function.
        skip_nulls
            Nulls will be skipped and not passed to the python function.
            This is faster because python can be skipped and because we call
            more specialized functions.

        Warnings
        --------
        If `return_dtype` is not provided, this may lead to unexpected results.
        We allow this, but it is considered a bug in the user\'s query.

        Notes
        -----
        If your function is expensive and you don\'t want it to be called more than
        once for a given input, consider applying an `@lru_cache` decorator to it.
        If your data is suitable you may achieve *significant* speedups.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.map_elements(lambda x: x + 10, return_dtype=pl.Int64)  # doctest: +SKIP
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
        """
    def shift(self, n: int = ...) -> Series:
        """
        Shift values by the given number of indices.

        Parameters
        ----------
        n
            Number of indices to shift forward. If a negative value is passed, values
            are shifted in the opposite direction instead.
        fill_value
            Fill the resulting null values with this value. Accepts expression input.
            Non-expression inputs are parsed as literals.

        Notes
        -----
        This method is similar to the `LAG` operation in SQL when the value for `n`
        is positive. With a negative value for `n`, it is similar to `LEAD`.

        Examples
        --------
        By default, values are shifted forward by one index.

        >>> s = pl.Series([1, 2, 3, 4])
        >>> s.shift()
        shape: (4,)
        Series: '' [i64]
        [
                null
                1
                2
                3
        ]

        Pass a negative value to shift in the opposite direction instead.

        >>> s.shift(-2)
        shape: (4,)
        Series: '' [i64]
        [
                3
                4
                null
                null
        ]

        Specify `fill_value` to fill the resulting null values.

        >>> s.shift(-2, fill_value=100)
        shape: (4,)
        Series: '' [i64]
        [
                3
                4
                100
                100
        ]
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
    def rolling_min(self, window_size: int, weights: list[float] | None = ...) -> Series:
        """
        Apply a rolling min (moving min) over the values in this array.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their min.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window in number of elements.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If set to `None` (default), it will be set equal to `window_size`.
        center
            Set the labels at the center of the window.

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
        """
    def rolling_max(self, window_size: int, weights: list[float] | None = ...) -> Series:
        """
        Apply a rolling max (moving max) over the values in this array.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their max.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window in number of elements.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If set to `None` (default), it will be set equal to `window_size`.
        center
            Set the labels at the center of the window.

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
        """
    def rolling_mean(self, window_size: int, weights: list[float] | None = ...) -> Series:
        """
        Apply a rolling mean (moving mean) over the values in this array.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their mean.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window in number of elements.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If set to `None` (default), it will be set equal to `window_size`.
        center
            Set the labels at the center of the window.

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
        """
    def rolling_sum(self, window_size: int, weights: list[float] | None = ...) -> Series:
        """
        Apply a rolling sum (moving sum) over the values in this array.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window in number of elements.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If set to `None` (default), it will be set equal to `window_size`.
        center
            Set the labels at the center of the window.

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
        """
    def rolling_std(self, window_size: int, weights: list[float] | None = ...) -> Series:
        """
        Compute a rolling std dev.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their std dev.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window in number of elements.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If set to `None` (default), it will be set equal to `window_size`.
        center
            Set the labels at the center of the window.
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
        """
    def rolling_var(self, window_size: int, weights: list[float] | None = ...) -> Series:
        """
        Compute a rolling variance.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their variance.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window in number of elements.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If set to `None` (default), it will be set equal to `window_size`.
        center
            Set the labels at the center of the window.
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
        """
    def rolling_map(
        self, function: Callable[[Series], Any], window_size: int, weights: list[float] | None = ...
    ) -> Series:
        """
        Compute a custom rolling window function.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        function
            Custom aggregation function.
        window_size
            The length of the window in number of elements.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If set to `None` (default), it will be set equal to `window_size`.
        center
            Set the labels at the center of the window.

        Warnings
        --------
        Computing custom functions is extremely slow. Use specialized rolling
        functions such as :func:`Series.rolling_sum` if at all possible.

        Examples
        --------
        >>> from numpy import nansum
        >>> s = pl.Series([11.0, 2.0, 9.0, float("nan"), 8.0])
        >>> s.rolling_map(nansum, window_size=3)
        shape: (5,)
        Series: \'\' [f64]
        [
                null
                null
                22.0
                11.0
                17.0
        ]
        """
    def rolling_median(self, window_size: int, weights: list[float] | None = ...) -> Series:
        """
        Compute a rolling median.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        window_size
            The length of the window in number of elements.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If set to `None` (default), it will be set equal to `window_size`.
        center
            Set the labels at the center of the window.

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
        """
    def rolling_quantile(
        self,
        quantile: float,
        interpolation: RollingInterpolationMethod = ...,
        window_size: int = ...,
        weights: list[float] | None = ...,
    ) -> Series:
        """
        Compute a rolling quantile.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {\'nearest\', \'higher\', \'lower\', \'midpoint\', \'linear\'}
            Interpolation method.
        window_size
            The length of the window in number of elements.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If set to `None` (default), it will be set equal to `window_size`.
        center
            Set the labels at the center of the window.

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
        """
    def rolling_skew(self, window_size: int) -> Series:
        """
        Compute a rolling skew.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

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
    def sample(self, n: int | None = ...) -> Series:
        """
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
            Seed for the random number generator. If set to None (default), a
            random seed is generated for each sample operation.

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
        """
    def peak_max(self) -> Self:
        """
        Get a boolean mask of the local maximum peaks.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.peak_max()
        shape: (5,)
        Series: \'a\' [bool]
        [
                false
                false
                false
                false
                true
        ]
        """
    def peak_min(self) -> Self:
        """
        Get a boolean mask of the local minimum peaks.

        Examples
        --------
        >>> s = pl.Series("a", [4, 1, 3, 2, 5])
        >>> s.peak_min()
        shape: (5,)
        Series: \'a\' [bool]
        [
            false
            true
            false
            true
            false
        ]
        """
    def n_unique(self) -> int:
        """
        Count the number of unique values in this Series.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 2, 3])
        >>> s.n_unique()
        3
        """
    def shrink_to_fit(self) -> Series:
        """
        Shrink Series memory usage.

        Shrinks the underlying array capacity to exactly fit the actual data.
        (Note that this function does not change the Series data type).
        """
    def hash(
        self,
        seed: int = ...,
        seed_1: int | None = ...,
        seed_2: int | None = ...,
        seed_3: int | None = ...,
    ) -> Series:
        """
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

        Notes
        -----
        This implementation of `hash` does not guarantee stable results
        across different Polars versions. Its stability is only guaranteed within a
        single version.

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
        """
    def reinterpret(self) -> Series:
        """
        Reinterpret the underlying bits as a signed/unsigned integer.

        This operation is only allowed for 64bit integers. For lower bits integers,
        you can safely use that cast operation.

        Parameters
        ----------
        signed
            If True, reinterpret as `pl.Int64`. Otherwise, reinterpret as `pl.UInt64`.

        Examples
        --------
        >>> s = pl.Series("a", [-(2**60), -2, 3])
        >>> s
        shape: (3,)
        Series: \'a\' [i64]
        [
                -1152921504606846976
                -2
                3
        ]
        >>> s.reinterpret(signed=False)
        shape: (3,)
        Series: \'a\' [u64]
        [
                17293822569102704640
                18446744073709551614
                3
        ]
        """
    def interpolate(self, method: InterpolationMethod = ...) -> Series:
        """
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
        Series: \'a\' [f64]
        [
            1.0
            2.0
            3.0
            4.0
            5.0
        ]
        """
    def interpolate_by(self, by: IntoExpr) -> Series:
        """
        Fill null values using interpolation based on another column.

        Parameters
        ----------
        by
            Column to interpolate values based on.

        Examples
        --------
        Fill null values using linear interpolation.

        >>> s = pl.Series([1, None, None, 3])
        >>> by = pl.Series([1, 2, 7, 8])
        >>> s.interpolate_by(by)
        shape: (4,)
        Series: '' [f64]
        [
            1.0
            1.285714
            2.714286
            3.0
        ]
        """
    def abs(self) -> Series:
        """
        Compute absolute values.

        Same as `abs(series)`.

        Examples
        --------
        >>> s = pl.Series([1, -2, -3])
        >>> s.abs()
        shape: (3,)
        Series: '' [i64]
        [
            1
            2
            3
        ]
        """
    def rank(self, method: RankMethod = ...) -> Series:
        """
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
        Series: \'a\' [f64]
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
        """
    def diff(self, n: int = ..., null_behavior: NullBehavior = ...) -> Series:
        """
        Calculate the first discrete difference between shifted items.

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
        """
    def pct_change(self, n: int | IntoExprColumn = ...) -> Series:
        """
        Computes percentage change between values.

        Percentage change (as fraction) between current element and most-recent
        non-null element at least `n` period(s) before the current element.

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
        the sample mean.  If `bias` is False, the calculations are
        corrected for bias and the value computed is the adjusted
        Fisher-Pearson standardized moment coefficient, i.e.

        .. math::
            G_1 = \\frac{k_3}{k_2^{3/2}} = \\frac{\\sqrt{N(N-1)}}{N-2}\\frac{m_3}{m_2^{3/2}}

        Examples
        --------
        >>> s = pl.Series([1, 2, 2, 4, 5])
        >>> s.skew()
        0.34776706224699483
        """
    def kurtosis(self) -> float | None:
        """
        Compute the kurtosis (Fisher or Pearson) of a dataset.

        Kurtosis is the fourth central moment divided by the square of the
        variance. If Fisher\'s definition is used, then 3.0 is subtracted from
        the result to give 0.0 for a normal distribution.
        If bias is False then the kurtosis is calculated using k statistics to
        eliminate bias coming from biased moment estimators

        See scipy.stats for more information

        Parameters
        ----------
        fisher : bool, optional
            If True, Fisher\'s definition is used (normal ==> 0.0). If False,
            Pearson\'s definition is used (normal ==> 3.0).
        bias : bool, optional
            If False, the calculations are corrected for statistical bias.

        Examples
        --------
        >>> s = pl.Series("grades", [66, 79, 54, 97, 96, 70, 69, 85, 93, 75])
        >>> s.kurtosis()
        -1.0522623626787952
        >>> s.kurtosis(fisher=False)
        1.9477376373212048
        >>> s.kurtosis(fisher=False, bias=False)
        2.1040361802642726
        """
    def clip(
        self,
        lower_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None = ...,
        upper_bound: NumericLiteral | TemporalLiteral | IntoExprColumn | None = ...,
    ) -> Series:
        """
        Set values outside the given boundaries to the boundary value.

        Parameters
        ----------
        lower_bound
            Lower bound. Accepts expression input.
            Non-expression inputs are parsed as literals.
            If set to `None` (default), no lower bound is applied.
        upper_bound
            Upper bound. Accepts expression input.
            Non-expression inputs are parsed as literals.
            If set to `None` (default), no upper bound is applied.

        See Also
        --------
        when

        Notes
        -----
        This method only works for numeric and temporal columns. To clip other data
        types, consider writing a `when-then-otherwise` expression. See :func:`when`.

        Examples
        --------
        Specifying both a lower and upper bound:

        >>> s = pl.Series([-50, 5, 50, None])
        >>> s.clip(1, 10)
        shape: (4,)
        Series: '' [i64]
        [
                1
                5
                10
                null
        ]

        Specifying only a single bound:

        >>> s.clip(upper_bound=10)
        shape: (4,)
        Series: '' [i64]
        [
                -50
                5
                10
                null
        ]
        """
    def lower_bound(self) -> Self:
        """
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
        """
    def upper_bound(self) -> Self:
        """
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
        """
    def replace(
        self,
        old: IntoExpr | Sequence[Any] | Mapping[Any, Any],
        new: IntoExpr | Sequence[Any] | NoDefault = ...,
    ) -> Self:
        """
        Replace values by different values of the same data type.

        Parameters
        ----------
        old
            Value or sequence of values to replace.
            Also accepts a mapping of values to their replacement as syntactic sugar for
            `replace(old=Series(mapping.keys()), new=Series(mapping.values()))`.
        new
            Value or sequence of values to replace by.
            Length must match the length of `old` or have length 1.

        default
            Set values that were not replaced to this value.
            Defaults to keeping the original value.
            Accepts expression input. Non-expression inputs are parsed as literals.

            .. deprecated:: 0.20.31
                Use :meth:`replace_all` instead to set a default while replacing values.

        return_dtype
            The data type of the resulting expression. If set to `None` (default),
            the data type is determined automatically based on the other inputs.

            .. deprecated:: 0.20.31
                Use :meth:`replace_all` instead to set a return data type while
                replacing values.


        See Also
        --------
        replace_strict
        str.replace

        Notes
        -----
        The global string cache must be enabled when replacing categorical values.

        Examples
        --------
        Replace a single value by another value. Values that were not replaced remain
        unchanged.

        >>> s = pl.Series([1, 2, 2, 3])
        >>> s.replace(2, 100)
        shape: (4,)
        Series: \'\' [i64]
        [
                1
                100
                100
                3
        ]

        Replace multiple values by passing sequences to the `old` and `new` parameters.

        >>> s.replace([2, 3], [100, 200])
        shape: (4,)
        Series: \'\' [i64]
        [
                1
                100
                100
                200
        ]

        Passing a mapping with replacements is also supported as syntactic sugar.

        >>> mapping = {2: 100, 3: 200}
        >>> s.replace(mapping)
        shape: (4,)
        Series: \'\' [i64]
        [
                1
                100
                100
                200
        ]

        The original data type is preserved when replacing by values of a different
        data type. Use :meth:`replace_strict` to replace and change the return data
        type.

        >>> s = pl.Series(["x", "y", "z"])
        >>> mapping = {"x": 1, "y": 2, "z": 3}
        >>> s.replace(mapping)
        shape: (3,)
        Series: \'\' [str]
        [
                "1"
                "2"
                "3"
        ]
        """
    def replace_strict(
        self,
        old: IntoExpr | Sequence[Any] | Mapping[Any, Any],
        new: IntoExpr | Sequence[Any] | NoDefault = ...,
    ) -> Self:
        """
        Replace all values by different values.

        Parameters
        ----------
        old
            Value or sequence of values to replace.
            Also accepts a mapping of values to their replacement as syntactic sugar for
            `replace_all(old=Series(mapping.keys()), new=Series(mapping.values()))`.
        new
            Value or sequence of values to replace by.
            Length must match the length of `old` or have length 1.
        default
            Set values that were not replaced to this value. If no default is specified,
            (default), an error is raised if any values were not replaced.
            Accepts expression input. Non-expression inputs are parsed as literals.
        return_dtype
            The data type of the resulting Series. If set to `None` (default),
            the data type is determined automatically based on the other inputs.

        Raises
        ------
        InvalidOperationError
            If any non-null values in the original column were not replaced, and no
            `default` was specified.

        See Also
        --------
        replace
        str.replace

        Notes
        -----
        The global string cache must be enabled when replacing categorical values.

        Examples
        --------
        Replace values by passing sequences to the `old` and `new` parameters.

        >>> s = pl.Series([1, 2, 2, 3])
        >>> s.replace_strict([1, 2, 3], [100, 200, 300])
        shape: (4,)
        Series: \'\' [i64]
        [
                100
                200
                200
                300
        ]

        Passing a mapping with replacements is also supported as syntactic sugar.

        >>> mapping = {1: 100, 2: 200, 3: 300}
        >>> s.replace_strict(mapping)
        shape: (4,)
        Series: \'\' [i64]
        [
                100
                200
                200
                300
        ]

        By default, an error is raised if any non-null values were not replaced.
        Specify a default to set all values that were not matched.

        >>> mapping = {2: 200, 3: 300}
        >>> s.replace_strict(mapping)  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        polars.exceptions.InvalidOperationError: incomplete mapping specified for `replace_strict`
        >>> s.replace_strict(mapping, default=-1)
        shape: (4,)
        Series: \'\' [i64]
        [
                -1
                200
                200
                300
        ]

        The default can be another Series.

        >>> default = pl.Series([2.5, 5.0, 7.5, 10.0])
        >>> s.replace_strict(2, 200, default=default)
        shape: (4,)
        Series: \'\' [f64]
        [
                2.5
                200.0
                200.0
                10.0
        ]

        Replacing by values of a different data type sets the return type based on
        a combination of the `new` data type and the `default` data type.

        >>> s = pl.Series(["x", "y", "z"])
        >>> mapping = {"x": 1, "y": 2, "z": 3}
        >>> s.replace_strict(mapping)
        shape: (3,)
        Series: \'\' [i64]
        [
                1
                2
                3
        ]
        >>> s.replace_strict(mapping, default="x")
        shape: (3,)
        Series: \'\' [str]
        [
                "1"
                "2"
                "3"
        ]

        Set the `return_dtype` parameter to control the resulting data type directly.

        >>> s.replace_strict(mapping, return_dtype=pl.UInt8)
        shape: (3,)
        Series: \'\' [u8]
        [
                1
                2
                3
        ]
        """
    def reshape(self, dimensions: tuple[int, ...]) -> Series:
        """
        Reshape this Series to a flat Series or an Array Series.

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
            :class:`Array` with shape `dimensions`.

        See Also
        --------
        Series.list.explode : Explode a list column.

        Examples
        --------
        >>> s = pl.Series("foo", [1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> square = s.reshape((3, 3))
        >>> square
        shape: (3,)
        Series: \'foo\' [array[i64, 3]]
        [
                [1, 2, 3]
                [4, 5, 6]
                [7, 8, 9]
        ]
        >>> square.reshape((9,))
        shape: (9,)
        Series: \'foo\' [i64]
        [
                1
                2
                3
                4
                5
                6
                7
                8
                9
        ]
        """
    def shuffle(self, seed: int | None = ...) -> Series:
        """
        Shuffle the contents of this Series.

        Parameters
        ----------
        seed
            Seed for the random number generator. If set to None (default), a
            random seed is generated each time the shuffle is called.

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
        """
    def ewm_mean(self) -> Series:
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

                - When `adjust=True` (the default) the EW function is calculated
                  using weights :math:`w_i = (1 - \\alpha)^i`
                - When `adjust=False` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\\\\n                    y_t &= (1 - \\alpha)y_{t - 1} + \\alpha x_t
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When `ignore_nulls=False` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\\alpha)^2` and :math:`1` if `adjust=True`, and
                  :math:`(1-\\alpha)^2` and :math:`\\alpha` if `adjust=False`.

                - When `ignore_nulls=True`, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\\alpha` and :math:`1` if `adjust=True`,
                  and :math:`1-\\alpha` and :math:`\\alpha` if `adjust=False`.

        Examples
        --------
        >>> s = pl.Series([1, 2, 3])
        >>> s.ewm_mean(com=1, ignore_nulls=False)
        shape: (3,)
        Series: '' [f64]
        [
                1.0
                1.666667
                2.428571
        ]
        """
    def ewm_mean_by(self, by: IntoExpr) -> Series:
        """
        Calculate time-based exponentially weighted moving average.

        Given observations :math:`x_1, x_2, \\ldots, x_n` at times
        :math:`t_1, t_2, \\ldots, t_n`, the EWMA is calculated as

            .. math::

                y_0 &= x_0

                \\alpha_i &= \\exp(-\\lambda(t_i - t_{i-1}))

                y_i &= \\alpha_i x_i + (1 - \\alpha_i) y_{i-1}; \\quad i > 0

        where :math:`\\lambda` equals :math:`\\ln(2) / \\text{half_life}`.

        Parameters
        ----------
        by
            Times to calculate average by. Should be ``DateTime``, ``Date``, ``UInt64``,
            ``UInt32``, ``Int64``, or ``Int32`` data type.
        half_life
            Unit over which observation decays to half its value.

            Can be created either from a timedelta, or
            by using the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1i    (1 index count)

            Or combine them:
            "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

            Note that `half_life` is treated as a constant duration - calendar
            durations such as months (or even days in the time-zone-aware case)
            are not supported, please express your duration in an approximately
            equivalent number of hours (e.g. \'370h\' instead of \'1mo\').

        Returns
        -------
        Expr
            Float32 if input is Float32, otherwise Float64.

        Examples
        --------
        >>> from datetime import date, timedelta
        >>> df = pl.DataFrame(
        ...     {
        ...         "values": [0, 1, 2, None, 4],
        ...         "times": [
        ...             date(2020, 1, 1),
        ...             date(2020, 1, 3),
        ...             date(2020, 1, 10),
        ...             date(2020, 1, 15),
        ...             date(2020, 1, 17),
        ...         ],
        ...     }
        ... ).sort("times")
        >>> df["values"].ewm_mean_by(df["times"], half_life="4d")
        shape: (5,)
        Series: \'values\' [f64]
        [
                0.0
                0.292893
                1.492474
                null
                3.254508
        ]
        """
    def ewm_std(self) -> Series:
        """
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

                - When `adjust=True` (the default) the EW function is calculated
                  using weights :math:`w_i = (1 - \\alpha)^i`
                - When `adjust=False` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\\\\n                    y_t &= (1 - \\alpha)y_{t - 1} + \\alpha x_t
        bias
            When `bias=False`, apply a correction to make the estimate statistically
            unbiased.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When `ignore_nulls=False` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\\alpha)^2` and :math:`1` if `adjust=True`, and
                  :math:`(1-\\alpha)^2` and :math:`\\alpha` if `adjust=False`.

                - When `ignore_nulls=True`, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\\alpha` and :math:`1` if `adjust=True`,
                  and :math:`1-\\alpha` and :math:`\\alpha` if `adjust=False`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.ewm_std(com=1, ignore_nulls=False)
        shape: (3,)
        Series: \'a\' [f64]
        [
            0.0
            0.707107
            0.963624
        ]
        """
    def ewm_var(self) -> Series:
        """
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

                - When `adjust=True` (the default) the EW function is calculated
                  using weights :math:`w_i = (1 - \\alpha)^i`
                - When `adjust=False` the EW function is calculated
                  recursively by

                  .. math::
                    y_0 &= x_0 \\\\\n                    y_t &= (1 - \\alpha)y_{t - 1} + \\alpha x_t
        bias
            When `bias=False`, apply a correction to make the estimate statistically
            unbiased.
        min_periods
            Minimum number of observations in window required to have a value
            (otherwise result is null).
        ignore_nulls
            Ignore missing values when calculating weights.

                - When `ignore_nulls=False` (default), weights are based on absolute
                  positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in
                  calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\\alpha)^2` and :math:`1` if `adjust=True`, and
                  :math:`(1-\\alpha)^2` and :math:`\\alpha` if `adjust=False`.

                - When `ignore_nulls=True`, weights are based
                  on relative positions. For example, the weights of
                  :math:`x_0` and :math:`x_2` used in calculating the final weighted
                  average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`1-\\alpha` and :math:`1` if `adjust=True`,
                  and :math:`1-\\alpha` and :math:`\\alpha` if `adjust=False`.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.ewm_var(com=1, ignore_nulls=False)
        shape: (3,)
        Series: \'a\' [f64]
        [
            0.0
            0.5
            0.928571
        ]
        """
    def extend_constant(self, value: IntoExpr, n: int | IntoExprColumn) -> Series:
        """
        Extremely fast method for extending the Series with 'n' copies of a value.

        Parameters
        ----------
        value
            A constant literal value or a unit expressioin with which to extend the
            expression result Series; can pass None to extend with nulls.
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
        """
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
        """
    def new_from_index(self, index: int, length: int) -> Self:
        """
        Create a new Series filled with values from the given index.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5])
        >>> s.new_from_index(1, 3)
        shape: (3,)
        Series: \'a\' [i64]
        [
            2
            2
            2
        ]
        """
    def shrink_dtype(self) -> Series:
        """
        Shrink numeric columns to the minimal required datatype.

        Shrink to the dtype needed to fit the extrema of this [`Series`].
        This can be used to reduce memory pressure.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3, 4, 5, 6])
        >>> s
        shape: (6,)
        Series: \'a\' [i64]
        [
                1
                2
                3
                4
                5
                6
        ]
        >>> s.shrink_dtype()
        shape: (6,)
        Series: \'a\' [i8]
        [
                1
                2
                3
                4
                5
                6
        ]
        """
    def get_chunks(self) -> list[Series]:
        """
        Get the chunks of this Series as a list of Series.

        Examples
        --------
        >>> s1 = pl.Series("a", [1, 2, 3])
        >>> s2 = pl.Series("a", [4, 5, 6])
        >>> s = pl.concat([s1, s2], rechunk=False)
        >>> s.get_chunks()
        [shape: (3,)
        Series: \'a\' [i64]
        [
                1
                2
                3
        ], shape: (3,)
        Series: \'a\' [i64]
        [
                4
                5
                6
        ]]
        """
    def implode(self) -> Self:
        """
        Aggregate values into a list.

        Examples
        --------
        >>> s = pl.Series("a", [1, 2, 3])
        >>> s.implode()
        shape: (1,)
        Series: \'a\' [list[i64]]
        [
            [1, 2, 3]
        ]
        """
    @property
    def dtype(self): ...
    @property
    def flags(self): ...
    @property
    def name(self): ...
    @property
    def shape(self): ...
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
    @property
    def plot(self): ...

def _resolve_temporal_dtype(
    dtype: PolarsDataType | None, ndtype: np.dtype[np.datetime64] | np.dtype[np.timedelta64]
) -> PolarsDataType | None:
    """Given polars/numpy temporal dtypes, resolve to an explicit unit."""

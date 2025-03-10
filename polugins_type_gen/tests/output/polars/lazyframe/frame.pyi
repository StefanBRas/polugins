from __future__ import annotations
import contextlib
import os
import warnings
from collections.abc import Collection, Mapping
from datetime import date, datetime, time, timedelta
from functools import lru_cache, partial, reduce
from io import BytesIO, StringIO
from operator import and_
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, NoReturn, TypeVar, overload
import polars._reexport as pl
from polars import functions as F
from polars._utils.async_ import _AioDataFrameResult, _GeventDataFrameResult
from polars._utils.convert import negate_duration_string, parse_as_duration_string
from polars._utils.deprecation import deprecate_function, deprecate_renamed_parameter, issue_deprecation_warning
from polars._utils.parse import parse_into_expression, parse_into_list_of_expressions
from polars._utils.serde import serialize_polars_object
from polars._utils.slice import LazyPolarsSlice
from polars._utils.unstable import issue_unstable_warning, unstable
from polars._utils.various import _in_notebook, _is_generator, extend_bool, find_stacklevel, is_bool_sequence, is_sequence, issue_warning, normalize_filepath, parse_percentiles
from polars._utils.wrap import wrap_df, wrap_expr
from polars.datatypes import DTYPE_TEMPORAL_UNITS, N_INFER_DEFAULT, Boolean, Categorical, Date, Datetime, Duration, Enum, Float32, Float64, Int8, Int16, Int32, Int64, Null, Object, String, Time, UInt8, UInt16, UInt32, UInt64, Unknown, is_polars_dtype, parse_into_dtype
from polars.datatypes.group import DataTypeGroup
from polars.dependencies import import_optional, subprocess
from polars.exceptions import PerformanceWarning
from polars.lazyframe.engine_config import GPUEngine
from polars.lazyframe.group_by import LazyGroupBy
from polars.lazyframe.in_process import InProcessQuery
from polars.schema import Schema
from polars.selectors import by_dtype, expand_selector
from polars.polars import PyLazyFrame
import sys
from collections.abc import Awaitable, Iterable, Sequence
from io import IOBase
from typing import Literal
import pyarrow as pa
from polars import DataFrame, DataType, Expr
from polars._typing import AsofJoinStrategy, ClosedInterval, ColumnNameOrSelector, CsvQuoteStyle, EngineType, ExplainFormat, FillNullStrategy, FrameInitTypes, IntoExpr, IntoExprColumn, JoinStrategy, JoinValidation, Label, Orientation, PolarsDataType, RollingInterpolationMethod, SchemaDefinition, SchemaDict, SerializationFormat, StartBy, UniqueKeepStrategy
from polars.dependencies import numpy as np
if sys.version_info >= (3, 10):
    from typing import Concatenate, ParamSpec
else:
    from typing_extensions import Concatenate, ParamSpec
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
T = TypeVar('T')
P = ParamSpec('P')

class LazyFrame:
    """
    Representation of a Lazy computation graph/query against a DataFrame.

    This allows for whole-query optimisation in addition to parallelism, and
    is the preferred (and highest-performance) mode of operation for polars.

    Parameters
    ----------
    data : dict, Sequence, ndarray, Series, or pandas.DataFrame
        Two-dimensional data in various forms; dict input must contain Sequences,
        Generators, or a `range`. Sequence may contain Series or other Sequences.
    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
        The LazyFrame schema may be declared in several ways:

        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
        * As a list of column names; in this case types are automatically inferred.
        * As a list of (name,type) pairs; this is equivalent to the dictionary form.

        If you supply a list of column names that does not match the names in the
        underlying data, the names given here will overwrite them. The number
        of names given in the schema should match the underlying data dimensions.
    schema_overrides : dict, default None
        Support type specification or override of one or more columns; note that
        any dtypes inferred from the schema param will be overridden.

        The number of entries in the schema should match the underlying data
        dimensions, unless a sequence of dictionaries is being passed, in which case
        a *partial* schema can be declared to prevent specific fields from being loaded.
    strict : bool, default True
        Throw an error if any `data` value does not exactly match the given or inferred
        data type for that column. If set to `False`, values that do not match the data
        type are cast to that data type or, if casting is not possible, set to null
        instead.
    orient : {'col', 'row'}, default None
        Whether to interpret two-dimensional data as columns or as rows. If None,
        the orientation is inferred by matching the columns and data dimensions. If
        this does not yield conclusive results, column orientation is used.
    infer_schema_length : int or None
        The maximum number of rows to scan for schema inference. If set to `None`, the
        full data may be scanned *(this can be slow)*. This parameter only applies if
        the input data is a sequence or generator of rows; other input is read as-is.
    nan_to_null : bool, default False
        If the data comes from one or more numpy arrays, can optionally convert input
        data np.nan values to null instead. This is a no-op for all other input data.

    Notes
    -----
    Initialising `LazyFrame(...)` directly is equivalent to `DataFrame(...).lazy()`.

    Examples
    --------
    Constructing a LazyFrame directly from a dictionary:

    >>> data = {"a": [1, 2], "b": [3, 4]}
    >>> lf = pl.LazyFrame(data)
    >>> lf.collect()
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    │ 2   ┆ 4   │
    └─────┴─────┘

    Notice that the dtypes are automatically inferred as Polars Int64:

    >>> lf.collect_schema().dtypes()
    [Int64, Int64]

    To specify a more detailed/specific frame schema you can supply the `schema`
    parameter with a dictionary of (name,dtype) pairs...

    >>> data = {"col1": [0, 2], "col2": [3, 7]}
    >>> lf2 = pl.LazyFrame(data, schema={"col1": pl.Float32, "col2": pl.Int64})
    >>> lf2.collect()
    shape: (2, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ f32  ┆ i64  │
    ╞══════╪══════╡
    │ 0.0  ┆ 3    │
    │ 2.0  ┆ 7    │
    └──────┴──────┘

    ...a sequence of (name,dtype) pairs...

    >>> data = {"col1": [1, 2], "col2": [3, 4]}
    >>> lf3 = pl.LazyFrame(data, schema=[("col1", pl.Float32), ("col2", pl.Int64)])
    >>> lf3.collect()
    shape: (2, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ f32  ┆ i64  │
    ╞══════╪══════╡
    │ 1.0  ┆ 3    │
    │ 2.0  ┆ 4    │
    └──────┴──────┘

    ...or a list of typed Series.

    >>> data = [
    ...     pl.Series("col1", [1, 2], dtype=pl.Float32),
    ...     pl.Series("col2", [3, 4], dtype=pl.Int64),
    ... ]
    >>> lf4 = pl.LazyFrame(data)
    >>> lf4.collect()
    shape: (2, 2)
    ┌──────┬──────┐
    │ col1 ┆ col2 │
    │ ---  ┆ ---  │
    │ f32  ┆ i64  │
    ╞══════╪══════╡
    │ 1.0  ┆ 3    │
    │ 2.0  ┆ 4    │
    └──────┴──────┘

    Constructing a LazyFrame from a numpy ndarray, specifying column names:

    >>> import numpy as np
    >>> data = np.array([(1, 2), (3, 4)], dtype=np.int64)
    >>> lf5 = pl.LazyFrame(data, schema=["a", "b"], orient="col")
    >>> lf5.collect()
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    │ 2   ┆ 4   │
    └─────┴─────┘

    Constructing a LazyFrame from a list of lists, row orientation specified:

    >>> data = [[1, 2, 3], [4, 5, 6]]
    >>> lf6 = pl.LazyFrame(data, schema=["a", "b", "c"], orient="row")
    >>> lf6.collect()
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 2   ┆ 3   │
    │ 4   ┆ 5   ┆ 6   │
    └─────┴─────┴─────┘
    """
    _ldf: PyLazyFrame
    _accessors: ClassVar[set[str]]

    def __init__(self, data: FrameInitTypes | None=None, schema: SchemaDefinition | None=None, *, schema_overrides: SchemaDict | None=None, strict: bool=True, orient: Orientation | None=None, infer_schema_length: int | None=N_INFER_DEFAULT, nan_to_null: bool=False) -> None:
        ...

    @classmethod
    def _from_pyldf(cls, ldf: PyLazyFrame) -> LazyFrame:
        ...

    def __getstate__(self) -> bytes:
        ...

    def __setstate__(self, state: bytes) -> None:
        ...

    @classmethod
    def _scan_python_function(cls, schema: pa.schema | Mapping[str, PolarsDataType], scan_fn: Any, *, pyarrow: bool=False) -> LazyFrame:
        ...

    @classmethod
    def deserialize(cls, source: str | Path | IOBase, *, format: SerializationFormat='binary') -> LazyFrame:
        """
        Read a logical plan from a file to construct a LazyFrame.

        Parameters
        ----------
        source
            Path to a file or a file-like object (by file-like object, we refer to
            objects that have a `read()` method, such as a file handler (e.g.
            via builtin `open` function) or `BytesIO`).
        format
            The format with which the LazyFrame was serialized. Options:

            - `"binary"`: Deserialize from binary format (bytes). This is the default.
            - `"json"`: Deserialize from JSON format (string).

        Warnings
        --------
        This function uses :mod:`pickle` if the logical plan contains Python UDFs,
        and as such inherits the security implications. Deserializing can execute
        arbitrary code, so it should only be attempted on trusted data.

        See Also
        --------
        LazyFrame.serialize

        Notes
        -----
        Serialization is not stable across Polars versions: a LazyFrame serialized
        in one Polars version may not be deserializable in another Polars version.

        Examples
        --------
        >>> import io
        >>> lf = pl.LazyFrame({"a": [1, 2, 3]}).sum()
        >>> bytes = lf.serialize()
        >>> pl.LazyFrame.deserialize(io.BytesIO(bytes)).collect()
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 6   │
        └─────┘
        """
        ...

    @property
    def columns(self) -> list[str]:
        """
        Get the column names.

        Returns
        -------
        list of str
            A list containing the name of each column in order.

        Warnings
        --------
        Determining the column names of a LazyFrame requires resolving its schema,
        which is a potentially expensive operation.
        Using :meth:`collect_schema` is the idiomatic way of resolving the schema.
        This property exists only for symmetry with the DataFrame class.

        See Also
        --------
        collect_schema
        Schema.names

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... ).select("foo", "bar")
        >>> lf.columns  # doctest: +SKIP
        ['foo', 'bar']
        """
        ...

    @property
    def dtypes(self) -> list[DataType]:
        """
        Get the column data types.

        Returns
        -------
        list of DataType
            A list containing the data type of each column in order.

        Warnings
        --------
        Determining the data types of a LazyFrame requires resolving its schema,
        which is a potentially expensive operation.
        Using :meth:`collect_schema` is the idiomatic way to resolve the schema.
        This property exists only for symmetry with the DataFrame class.

        See Also
        --------
        collect_schema
        Schema.dtypes

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.dtypes  # doctest: +SKIP
        [Int64, Float64, String]
        """
        ...

    @property
    def schema(self) -> Schema:
        """
        Get an ordered mapping of column names to their data type.

        Warnings
        --------
        Resolving the schema of a LazyFrame is a potentially expensive operation.
        Using :meth:`collect_schema` is the idiomatic way to resolve the schema.
        This property exists only for symmetry with the DataFrame class.

        See Also
        --------
        collect_schema
        Schema

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.schema  # doctest: +SKIP
        Schema({'foo': Int64, 'bar': Float64, 'ham': String})
        """
        ...

    @property
    def width(self) -> int:
        """
        Get the number of columns.

        Returns
        -------
        int

        Warnings
        --------
        Determining the width of a LazyFrame requires resolving its schema,
        which is a potentially expensive operation.
        Using :meth:`collect_schema` is the idiomatic way to resolve the schema.
        This property exists only for symmetry with the DataFrame class.

        See Also
        --------
        collect_schema
        Schema.len

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [4, 5, 6],
        ...     }
        ... )
        >>> lf.width  # doctest: +SKIP
        2
        """
        ...

    def __bool__(self) -> NoReturn:
        ...

    def _comparison_error(self, operator: str) -> NoReturn:
        ...

    def __eq__(self, other: object) -> NoReturn:
        ...

    def __ne__(self, other: object) -> NoReturn:
        ...

    def __gt__(self, other: Any) -> NoReturn:
        ...

    def __lt__(self, other: Any) -> NoReturn:
        ...

    def __ge__(self, other: Any) -> NoReturn:
        ...

    def __le__(self, other: Any) -> NoReturn:
        ...

    def __contains__(self, key: str) -> bool:
        ...

    def __copy__(self) -> LazyFrame:
        ...

    def __deepcopy__(self, memo: None=None) -> LazyFrame:
        ...

    def __getitem__(self, item: int | range | slice) -> LazyFrame:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def _repr_html_(self) -> str:
        ...

    @overload
    def serialize(self, file: None=..., *, format: Literal['binary']=...) -> bytes:
        ...

    @overload
    def serialize(self, file: None=..., *, format: Literal['json']) -> str:
        ...

    @overload
    def serialize(self, file: IOBase | str | Path, *, format: SerializationFormat=...) -> None:
        ...

    def serialize(self, file: IOBase | str | Path | None=None, *, format: SerializationFormat='binary') -> bytes | str | None:
        """
        Serialize the logical plan of this LazyFrame to a file or string in JSON format.

        Parameters
        ----------
        file
            File path to which the result should be written. If set to `None`
            (default), the output is returned as a string instead.
        format
            The format in which to serialize. Options:

            - `"binary"`: Serialize to binary format (bytes). This is the default.
            - `"json"`: Serialize to JSON format (string) (deprecated).

        See Also
        --------
        LazyFrame.deserialize

        Notes
        -----
        Serialization is not stable across Polars versions: a LazyFrame serialized
        in one Polars version may not be deserializable in another Polars version.

        Examples
        --------
        Serialize the logical plan into a binary representation.

        >>> lf = pl.LazyFrame({"a": [1, 2, 3]}).sum()
        >>> bytes = lf.serialize()

        The bytes can later be deserialized back into a LazyFrame.

        >>> import io
        >>> pl.LazyFrame.deserialize(io.BytesIO(bytes)).collect()
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 6   │
        └─────┘
        """
        ...

    def pipe(self, function: Callable[Concatenate[LazyFrame, P], T], *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Offers a structured way to apply a sequence of user-defined functions (UDFs).

        Parameters
        ----------
        function
            Callable; will receive the frame as the first parameter,
            followed by any given args/kwargs.
        *args
            Arguments to pass to the UDF.
        **kwargs
            Keyword arguments to pass to the UDF.

        Examples
        --------
        >>> def cast_str_to_int(lf: pl.LazyFrame, col_name: str) -> pl.LazyFrame:
        ...     return lf.with_columns(pl.col(col_name).cast(pl.Int64))
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": ["10", "20", "30", "40"],
        ...     }
        ... )
        >>> lf.pipe(cast_str_to_int, col_name="b").collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 10  │
        │ 2   ┆ 20  │
        │ 3   ┆ 30  │
        │ 4   ┆ 40  │
        └─────┴─────┘

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "b": [1, 2],
        ...         "a": [3, 4],
        ...     }
        ... )
        >>> lf.collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ b   ┆ a   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        └─────┴─────┘
        >>> lf.pipe(lambda lf: lf.select(sorted(lf.collect_schema()))).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 3   ┆ 1   │
        │ 4   ┆ 2   │
        └─────┴─────┘
        """
        ...

    def describe(self, percentiles: Sequence[float] | float | None=(0.25, 0.5, 0.75), *, interpolation: RollingInterpolationMethod='nearest') -> DataFrame:
        """
        Creates a summary of statistics for a LazyFrame, returning a DataFrame.

        Parameters
        ----------
        percentiles
            One or more percentiles to include in the summary statistics.
            All values must be in the range `[0, 1]`.

        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method used when calculating percentiles.

        Returns
        -------
        DataFrame

        Notes
        -----
        The median is included by default as the 50% percentile.

        Warnings
        --------
        * This method does *not* maintain the laziness of the frame, and will `collect`
          the final result. This could potentially be an expensive operation.
        * We do not guarantee the output of `describe` to be stable. It will show
          statistics that we deem informative, and may be updated in the future.
          Using `describe` programmatically (versus interactive exploration) is
          not recommended for this reason.

        Examples
        --------
        >>> from datetime import date, time
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "float": [1.0, 2.8, 3.0],
        ...         "int": [40, 50, None],
        ...         "bool": [True, False, True],
        ...         "str": ["zz", "xx", "yy"],
        ...         "date": [date(2020, 1, 1), date(2021, 7, 5), date(2022, 12, 31)],
        ...         "time": [time(10, 20, 30), time(14, 45, 50), time(23, 15, 10)],
        ...     }
        ... )

        Show default frame statistics:

        >>> lf.describe()
        shape: (9, 7)
        ┌────────────┬──────────┬──────────┬──────────┬──────┬─────────────────────┬──────────┐
        │ statistic  ┆ float    ┆ int      ┆ bool     ┆ str  ┆ date                ┆ time     │
        │ ---        ┆ ---      ┆ ---      ┆ ---      ┆ ---  ┆ ---                 ┆ ---      │
        │ str        ┆ f64      ┆ f64      ┆ f64      ┆ str  ┆ str                 ┆ str      │
        ╞════════════╪══════════╪══════════╪══════════╪══════╪═════════════════════╪══════════╡
        │ count      ┆ 3.0      ┆ 2.0      ┆ 3.0      ┆ 3    ┆ 3                   ┆ 3        │
        │ null_count ┆ 0.0      ┆ 1.0      ┆ 0.0      ┆ 0    ┆ 0                   ┆ 0        │
        │ mean       ┆ 2.266667 ┆ 45.0     ┆ 0.666667 ┆ null ┆ 2021-07-02 16:00:00 ┆ 16:07:10 │
        │ std        ┆ 1.101514 ┆ 7.071068 ┆ null     ┆ null ┆ null                ┆ null     │
        │ min        ┆ 1.0      ┆ 40.0     ┆ 0.0      ┆ xx   ┆ 2020-01-01          ┆ 10:20:30 │
        │ 25%        ┆ 2.8      ┆ 40.0     ┆ null     ┆ null ┆ 2021-07-05          ┆ 14:45:50 │
        │ 50%        ┆ 2.8      ┆ 50.0     ┆ null     ┆ null ┆ 2021-07-05          ┆ 14:45:50 │
        │ 75%        ┆ 3.0      ┆ 50.0     ┆ null     ┆ null ┆ 2022-12-31          ┆ 23:15:10 │
        │ max        ┆ 3.0      ┆ 50.0     ┆ 1.0      ┆ zz   ┆ 2022-12-31          ┆ 23:15:10 │
        └────────────┴──────────┴──────────┴──────────┴──────┴─────────────────────┴──────────┘

        Customize which percentiles are displayed, applying linear interpolation:

        >>> with pl.Config(tbl_rows=12):
        ...     lf.describe(
        ...         percentiles=[0.1, 0.3, 0.5, 0.7, 0.9],
        ...         interpolation="linear",
        ...     )
        shape: (11, 7)
        ┌────────────┬──────────┬──────────┬──────────┬──────┬─────────────────────┬──────────┐
        │ statistic  ┆ float    ┆ int      ┆ bool     ┆ str  ┆ date                ┆ time     │
        │ ---        ┆ ---      ┆ ---      ┆ ---      ┆ ---  ┆ ---                 ┆ ---      │
        │ str        ┆ f64      ┆ f64      ┆ f64      ┆ str  ┆ str                 ┆ str      │
        ╞════════════╪══════════╪══════════╪══════════╪══════╪═════════════════════╪══════════╡
        │ count      ┆ 3.0      ┆ 2.0      ┆ 3.0      ┆ 3    ┆ 3                   ┆ 3        │
        │ null_count ┆ 0.0      ┆ 1.0      ┆ 0.0      ┆ 0    ┆ 0                   ┆ 0        │
        │ mean       ┆ 2.266667 ┆ 45.0     ┆ 0.666667 ┆ null ┆ 2021-07-02 16:00:00 ┆ 16:07:10 │
        │ std        ┆ 1.101514 ┆ 7.071068 ┆ null     ┆ null ┆ null                ┆ null     │
        │ min        ┆ 1.0      ┆ 40.0     ┆ 0.0      ┆ xx   ┆ 2020-01-01          ┆ 10:20:30 │
        │ 10%        ┆ 1.36     ┆ 41.0     ┆ null     ┆ null ┆ 2020-04-20          ┆ 11:13:34 │
        │ 30%        ┆ 2.08     ┆ 43.0     ┆ null     ┆ null ┆ 2020-11-26          ┆ 12:59:42 │
        │ 50%        ┆ 2.8      ┆ 45.0     ┆ null     ┆ null ┆ 2021-07-05          ┆ 14:45:50 │
        │ 70%        ┆ 2.88     ┆ 47.0     ┆ null     ┆ null ┆ 2022-02-07          ┆ 18:09:34 │
        │ 90%        ┆ 2.96     ┆ 49.0     ┆ null     ┆ null ┆ 2022-09-13          ┆ 21:33:18 │
        │ max        ┆ 3.0      ┆ 50.0     ┆ 1.0      ┆ zz   ┆ 2022-12-31          ┆ 23:15:10 │
        └────────────┴──────────┴──────────┴──────────┴──────┴─────────────────────┴──────────┘
        """
        ...

    def explain(self, *, format: ExplainFormat='plain', optimized: bool=True, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, streaming: bool=False, tree_format: bool | None=None) -> str:
        """
        Create a string representation of the query plan.

        Different optimizations can be turned on or off.

        Parameters
        ----------
        format : {'plain', 'tree'}
            The format to use for displaying the logical plan.
        optimized
            Return an optimized query plan. Defaults to `True`.
            If this is set to `True` the subsequent
            optimization flags control which optimizations
            run.
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        slice_pushdown
            Slice pushdown optimization.
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        cluster_with_columns
            Combine sequential independent calls to with_columns
        collapse_joins
            Collapse a join and filters into a faster join
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

            .. warning::
                Streaming mode is considered **unstable**. It may be changed
                at any point without it being considered a breaking change.

        tree_format
            Format the output as a tree.

            .. deprecated:: 0.20.30
                Use `format="tree"` instead.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a", maintain_order=True).agg(pl.all().sum()).sort(
        ...     "a"
        ... ).explain()  # doctest: +SKIP
        """
        ...

    def show_graph(self, *, optimized: bool=True, show: bool=True, output_path: str | Path | None=None, raw_output: bool=False, figsize: tuple[float, float]=(16.0, 12.0), type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, streaming: bool=False) -> str | None:
        """
        Show a plot of the query plan.

        Note that graphviz must be installed to render the visualization (if not
        already present you can download it here: <https://graphviz.org/download>`_).

        Parameters
        ----------
        optimized
            Optimize the query plan.
        show
            Show the figure.
        output_path
            Write the figure to disk.
        raw_output
            Return dot syntax. This cannot be combined with `show` and/or `output_path`.
        figsize
            Passed to matplotlib if `show` == True.
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        slice_pushdown
            Slice pushdown optimization.
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        cluster_with_columns
            Combine sequential independent calls to with_columns
        collapse_joins
            Collapse a join and filters into a faster join
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a", maintain_order=True).agg(pl.all().sum()).sort(
        ...     "a"
        ... ).show_graph()  # doctest: +SKIP
        """
        ...

    def inspect(self, fmt: str='{}') -> LazyFrame:
        """
        Inspect a node in the computation graph.

        Print the value that this node in the computation graph evaluates to and pass on
        the value.

        Examples
        --------
        >>> lf = pl.LazyFrame({"foo": [1, 1, -2, 3]})
        >>> (
        ...     lf.with_columns(pl.col("foo").cum_sum().alias("bar"))
        ...     .inspect()  # print the node before the filter
        ...     .filter(pl.col("bar") == pl.col("foo"))
        ... )  # doctest: +ELLIPSIS
        <LazyFrame at ...>
        """
        ...

    def sort(self, by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr, descending: bool | Sequence[bool]=False, nulls_last: bool | Sequence[bool]=False, maintain_order: bool=False, multithreaded: bool=True) -> LazyFrame:
        """
        Sort the LazyFrame by the given columns.

        Parameters
        ----------
        by
            Column(s) to sort by. Accepts expression input, including selectors. Strings
            are parsed as column names.
        *more_by
            Additional columns to sort by, specified as positional arguments.
        descending
            Sort in descending order. When sorting by multiple columns, can be specified
            per column by passing a sequence of booleans.
        nulls_last
            Place null values last; can specify a single boolean applying to all columns
            or a sequence of booleans for per-column control.
        maintain_order
            Whether the order should be maintained if elements are equal.
            Note that if `true` streaming is not possible and performance might be
            worse since this requires a stable search.
        multithreaded
            Sort using multiple threads.

        Examples
        --------
        Pass a single column name to sort by that column.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": [6.0, 5.0, 4.0],
        ...         "c": ["a", "c", "b"],
        ...     }
        ... )
        >>> lf.sort("a").collect()
        shape: (3, 3)
        ┌──────┬─────┬─────┐
        │ a    ┆ b   ┆ c   │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ null ┆ 4.0 ┆ b   │
        │ 1    ┆ 6.0 ┆ a   │
        │ 2    ┆ 5.0 ┆ c   │
        └──────┴─────┴─────┘

        Sorting by expressions is also supported.

        >>> lf.sort(pl.col("a") + pl.col("b") * 2, nulls_last=True).collect()
        shape: (3, 3)
        ┌──────┬─────┬─────┐
        │ a    ┆ b   ┆ c   │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ 2    ┆ 5.0 ┆ c   │
        │ 1    ┆ 6.0 ┆ a   │
        │ null ┆ 4.0 ┆ b   │
        └──────┴─────┴─────┘

        Sort by multiple columns by passing a list of columns.

        >>> lf.sort(["c", "a"], descending=True).collect()
        shape: (3, 3)
        ┌──────┬─────┬─────┐
        │ a    ┆ b   ┆ c   │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ 2    ┆ 5.0 ┆ c   │
        │ null ┆ 4.0 ┆ b   │
        │ 1    ┆ 6.0 ┆ a   │
        └──────┴─────┴─────┘

        Or use positional arguments to sort by multiple columns in the same way.

        >>> lf.sort("c", "a", descending=[False, True]).collect()
        shape: (3, 3)
        ┌──────┬─────┬─────┐
        │ a    ┆ b   ┆ c   │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ f64 ┆ str │
        ╞══════╪═════╪═════╡
        │ 1    ┆ 6.0 ┆ a   │
        │ null ┆ 4.0 ┆ b   │
        │ 2    ┆ 5.0 ┆ c   │
        └──────┴─────┴─────┘
        """
        ...

    def sql(self, query: str, *, table_name: str='self') -> LazyFrame:
        """
        Execute a SQL query against the LazyFrame.

        .. versionadded:: 0.20.23

        .. warning::
            This functionality is considered **unstable**, although it is close to
            being considered stable. It may be changed at any point without it being
            considered a breaking change.

        Parameters
        ----------
        query
            SQL query to execute.
        table_name
            Optionally provide an explicit name for the table that represents the
            calling frame (defaults to "self").

        Notes
        -----
        * The calling frame is automatically registered as a table in the SQL context
          under the name "self". If you want access to the DataFrames and LazyFrames
          found in the current globals, use the top-level :meth:`pl.sql <polars.sql>`.
        * More control over registration and execution behaviour is available by
          using the :class:`SQLContext` object.

        See Also
        --------
        SQLContext

        Examples
        --------
        >>> lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": [6, 7, 8], "c": ["z", "y", "x"]})
        >>> lf2 = pl.LazyFrame({"a": [3, 2, 1], "d": [125, -654, 888]})

        Query the LazyFrame using SQL:

        >>> lf1.sql("SELECT c, b FROM self WHERE a > 1").collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ c   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ y   ┆ 7   │
        │ x   ┆ 8   │
        └─────┴─────┘

        Apply SQL transforms (aliasing "self" to "frame") then filter
        natively (you can freely mix SQL and native operations):

        >>> lf1.sql(
        ...     query='''
        ...         SELECT
        ...             a,
        ...             (a % 2 == 0) AS a_is_even,
        ...             (b::float4 / 2) AS "b/2",
        ...             CONCAT_WS(':', c, c, c) AS c_c_c
        ...         FROM frame
        ...         ORDER BY a
        ...     ''',
        ...     table_name="frame",
        ... ).filter(~pl.col("c_c_c").str.starts_with("x")).collect()
        shape: (2, 4)
        ┌─────┬───────────┬─────┬───────┐
        │ a   ┆ a_is_even ┆ b/2 ┆ c_c_c │
        │ --- ┆ ---       ┆ --- ┆ ---   │
        │ i64 ┆ bool      ┆ f32 ┆ str   │
        ╞═════╪═══════════╪═════╪═══════╡
        │ 1   ┆ false     ┆ 3.0 ┆ z:z:z │
        │ 2   ┆ true      ┆ 3.5 ┆ y:y:y │
        └─────┴───────────┴─────┴───────┘
        """
        ...

    @deprecate_renamed_parameter('descending', 'reverse', version='1.0.0')
    def top_k(self, k: int, *, by: IntoExpr | Iterable[IntoExpr], reverse: bool | Sequence[bool]=False) -> LazyFrame:
        """
        Return the `k` largest rows.

        Non-null elements are always preferred over null elements, regardless of
        the value of `reverse`. The output is not guaranteed to be in any
        particular order, call :func:`sort` after this function if you wish the
        output to be sorted.

        Parameters
        ----------
        k
            Number of rows to return.
        by
            Column(s) used to determine the top rows.
            Accepts expression input. Strings are parsed as column names.
        reverse
            Consider the `k` smallest elements of the `by` column(s) (instead of the `k`
            largest). This can be specified per column by passing a sequence of
            booleans.

        See Also
        --------
        bottom_k

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [2, 1, 1, 3, 2, 1],
        ...     }
        ... )

        Get the rows which contain the 4 largest values in column b.

        >>> lf.top_k(4, by="b").collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ b   ┆ 3   │
        │ a   ┆ 2   │
        │ b   ┆ 2   │
        │ b   ┆ 1   │
        └─────┴─────┘

        Get the rows which contain the 4 largest values when sorting on column b and a.

        >>> lf.top_k(4, by=["b", "a"]).collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ b   ┆ 3   │
        │ b   ┆ 2   │
        │ a   ┆ 2   │
        │ c   ┆ 1   │
        └─────┴─────┘
        """
        ...

    @deprecate_renamed_parameter('descending', 'reverse', version='1.0.0')
    def bottom_k(self, k: int, *, by: IntoExpr | Iterable[IntoExpr], reverse: bool | Sequence[bool]=False) -> LazyFrame:
        """
        Return the `k` smallest rows.

        Non-null elements are always preferred over null elements, regardless of
        the value of `reverse`. The output is not guaranteed to be in any
        particular order, call :func:`sort` after this function if you wish the
        output to be sorted.

        Parameters
        ----------
        k
            Number of rows to return.
        by
            Column(s) used to determine the bottom rows.
            Accepts expression input. Strings are parsed as column names.
        reverse
            Consider the `k` largest elements of the `by` column(s) (instead of the `k`
            smallest). This can be specified per column by passing a sequence of
            booleans.

        See Also
        --------
        top_k

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [2, 1, 1, 3, 2, 1],
        ...     }
        ... )

        Get the rows which contain the 4 smallest values in column b.

        >>> lf.bottom_k(4, by="b").collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ b   ┆ 1   │
        │ a   ┆ 1   │
        │ c   ┆ 1   │
        │ a   ┆ 2   │
        └─────┴─────┘

        Get the rows which contain the 4 smallest values when sorting on column a and b.

        >>> lf.bottom_k(4, by=["a", "b"]).collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 1   │
        │ a   ┆ 2   │
        │ b   ┆ 1   │
        │ b   ┆ 2   │
        └─────┴─────┘
        """
        ...

    def profile(self, *, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, no_optimization: bool=False, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, show_plot: bool=False, truncate_nodes: int=0, figsize: tuple[int, int]=(18, 8), streaming: bool=False) -> tuple[DataFrame, DataFrame]:
        """
        Profile a LazyFrame.

        This will run the query and return a tuple
        containing the materialized DataFrame and a DataFrame that
        contains profiling information of each node that is executed.

        The units of the timings are microseconds.

        Parameters
        ----------
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        no_optimization
            Turn off (certain) optimizations.
        slice_pushdown
            Slice pushdown optimization.
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        cluster_with_columns
            Combine sequential independent calls to with_columns
        collapse_joins
            Collapse a join and filters into a faster join
        show_plot
            Show a gantt chart of the profiling result
        truncate_nodes
            Truncate the label lengths in the gantt chart to this number of
            characters.
        figsize
            matplotlib figsize of the profiling plot
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a", maintain_order=True).agg(pl.all().sum()).sort(
        ...     "a"
        ... ).profile()  # doctest: +SKIP
        (shape: (3, 3)
         ┌─────┬─────┬─────┐
         │ a   ┆ b   ┆ c   │
         │ --- ┆ --- ┆ --- │
         │ str ┆ i64 ┆ i64 │
         ╞═════╪═════╪═════╡
         │ a   ┆ 4   ┆ 10  │
         │ b   ┆ 11  ┆ 10  │
         │ c   ┆ 6   ┆ 1   │
         └─────┴─────┴─────┘,
         shape: (3, 3)
         ┌─────────────────────────┬───────┬──────┐
         │ node                    ┆ start ┆ end  │
         │ ---                     ┆ ---   ┆ ---  │
         │ str                     ┆ u64   ┆ u64  │
         ╞═════════════════════════╪═══════╪══════╡
         │ optimization            ┆ 0     ┆ 5    │
         │ group_by_partitioned(a) ┆ 5     ┆ 470  │
         │ sort(a)                 ┆ 475   ┆ 1964 │
         └─────────────────────────┴───────┴──────┘)
        """
        ...

    @overload
    def collect(self, *, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, no_optimization: bool=False, streaming: bool=False, engine: EngineType='cpu', background: Literal[True], _eager: bool=False) -> InProcessQuery:
        ...

    @overload
    def collect(self, *, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, no_optimization: bool=False, streaming: bool=False, engine: EngineType='cpu', background: Literal[False]=False, _eager: bool=False) -> DataFrame:
        ...

    def collect(self, *, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, no_optimization: bool=False, streaming: bool=False, engine: EngineType='cpu', background: bool=False, _eager: bool=False, **_kwargs: Any) -> DataFrame | InProcessQuery:
        """
        Materialize this LazyFrame into a DataFrame.

        By default, all query optimizations are enabled. Individual optimizations may
        be disabled by setting the corresponding parameter to `False`.

        Parameters
        ----------
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        slice_pushdown
            Slice pushdown optimization.
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        cluster_with_columns
            Combine sequential independent calls to with_columns
        collapse_joins
            Collapse a join and filters into a faster join
        no_optimization
            Turn off (certain) optimizations.
        streaming
            Process the query in batches to handle larger-than-memory data.
            If set to `False` (default), the entire query is processed in a single
            batch.

            .. warning::
                Streaming mode is considered **unstable**. It may be changed
                at any point without it being considered a breaking change.

            .. note::
                Use :func:`explain` to see if Polars can process the query in streaming
                mode.
        engine
            Select the engine used to process the query, optional.
            If set to `"cpu"` (default), the query is run using the
            polars CPU engine. If set to `"gpu"`, the GPU engine is
            used. Fine-grained control over the GPU engine, for
            example which device to use on a system with multiple
            devices, is possible by providing a :class:`~.GPUEngine` object
            with configuration options.

            .. note::
               GPU mode is considered **unstable**. Not all queries will run
               successfully on the GPU, however, they should fall back transparently
               to the default engine if execution is not supported.

               Running with `POLARS_VERBOSE=1` will provide information if a query
               falls back (and why).

            .. note::
               The GPU engine does not support streaming, or running in the
               background. If either are enabled, then GPU execution is switched off.

        background
            Run the query in the background and get a handle to the query.
            This handle can be used to fetch the result or cancel the query.

            .. warning::
                Background mode is considered **unstable**. It may be changed
                at any point without it being considered a breaking change.

        Returns
        -------
        DataFrame

        See Also
        --------
        explain : Print the query plan that is evaluated with collect.
        profile : Collect the LazyFrame and time each node in the computation graph.
        polars.collect_all : Collect multiple LazyFrames at the same time.
        polars.Config.set_streaming_chunk_size : Set the size of streaming batches.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a").agg(pl.all().sum()).collect()  # doctest: +SKIP
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 4   ┆ 10  │
        │ b   ┆ 11  ┆ 10  │
        │ c   ┆ 6   ┆ 1   │
        └─────┴─────┴─────┘

        Collect in streaming mode

        >>> lf.group_by("a").agg(pl.all().sum()).collect(
        ...     streaming=True
        ... )  # doctest: +SKIP
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 4   ┆ 10  │
        │ b   ┆ 11  ┆ 10  │
        │ c   ┆ 6   ┆ 1   │
        └─────┴─────┴─────┘

        Collect in GPU mode

        >>> lf.group_by("a").agg(pl.all().sum()).collect(engine="gpu")  # doctest: +SKIP
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 11  ┆ 10  │
        │ a   ┆ 4   ┆ 10  │
        │ c   ┆ 6   ┆ 1   │
        └─────┴─────┴─────┘

        With control over the device used

        >>> lf.group_by("a").agg(pl.all().sum()).collect(
        ...     engine=pl.GPUEngine(device=1)
        ... )  # doctest: +SKIP
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 11  ┆ 10  │
        │ a   ┆ 4   ┆ 10  │
        │ c   ┆ 6   ┆ 1   │
        └─────┴─────┴─────┘
        """
        ...

    @overload
    def collect_async(self, *, gevent: Literal[True], type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, no_optimization: bool=True, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, streaming: bool=True) -> _GeventDataFrameResult[DataFrame]:
        ...

    @overload
    def collect_async(self, *, gevent: Literal[False]=False, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, no_optimization: bool=True, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, streaming: bool=True) -> Awaitable[DataFrame]:
        ...

    def collect_async(self, *, gevent: bool=False, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, no_optimization: bool=False, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, streaming: bool=False) -> Awaitable[DataFrame] | _GeventDataFrameResult[DataFrame]:
        """
        Collect DataFrame asynchronously in thread pool.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Collects into a DataFrame (like :func:`collect`) but, instead of returning
        a DataFrame directly, it is scheduled to be collected inside a thread pool,
        while this method returns almost instantly.

        This can be useful if you use `gevent` or `asyncio` and want to release
        control to other greenlets/tasks while LazyFrames are being collected.

        Parameters
        ----------
        gevent
            Return wrapper to `gevent.event.AsyncResult` instead of Awaitable
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        no_optimization
            Turn off (certain) optimizations.
        slice_pushdown
            Slice pushdown optimization.
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        cluster_with_columns
            Combine sequential independent calls to with_columns
        collapse_joins
            Collapse a join and filters into a faster join
        streaming
            Process the query in batches to handle larger-than-memory data.
            If set to `False` (default), the entire query is processed in a single
            batch.

            .. warning::
                Streaming mode is considered **unstable**. It may be changed
                at any point without it being considered a breaking change.

            .. note::
                Use :func:`explain` to see if Polars can process the query in
                streaming mode.

        Returns
        -------
        If `gevent=False` (default) then returns an awaitable.

        If `gevent=True` then returns wrapper that has a
        `.get(block=True, timeout=None)` method.

        See Also
        --------
        polars.collect_all : Collect multiple LazyFrames at the same time.
        polars.collect_all_async : Collect multiple LazyFrames at the same time lazily.

        Notes
        -----
        In case of error `set_exception` is used on
        `asyncio.Future`/`gevent.event.AsyncResult` and will be reraised by them.

        Examples
        --------
        >>> import asyncio
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> async def main():
        ...     return await (
        ...         lf.group_by("a", maintain_order=True)
        ...         .agg(pl.all().sum())
        ...         .collect_async()
        ...     )
        >>> asyncio.run(main())
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 4   ┆ 10  │
        │ b   ┆ 11  ┆ 10  │
        │ c   ┆ 6   ┆ 1   │
        └─────┴─────┴─────┘
        """
        ...

    def collect_schema(self) -> Schema:
        """
        Resolve the schema of this LazyFrame.

        Examples
        --------
        Determine the schema.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.collect_schema()
        Schema({'foo': Int64, 'bar': Float64, 'ham': String})

        Access various properties of the schema.

        >>> schema = lf.collect_schema()
        >>> schema["bar"]
        Float64
        >>> schema.names()
        ['foo', 'bar', 'ham']
        >>> schema.dtypes()
        [Int64, Float64, String]
        >>> schema.len()
        3
        """
        ...

    @unstable()
    def sink_parquet(self, path: str | Path, *, compression: str='zstd', compression_level: int | None=None, statistics: bool | str | dict[str, bool]=True, row_group_size: int | None=None, data_page_size: int | None=None, maintain_order: bool=True, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, slice_pushdown: bool=True, collapse_joins: bool=True, no_optimization: bool=False) -> None:
        """
        Evaluate the query in streaming mode and write to a Parquet file.

        .. warning::
            Streaming mode is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        This allows streaming results that are larger than RAM to be written to disk.

        Parameters
        ----------
        path
            File path to which the file should be written.
        compression : {'lz4', 'uncompressed', 'snappy', 'gzip', 'lzo', 'brotli', 'zstd'}
            Choose "zstd" for good compression performance.
            Choose "lz4" for fast compression/decompression.
            Choose "snappy" for more backwards compatibility guarantees
            when you deal with older parquet readers.
        compression_level
            The level of compression to use. Higher compression means smaller files on
            disk.

            - "gzip" : min-level: 0, max-level: 10.
            - "brotli" : min-level: 0, max-level: 11.
            - "zstd" : min-level: 1, max-level: 22.
        statistics
            Write statistics to the parquet headers. This is the default behavior.

            Possible values:

            - `True`: enable default set of statistics (default)
            - `False`: disable all statistics
            - "full": calculate and write all available statistics. Cannot be
              combined with `use_pyarrow`.
            - `{ "statistic-key": True / False, ... }`. Cannot be combined with
              `use_pyarrow`. Available keys:

              - "min": column minimum value (default: `True`)
              - "max": column maximum value (default: `True`)
              - "distinct_count": number of unique column values (default: `False`)
              - "null_count": number of null values in column (default: `True`)
        row_group_size
            Size of the row groups in number of rows.
            If None (default), the chunks of the `DataFrame` are
            used. Writing in smaller chunks may reduce memory pressure and improve
            writing speeds.
        data_page_size
            Size limit of individual data pages.
            If not set defaults to 1024 * 1024 bytes
        maintain_order
            Maintain the order in which data is processed.
            Setting this to `False` will be slightly faster.
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        slice_pushdown
            Slice pushdown optimization.
        collapse_joins
            Collapse a join and filters into a faster join
        no_optimization
            Turn off (certain) optimizations.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> lf = pl.scan_csv("/path/to/my_larger_than_ram_file.csv")  # doctest: +SKIP
        >>> lf.sink_parquet("out.parquet")  # doctest: +SKIP
        """
        ...

    @unstable()
    def sink_ipc(self, path: str | Path, *, compression: str | None='zstd', maintain_order: bool=True, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, slice_pushdown: bool=True, collapse_joins: bool=True, no_optimization: bool=False) -> None:
        """
        Evaluate the query in streaming mode and write to an IPC file.

        .. warning::
            Streaming mode is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        This allows streaming results that are larger than RAM to be written to disk.

        Parameters
        ----------
        path
            File path to which the file should be written.
        compression : {'lz4', 'zstd'}
            Choose "zstd" for good compression performance.
            Choose "lz4" for fast compression/decompression.
        maintain_order
            Maintain the order in which data is processed.
            Setting this to `False` will be slightly faster.
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        slice_pushdown
            Slice pushdown optimization.
        collapse_joins
            Collapse a join and filters into a faster join
        no_optimization
            Turn off (certain) optimizations.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> lf = pl.scan_csv("/path/to/my_larger_than_ram_file.csv")  # doctest: +SKIP
        >>> lf.sink_ipc("out.arrow")  # doctest: +SKIP
        """
        ...

    @unstable()
    def sink_csv(self, path: str | Path, *, include_bom: bool=False, include_header: bool=True, separator: str=',', line_terminator: str='\n', quote_char: str='"', batch_size: int=1024, datetime_format: str | None=None, date_format: str | None=None, time_format: str | None=None, float_scientific: bool | None=None, float_precision: int | None=None, null_value: str | None=None, quote_style: CsvQuoteStyle | None=None, maintain_order: bool=True, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, slice_pushdown: bool=True, collapse_joins: bool=True, no_optimization: bool=False) -> None:
        """
        Evaluate the query in streaming mode and write to a CSV file.

        .. warning::
            Streaming mode is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        This allows streaming results that are larger than RAM to be written to disk.

        Parameters
        ----------
        path
            File path to which the file should be written.
        include_bom
            Whether to include UTF-8 BOM in the CSV output.
        include_header
            Whether to include header in the CSV output.
        separator
            Separate CSV fields with this symbol.
        line_terminator
            String used to end each row.
        quote_char
            Byte to use as quoting character.
        batch_size
            Number of rows that will be processed per thread.
        datetime_format
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate. If no format specified, the default fractional-second
            precision is inferred from the maximum timeunit found in the frame's
            Datetime cols (if any).
        date_format
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate.
        time_format
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate.
        float_scientific
            Whether to use scientific form always (true), never (false), or
            automatically (None) for `Float32` and `Float64` datatypes.
        float_precision
            Number of decimal places to write, applied to both `Float32` and
            `Float64` datatypes.
        null_value
            A string representing null values (defaulting to the empty string).
        quote_style : {'necessary', 'always', 'non_numeric', 'never'}
            Determines the quoting strategy used.

            - necessary (default): This puts quotes around fields only when necessary.
              They are necessary when fields contain a quote,
              delimiter or record terminator.
              Quotes are also necessary when writing an empty record
              (which is indistinguishable from a record with one empty field).
              This is the default.
            - always: This puts quotes around every field. Always.
            - never: This never puts quotes around fields, even if that results in
              invalid CSV data (e.g.: by not quoting strings containing the
              separator).
            - non_numeric: This puts quotes around all fields that are non-numeric.
              Namely, when writing a field that does not parse as a valid float
              or integer, then quotes will be used even if they aren`t strictly
              necessary.
        maintain_order
            Maintain the order in which data is processed.
            Setting this to `False` will be slightly faster.
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        slice_pushdown
            Slice pushdown optimization.
        collapse_joins
            Collapse a join and filters into a faster join
        no_optimization
            Turn off (certain) optimizations.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> lf = pl.scan_csv("/path/to/my_larger_than_ram_file.csv")  # doctest: +SKIP
        >>> lf.sink_csv("out.csv")  # doctest: +SKIP
        """
        ...

    @unstable()
    def sink_ndjson(self, path: str | Path, *, maintain_order: bool=True, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, slice_pushdown: bool=True, collapse_joins: bool=True, no_optimization: bool=False) -> None:
        """
        Evaluate the query in streaming mode and write to an NDJSON file.

        .. warning::
            Streaming mode is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        This allows streaming results that are larger than RAM to be written to disk.

        Parameters
        ----------
        path
            File path to which the file should be written.
        maintain_order
            Maintain the order in which data is processed.
            Setting this to `False` will be slightly faster.
        type_coercion
            Do type coercion optimization.
        predicate_pushdown
            Do predicate pushdown optimization.
        projection_pushdown
            Do projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        slice_pushdown
            Slice pushdown optimization.
        collapse_joins
            Collapse a join and filters into a faster join
        no_optimization
            Turn off (certain) optimizations.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> lf = pl.scan_csv("/path/to/my_larger_than_ram_file.csv")  # doctest: +SKIP
        >>> lf.sink_ndjson("out.ndjson")  # doctest: +SKIP
        """
        ...

    def _set_sink_optimizations(self, *, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, slice_pushdown: bool=True, collapse_joins: bool=True, no_optimization: bool=False) -> PyLazyFrame:
        ...

    @deprecate_function('`LazyFrame.fetch` is deprecated; use `LazyFrame.collect` instead, in conjunction with a call to `head`.', version='1.0')
    def fetch(self, n_rows: int=500, *, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, no_optimization: bool=False, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, streaming: bool=False) -> DataFrame:
        """
        Collect a small number of rows for debugging purposes.

        .. deprecated:: 1.0
            Use :meth:`collect` instead, in conjunction with a call to :meth:`head`.`

        Notes
        -----
        This is similar to a :func:`collect` operation, but it overwrites the number of
        rows read by *every* scan operation. Be aware that `fetch` does not guarantee
        the final number of rows in the DataFrame. Filters, join operations and fewer
        rows being available in the scanned data will all influence the final number
        of rows (joins are especially susceptible to this, and may return no data
        at all if `n_rows` is too small as the join keys may not be present).

        Warnings
        --------
        This is strictly a utility function that can help to debug queries using a
        smaller number of rows, and should *not* be used in production code.
        """
        ...

    def _fetch(self, n_rows: int=500, *, type_coercion: bool=True, predicate_pushdown: bool=True, projection_pushdown: bool=True, simplify_expression: bool=True, no_optimization: bool=False, slice_pushdown: bool=True, comm_subplan_elim: bool=True, comm_subexpr_elim: bool=True, cluster_with_columns: bool=True, collapse_joins: bool=True, streaming: bool=False) -> DataFrame:
        """
        Collect a small number of rows for debugging purposes.

        Do not confuse with `collect`; this function will frequently return
        incorrect data (see the warning for additional details).

        Parameters
        ----------
        n_rows
            Collect n_rows from the data sources.
        type_coercion
            Run type coercion optimization.
        predicate_pushdown
            Run predicate pushdown optimization.
        projection_pushdown
            Run projection pushdown optimization.
        simplify_expression
            Run simplify expressions optimization.
        no_optimization
            Turn off optimizations.
        slice_pushdown
            Slice pushdown optimization
        comm_subplan_elim
            Will try to cache branching subplans that occur on self-joins or unions.
        comm_subexpr_elim
            Common subexpressions will be cached and reused.
        cluster_with_columns
            Combine sequential independent calls to with_columns
        collapse_joins
            Collapse a join and filters into a faster join
        streaming
            Run parts of the query in a streaming fashion (this is in an alpha state)

            .. warning::
                Streaming mode is considered **unstable**. It may be changed
                at any point without it being considered a breaking change.

        Notes
        -----
        This is similar to a :func:`collect` operation, but it overwrites the number of
        rows read by *every* scan operation. Be aware that `fetch` does not guarantee
        the final number of rows in the DataFrame. Filters, join operations and fewer
        rows being available in the scanned data will all influence the final number
        of rows (joins are especially susceptible to this, and may return no data
        at all if `n_rows` is too small as the join keys may not be present).

        Warnings
        --------
        This is strictly a utility function that can help to debug queries using a
        smaller number of rows, and should *not* be used in production code.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [1, 2, 3, 4, 5, 6],
        ...         "c": [6, 5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a", maintain_order=True).agg(pl.all().sum())._fetch(2)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 6   │
        │ b   ┆ 2   ┆ 5   │
        └─────┴─────┴─────┘
        """
        ...

    def lazy(self) -> LazyFrame:
        """
        Return lazy representation, i.e. itself.

        Useful for writing code that expects either a :class:`DataFrame` or
        :class:`LazyFrame`. On LazyFrame this is a no-op, and returns the same object.

        Returns
        -------
        LazyFrame

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... )
        >>> lf.lazy()  # doctest: +ELLIPSIS
        <LazyFrame at ...>
        """
        ...

    def cache(self) -> LazyFrame:
        """
        Cache the result once the execution of the physical plan hits this node.

        It is not recommended using this as the optimizer likely can do a better job.
        """
        ...

    def cast(self, dtypes: Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType] | PolarsDataType, *, strict: bool=True) -> LazyFrame:
        """
        Cast LazyFrame column(s) to the specified dtype(s).

        Parameters
        ----------
        dtypes
            Mapping of column names (or selector) to dtypes, or a single dtype
            to which all columns will be cast.
        strict
            Throw an error if a cast could not be done (for instance, due to an
            overflow).

        Examples
        --------
        >>> from datetime import date
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": [date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)],
        ...     }
        ... )

        Cast specific frame columns to the specified dtypes:

        >>> lf.cast({"foo": pl.Float32, "bar": pl.UInt8}).collect()
        shape: (3, 3)
        ┌─────┬─────┬────────────┐
        │ foo ┆ bar ┆ ham        │
        │ --- ┆ --- ┆ ---        │
        │ f32 ┆ u8  ┆ date       │
        ╞═════╪═════╪════════════╡
        │ 1.0 ┆ 6   ┆ 2020-01-02 │
        │ 2.0 ┆ 7   ┆ 2021-03-04 │
        │ 3.0 ┆ 8   ┆ 2022-05-06 │
        └─────┴─────┴────────────┘

        Cast all frame columns matching one dtype (or dtype group) to another dtype:

        >>> lf.cast({pl.Date: pl.Datetime}).collect()
        shape: (3, 3)
        ┌─────┬─────┬─────────────────────┐
        │ foo ┆ bar ┆ ham                 │
        │ --- ┆ --- ┆ ---                 │
        │ i64 ┆ f64 ┆ datetime[μs]        │
        ╞═════╪═════╪═════════════════════╡
        │ 1   ┆ 6.0 ┆ 2020-01-02 00:00:00 │
        │ 2   ┆ 7.0 ┆ 2021-03-04 00:00:00 │
        │ 3   ┆ 8.0 ┆ 2022-05-06 00:00:00 │
        └─────┴─────┴─────────────────────┘

        Use selectors to define the columns being cast:

        >>> import polars.selectors as cs
        >>> lf.cast({cs.numeric(): pl.UInt32, cs.temporal(): pl.String}).collect()
        shape: (3, 3)
        ┌─────┬─────┬────────────┐
        │ foo ┆ bar ┆ ham        │
        │ --- ┆ --- ┆ ---        │
        │ u32 ┆ u32 ┆ str        │
        ╞═════╪═════╪════════════╡
        │ 1   ┆ 6   ┆ 2020-01-02 │
        │ 2   ┆ 7   ┆ 2021-03-04 │
        │ 3   ┆ 8   ┆ 2022-05-06 │
        └─────┴─────┴────────────┘

        Cast all frame columns to the specified dtype:

        >>> lf.cast(pl.String).collect().to_dict(as_series=False)
        {'foo': ['1', '2', '3'],
         'bar': ['6.0', '7.0', '8.0'],
         'ham': ['2020-01-02', '2021-03-04', '2022-05-06']}
        """
        ...

    def clear(self, n: int=0) -> LazyFrame:
        """
        Create an empty copy of the current LazyFrame, with zero to 'n' rows.

        Returns a copy with an identical schema but no data.

        Parameters
        ----------
        n
            Number of (empty) rows to return in the cleared frame.

        See Also
        --------
        clone : Cheap deepcopy/clone.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... )
        >>> lf.clear().collect()
        shape: (0, 3)
        ┌─────┬─────┬──────┐
        │ a   ┆ b   ┆ c    │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ f64 ┆ bool │
        ╞═════╪═════╪══════╡
        └─────┴─────┴──────┘

        >>> lf.clear(2).collect()
        shape: (2, 3)
        ┌──────┬──────┬──────┐
        │ a    ┆ b    ┆ c    │
        │ ---  ┆ ---  ┆ ---  │
        │ i64  ┆ f64  ┆ bool │
        ╞══════╪══════╪══════╡
        │ null ┆ null ┆ null │
        │ null ┆ null ┆ null │
        └──────┴──────┴──────┘
        """
        ...

    def clone(self) -> LazyFrame:
        """
        Create a copy of this LazyFrame.

        This is a cheap operation that does not copy data.

        See Also
        --------
        clear : Create an empty copy of the current LazyFrame, with identical
            schema but no data.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... )
        >>> lf.clone()  # doctest: +ELLIPSIS
        <LazyFrame at ...>
        """
        ...

    def filter(self, *predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool | list[bool] | np.ndarray[Any, Any], **constraints: Any) -> LazyFrame:
        """
        Filter the rows in the LazyFrame based on a predicate expression.

        The original order of the remaining rows is preserved.

        Rows where the filter does not evaluate to True are discarded, including nulls.

        Parameters
        ----------
        predicates
            Expression that evaluates to a boolean Series.
        constraints
            Column filters; use `name = value` to filter columns by the supplied value.
            Each constraint will behave the same as `pl.col(name).eq(value)`, and
            will be implicitly joined with the other filter conditions using `&`.

        Notes
        -----
        If you are transitioning from pandas and performing filter operations based on
        the comparison of two or more columns, please note that in Polars,
        any comparison involving null values will always result in null.
        As a result, these rows will be filtered out.
        Ensure to handle null values appropriately to avoid unintended filtering
        (See examples below).

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3, None, 4, None, 0],
        ...         "bar": [6, 7, 8, None, None, 9, 0],
        ...         "ham": ["a", "b", "c", None, "d", "e", "f"],
        ...     }
        ... )

        Filter on one condition:

        >>> lf.filter(pl.col("foo") > 1).collect()
        shape: (3, 3)
        ┌─────┬──────┬─────┐
        │ foo ┆ bar  ┆ ham │
        │ --- ┆ ---  ┆ --- │
        │ i64 ┆ i64  ┆ str │
        ╞═════╪══════╪═════╡
        │ 2   ┆ 7    ┆ b   │
        │ 3   ┆ 8    ┆ c   │
        │ 4   ┆ null ┆ d   │
        └─────┴──────┴─────┘

        Filter on multiple conditions:

        >>> lf.filter((pl.col("foo") < 3) & (pl.col("ham") == "a")).collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        Provide multiple filters using `*args` syntax:

        >>> lf.filter(
        ...     pl.col("foo") == 1,
        ...     pl.col("ham") == "a",
        ... ).collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        Provide multiple filters using `**kwargs` syntax:

        >>> lf.filter(foo=1, ham="a").collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        Filter on an OR condition:

        >>> lf.filter((pl.col("foo") == 1) | (pl.col("ham") == "c")).collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        Filter by comparing two columns against each other

        >>> lf.filter(pl.col("foo") == pl.col("bar")).collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 0   ┆ 0   ┆ f   │
        └─────┴─────┴─────┘

        >>> lf.filter(pl.col("foo") != pl.col("bar")).collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 2   ┆ 7   ┆ b   │
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        Notice how the row with `None` values is filtered out.
        In order to keep the same behavior as pandas, use:

        >>> lf.filter(pl.col("foo").ne_missing(pl.col("bar"))).collect()
        shape: (5, 3)
        ┌──────┬──────┬─────┐
        │ foo  ┆ bar  ┆ ham │
        │ ---  ┆ ---  ┆ --- │
        │ i64  ┆ i64  ┆ str │
        ╞══════╪══════╪═════╡
        │ 1    ┆ 6    ┆ a   │
        │ 2    ┆ 7    ┆ b   │
        │ 3    ┆ 8    ┆ c   │
        │ 4    ┆ null ┆ d   │
        │ null ┆ 9    ┆ e   │
        └──────┴──────┴─────┘
        """
        ...

    def select(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> LazyFrame:
        """
        Select columns from this LazyFrame.

        Parameters
        ----------
        *exprs
            Column(s) to select, specified as positional arguments.
            Accepts expression input. Strings are parsed as column names,
            other non-expression inputs are parsed as literals.
        **named_exprs
            Additional columns to select, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        Examples
        --------
        Pass the name of a column to select that column.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.select("foo").collect()
        shape: (3, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        │ 3   │
        └─────┘

        Multiple columns can be selected by passing a list of column names.

        >>> lf.select(["foo", "bar"]).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 6   │
        │ 2   ┆ 7   │
        │ 3   ┆ 8   │
        └─────┴─────┘

        Multiple columns can also be selected using positional arguments instead of a
        list. Expressions are also accepted.

        >>> lf.select(pl.col("foo"), pl.col("bar") + 1).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 7   │
        │ 2   ┆ 8   │
        │ 3   ┆ 9   │
        └─────┴─────┘

        Use keyword arguments to easily name your expression inputs.

        >>> lf.select(
        ...     threshold=pl.when(pl.col("foo") > 2).then(10).otherwise(0)
        ... ).collect()
        shape: (3, 1)
        ┌───────────┐
        │ threshold │
        │ ---       │
        │ i32       │
        ╞═══════════╡
        │ 0         │
        │ 0         │
        │ 10        │
        └───────────┘

        Expressions with multiple outputs can be automatically instantiated as Structs
        by enabling the setting `Config.set_auto_structify(True)`:

        >>> with pl.Config(auto_structify=True):
        ...     lf.select(
        ...         is_odd=(pl.col(pl.Int64) % 2 == 1).name.suffix("_is_odd"),
        ...     ).collect()
        shape: (3, 1)
        ┌──────────────┐
        │ is_odd       │
        │ ---          │
        │ struct[2]    │
        ╞══════════════╡
        │ {true,false} │
        │ {false,true} │
        │ {true,false} │
        └──────────────┘
        """
        ...

    def select_seq(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> LazyFrame:
        """
        Select columns from this LazyFrame.

        This will run all expression sequentially instead of in parallel.
        Use this when the work per expression is cheap.

        Parameters
        ----------
        *exprs
            Column(s) to select, specified as positional arguments.
            Accepts expression input. Strings are parsed as column names,
            other non-expression inputs are parsed as literals.
        **named_exprs
            Additional columns to select, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        See Also
        --------
        select
        """
        ...

    def group_by(self, *by: IntoExpr | Iterable[IntoExpr], maintain_order: bool=False, **named_by: IntoExpr) -> LazyGroupBy:
        """
        Start a group by operation.

        Parameters
        ----------
        *by
            Column(s) to group by. Accepts expression input. Strings are parsed as
            column names.
        maintain_order
            Ensure that the order of the groups is consistent with the input data.
            This is slower than a default group by.
            Setting this to `True` blocks the possibility
            to run on the streaming engine.
        **named_by
            Additional columns to group by, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        Examples
        --------
        Group by one column and call `agg` to compute the grouped sum of another
        column.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "c"],
        ...         "b": [1, 2, 1, 3, 3],
        ...         "c": [5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> lf.group_by("a").agg(pl.col("b").sum()).collect()  # doctest: +IGNORE_RESULT
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 2   │
        │ b   ┆ 5   │
        │ c   ┆ 3   │
        └─────┴─────┘

        Set `maintain_order=True` to ensure the order of the groups is consistent with
        the input.

        >>> lf.group_by("a", maintain_order=True).agg(pl.col("c")).collect()
        shape: (3, 2)
        ┌─────┬───────────┐
        │ a   ┆ c         │
        │ --- ┆ ---       │
        │ str ┆ list[i64] │
        ╞═════╪═══════════╡
        │ a   ┆ [5, 3]    │
        │ b   ┆ [4, 2]    │
        │ c   ┆ [1]       │
        └─────┴───────────┘

        Group by multiple columns by passing a list of column names.

        >>> lf.group_by(["a", "b"]).agg(pl.max("c")).collect()  # doctest: +SKIP
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 5   │
        │ b   ┆ 2   ┆ 4   │
        │ b   ┆ 3   ┆ 2   │
        │ c   ┆ 3   ┆ 1   │
        └─────┴─────┴─────┘

        Or use positional arguments to group by multiple columns in the same way.
        Expressions are also accepted.

        >>> lf.group_by("a", pl.col("b") // 2).agg(
        ...     pl.col("c").mean()
        ... ).collect()  # doctest: +SKIP
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ f64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 0   ┆ 4.0 │
        │ b   ┆ 1   ┆ 3.0 │
        │ c   ┆ 1   ┆ 1.0 │
        └─────┴─────┴─────┘
        """
        ...

    @deprecate_renamed_parameter('by', 'group_by', version='0.20.14')
    def rolling(self, index_column: IntoExpr, *, period: str | timedelta, offset: str | timedelta | None=None, closed: ClosedInterval='right', group_by: IntoExpr | Iterable[IntoExpr] | None=None) -> LazyGroupBy:
        """
        Create rolling groups based on a temporal or integer column.

        Different from a `group_by_dynamic` the windows are now determined by the
        individual values and are not of constant intervals. For constant intervals
        use :func:`LazyFrame.group_by_dynamic`.

        If you have a time series `<t_0, t_1, ..., t_n>`, then by default the
        windows created will be

            * (t_0 - period, t_0]
            * (t_1 - period, t_1]
            * ...
            * (t_n - period, t_n]

        whereas if you pass a non-default `offset`, then the windows will be

            * (t_0 + offset, t_0 + offset + period]
            * (t_1 + offset, t_1 + offset + period]
            * ...
            * (t_n + offset, t_n + offset + period]

        The `period` and `offset` arguments are created either from a timedelta, or
        by using the following string language:

        - 1ns   (1 nanosecond)
        - 1us   (1 microsecond)
        - 1ms   (1 millisecond)
        - 1s    (1 second)
        - 1m    (1 minute)
        - 1h    (1 hour)
        - 1d    (1 calendar day)
        - 1w    (1 calendar week)
        - 1mo   (1 calendar month)
        - 1q    (1 calendar quarter)
        - 1y    (1 calendar year)
        - 1i    (1 index count)

        Or combine them:
        "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

        By "calendar day", we mean the corresponding time on the next day (which may
        not be 24 hours, due to daylight savings). Similarly for "calendar week",
        "calendar month", "calendar quarter", and "calendar year".

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `group_by` is
            specified, then it must be sorted in ascending order within each group).

            In case of a rolling group by on indices, dtype needs to be one of
            {UInt32, UInt64, Int32, Int64}. Note that the first three get temporarily
            cast to Int64, so if performance matters use an Int64 column.
        period
            Length of the window - must be non-negative.
        offset
            Offset of the window. Default is `-period`.
        closed : {'right', 'left', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).
        group_by
            Also group by this column/these columns

        Returns
        -------
        LazyGroupBy
            Object you can call `.agg` on to aggregate by groups, the result
            of which will be sorted by `index_column` (but note that if `group_by`
            columns are passed, it will only be sorted within each group).

        See Also
        --------
        group_by_dynamic

        Examples
        --------
        >>> dates = [
        ...     "2020-01-01 13:45:48",
        ...     "2020-01-01 16:42:13",
        ...     "2020-01-01 16:45:09",
        ...     "2020-01-02 18:12:48",
        ...     "2020-01-03 19:45:32",
        ...     "2020-01-08 23:16:43",
        ... ]
        >>> df = pl.LazyFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]}).with_columns(
        ...     pl.col("dt").str.strptime(pl.Datetime).set_sorted()
        ... )
        >>> out = (
        ...     df.rolling(index_column="dt", period="2d")
        ...     .agg(
        ...         pl.sum("a").alias("sum_a"),
        ...         pl.min("a").alias("min_a"),
        ...         pl.max("a").alias("max_a"),
        ...     )
        ...     .collect()
        ... )
        >>> out
        shape: (6, 4)
        ┌─────────────────────┬───────┬───────┬───────┐
        │ dt                  ┆ sum_a ┆ min_a ┆ max_a │
        │ ---                 ┆ ---   ┆ ---   ┆ ---   │
        │ datetime[μs]        ┆ i64   ┆ i64   ┆ i64   │
        ╞═════════════════════╪═══════╪═══════╪═══════╡
        │ 2020-01-01 13:45:48 ┆ 3     ┆ 3     ┆ 3     │
        │ 2020-01-01 16:42:13 ┆ 10    ┆ 3     ┆ 7     │
        │ 2020-01-01 16:45:09 ┆ 15    ┆ 3     ┆ 7     │
        │ 2020-01-02 18:12:48 ┆ 24    ┆ 3     ┆ 9     │
        │ 2020-01-03 19:45:32 ┆ 11    ┆ 2     ┆ 9     │
        │ 2020-01-08 23:16:43 ┆ 1     ┆ 1     ┆ 1     │
        └─────────────────────┴───────┴───────┴───────┘
        """
        ...

    @deprecate_renamed_parameter('by', 'group_by', version='0.20.14')
    def group_by_dynamic(self, index_column: IntoExpr, *, every: str | timedelta, period: str | timedelta | None=None, offset: str | timedelta | None=None, include_boundaries: bool=False, closed: ClosedInterval='left', label: Label='left', group_by: IntoExpr | Iterable[IntoExpr] | None=None, start_by: StartBy='window') -> LazyGroupBy:
        """
        Group based on a time value (or index value of type Int32, Int64).

        Time windows are calculated and rows are assigned to windows. Different from a
        normal group by is that a row can be member of multiple groups.
        By default, the windows look like:

        - [start, start + period)
        - [start + every, start + every + period)
        - [start + 2*every, start + 2*every + period)
        - ...

        where `start` is determined by `start_by`, `offset`, `every`, and the earliest
        datapoint. See the `start_by` argument description for details.

        .. warning::
            The index column must be sorted in ascending order. If `group_by` is passed, then
            the index column must be sorted in ascending order within each group.

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `group_by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a dynamic group by on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        every
            interval of the window
        period
            length of the window, if None it will equal 'every'
        offset
            offset of the window, does not take effect if `start_by` is 'datapoint'.
            Defaults to zero.
        include_boundaries
            Add the lower and upper bound of the window to the "_lower_boundary" and
            "_upper_boundary" columns. This will impact performance because it's harder to
            parallelize
        closed : {'left', 'right', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).
        label : {'left', 'right', 'datapoint'}
            Define which label to use for the window:

            - 'left': lower boundary of the window
            - 'right': upper boundary of the window
            - 'datapoint': the first value of the index column in the given window.
              If you don't need the label to be at one of the boundaries, choose this
              option for maximum performance
        group_by
            Also group by this column/these columns
        start_by : {'window', 'datapoint', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
            The strategy to determine the start of the first window by.

            * 'window': Start by taking the earliest timestamp, truncating it with
              `every`, and then adding `offset`.
              Note that weekly windows start on Monday.
            * 'datapoint': Start from the first encountered data point.
            * a day of the week (only takes effect if `every` contains `'w'`):

              * 'monday': Start the window on the Monday before the first data point.
              * 'tuesday': Start the window on the Tuesday before the first data point.
              * ...
              * 'sunday': Start the window on the Sunday before the first data point.

              The resulting window is then shifted back until the earliest datapoint
              is in or in front of it.

        Returns
        -------
        LazyGroupBy
            Object you can call `.agg` on to aggregate by groups, the result
            of which will be sorted by `index_column` (but note that if `group_by` columns are
            passed, it will only be sorted within each group).

        See Also
        --------
        rolling

        Notes
        -----
        1) If you're coming from pandas, then

           .. code-block:: python

               # polars
               df.group_by_dynamic("ts", every="1d").agg(pl.col("value").sum())

           is equivalent to

           .. code-block:: python

               # pandas
               df.set_index("ts").resample("D")["value"].sum().reset_index()

           though note that, unlike pandas, polars doesn't add extra rows for empty
           windows. If you need `index_column` to be evenly spaced, then please combine
           with :func:`DataFrame.upsample`.

        2) The `every`, `period` and `offset` arguments are created with
           the following string language:

           - 1ns   (1 nanosecond)
           - 1us   (1 microsecond)
           - 1ms   (1 millisecond)
           - 1s    (1 second)
           - 1m    (1 minute)
           - 1h    (1 hour)
           - 1d    (1 calendar day)
           - 1w    (1 calendar week)
           - 1mo   (1 calendar month)
           - 1q    (1 calendar quarter)
           - 1y    (1 calendar year)
           - 1i    (1 index count)

           Or combine them:
           "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

           By "calendar day", we mean the corresponding time on the next day (which may
           not be 24 hours, due to daylight savings). Similarly for "calendar week",
           "calendar month", "calendar quarter", and "calendar year".

           In case of a group_by_dynamic on an integer column, the windows are defined by:

           - "1i"      # length 1
           - "10i"     # length 10

        Examples
        --------
        >>> from datetime import datetime
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "time": pl.datetime_range(
        ...             start=datetime(2021, 12, 16),
        ...             end=datetime(2021, 12, 16, 3),
        ...             interval="30m",
        ...             eager=True,
        ...         ),
        ...         "n": range(7),
        ...     }
        ... )
        >>> lf.collect()
        shape: (7, 2)
        ┌─────────────────────┬─────┐
        │ time                ┆ n   │
        │ ---                 ┆ --- │
        │ datetime[μs]        ┆ i64 │
        ╞═════════════════════╪═════╡
        │ 2021-12-16 00:00:00 ┆ 0   │
        │ 2021-12-16 00:30:00 ┆ 1   │
        │ 2021-12-16 01:00:00 ┆ 2   │
        │ 2021-12-16 01:30:00 ┆ 3   │
        │ 2021-12-16 02:00:00 ┆ 4   │
        │ 2021-12-16 02:30:00 ┆ 5   │
        │ 2021-12-16 03:00:00 ┆ 6   │
        └─────────────────────┴─────┘

        Group by windows of 1 hour.

        >>> lf.group_by_dynamic("time", every="1h", closed="right").agg(
        ...     pl.col("n")
        ... ).collect()
        shape: (4, 2)
        ┌─────────────────────┬───────────┐
        │ time                ┆ n         │
        │ ---                 ┆ ---       │
        │ datetime[μs]        ┆ list[i64] │
        ╞═════════════════════╪═══════════╡
        │ 2021-12-15 23:00:00 ┆ [0]       │
        │ 2021-12-16 00:00:00 ┆ [1, 2]    │
        │ 2021-12-16 01:00:00 ┆ [3, 4]    │
        │ 2021-12-16 02:00:00 ┆ [5, 6]    │
        └─────────────────────┴───────────┘

        The window boundaries can also be added to the aggregation result

        >>> lf.group_by_dynamic(
        ...     "time", every="1h", include_boundaries=True, closed="right"
        ... ).agg(pl.col("n").mean()).collect()
        shape: (4, 4)
        ┌─────────────────────┬─────────────────────┬─────────────────────┬─────┐
        │ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ n   │
        │ ---                 ┆ ---                 ┆ ---                 ┆ --- │
        │ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ f64 │
        ╞═════════════════════╪═════════════════════╪═════════════════════╪═════╡
        │ 2021-12-15 23:00:00 ┆ 2021-12-16 00:00:00 ┆ 2021-12-15 23:00:00 ┆ 0.0 │
        │ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ 1.5 │
        │ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ 3.5 │
        │ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ 5.5 │
        └─────────────────────┴─────────────────────┴─────────────────────┴─────┘

        When closed="left", the window excludes the right end of interval:
        [lower_bound, upper_bound)

        >>> lf.group_by_dynamic("time", every="1h", closed="left").agg(
        ...     pl.col("n")
        ... ).collect()
        shape: (4, 2)
        ┌─────────────────────┬───────────┐
        │ time                ┆ n         │
        │ ---                 ┆ ---       │
        │ datetime[μs]        ┆ list[i64] │
        ╞═════════════════════╪═══════════╡
        │ 2021-12-16 00:00:00 ┆ [0, 1]    │
        │ 2021-12-16 01:00:00 ┆ [2, 3]    │
        │ 2021-12-16 02:00:00 ┆ [4, 5]    │
        │ 2021-12-16 03:00:00 ┆ [6]       │
        └─────────────────────┴───────────┘

        When closed="both" the time values at the window boundaries belong to 2 groups.

        >>> lf.group_by_dynamic("time", every="1h", closed="both").agg(
        ...     pl.col("n")
        ... ).collect()
        shape: (4, 2)
        ┌─────────────────────┬───────────┐
        │ time                ┆ n         │
        │ ---                 ┆ ---       │
        │ datetime[μs]        ┆ list[i64] │
        ╞═════════════════════╪═══════════╡
        │ 2021-12-16 00:00:00 ┆ [0, 1, 2] │
        │ 2021-12-16 01:00:00 ┆ [2, 3, 4] │
        │ 2021-12-16 02:00:00 ┆ [4, 5, 6] │
        │ 2021-12-16 03:00:00 ┆ [6]       │
        └─────────────────────┴───────────┘

        Dynamic group bys can also be combined with grouping on normal keys

        >>> lf = lf.with_columns(groups=pl.Series(["a", "a", "a", "b", "b", "a", "a"]))
        >>> lf.collect()
        shape: (7, 3)
        ┌─────────────────────┬─────┬────────┐
        │ time                ┆ n   ┆ groups │
        │ ---                 ┆ --- ┆ ---    │
        │ datetime[μs]        ┆ i64 ┆ str    │
        ╞═════════════════════╪═════╪════════╡
        │ 2021-12-16 00:00:00 ┆ 0   ┆ a      │
        │ 2021-12-16 00:30:00 ┆ 1   ┆ a      │
        │ 2021-12-16 01:00:00 ┆ 2   ┆ a      │
        │ 2021-12-16 01:30:00 ┆ 3   ┆ b      │
        │ 2021-12-16 02:00:00 ┆ 4   ┆ b      │
        │ 2021-12-16 02:30:00 ┆ 5   ┆ a      │
        │ 2021-12-16 03:00:00 ┆ 6   ┆ a      │
        └─────────────────────┴─────┴────────┘
        >>> lf.group_by_dynamic(
        ...     "time",
        ...     every="1h",
        ...     closed="both",
        ...     group_by="groups",
        ...     include_boundaries=True,
        ... ).agg(pl.col("n")).collect()
        shape: (6, 5)
        ┌────────┬─────────────────────┬─────────────────────┬─────────────────────┬───────────┐
        │ groups ┆ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ n         │
        │ ---    ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---       │
        │ str    ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ list[i64] │
        ╞════════╪═════════════════════╪═════════════════════╪═════════════════════╪═══════════╡
        │ a      ┆ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ [0, 1, 2] │
        │ a      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ [2]       │
        │ a      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ [5, 6]    │
        │ a      ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 04:00:00 ┆ 2021-12-16 03:00:00 ┆ [6]       │
        │ b      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ [3, 4]    │
        │ b      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ [4]       │
        └────────┴─────────────────────┴─────────────────────┴─────────────────────┴───────────┘

        Dynamic group by on an index column

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "idx": pl.int_range(0, 6, eager=True),
        ...         "A": ["A", "A", "B", "B", "B", "C"],
        ...     }
        ... )
        >>> lf.group_by_dynamic(
        ...     "idx",
        ...     every="2i",
        ...     period="3i",
        ...     include_boundaries=True,
        ...     closed="right",
        ... ).agg(pl.col("A").alias("A_agg_list")).collect()
        shape: (4, 4)
        ┌─────────────────┬─────────────────┬─────┬─────────────────┐
        │ _lower_boundary ┆ _upper_boundary ┆ idx ┆ A_agg_list      │
        │ ---             ┆ ---             ┆ --- ┆ ---             │
        │ i64             ┆ i64             ┆ i64 ┆ list[str]       │
        ╞═════════════════╪═════════════════╪═════╪═════════════════╡
        │ -2              ┆ 1               ┆ -2  ┆ ["A", "A"]      │
        │ 0               ┆ 3               ┆ 0   ┆ ["A", "B", "B"] │
        │ 2               ┆ 5               ┆ 2   ┆ ["B", "B", "C"] │
        │ 4               ┆ 7               ┆ 4   ┆ ["C"]           │
        └─────────────────┴─────────────────┴─────┴─────────────────┘
        """
        ...

    def join_asof(self, other: LazyFrame, *, left_on: str | None | Expr=None, right_on: str | None | Expr=None, on: str | None | Expr=None, by_left: str | Sequence[str] | None=None, by_right: str | Sequence[str] | None=None, by: str | Sequence[str] | None=None, strategy: AsofJoinStrategy='backward', suffix: str='_right', tolerance: str | int | float | timedelta | None=None, allow_parallel: bool=True, force_parallel: bool=False, coalesce: bool=True) -> LazyFrame:
        """
        Perform an asof join.

        This is similar to a left-join except that we match on nearest key rather than
        equal keys.

        Both DataFrames must be sorted by the join_asof key.

        For each row in the left DataFrame:

          - A "backward" search selects the last row in the right DataFrame whose
            'on' key is less than or equal to the left's key.

          - A "forward" search selects the first row in the right DataFrame whose
            'on' key is greater than or equal to the left's key.

            A "nearest" search selects the last row in the right DataFrame whose value
            is nearest to the left's key. String keys are not currently supported for a
            nearest search.

        The default is "backward".

        Parameters
        ----------
        other
            Lazy DataFrame to join with.
        left_on
            Join column of the left DataFrame.
        right_on
            Join column of the right DataFrame.
        on
            Join column of both DataFrames. If set, `left_on` and `right_on` should be
            None.
        by
            Join on these columns before doing asof join.
        by_left
            Join on these columns before doing asof join.
        by_right
            Join on these columns before doing asof join.
        strategy : {'backward', 'forward', 'nearest'}
            Join strategy.
        suffix
            Suffix to append to columns with a duplicate name.
        tolerance
            Numeric tolerance. By setting this the join will only be done if the near
            keys are within this distance. If an asof join is done on columns of dtype
            "Date", "Datetime", "Duration" or "Time", use either a datetime.timedelta
            object or the following string language:

                - 1ns   (1 nanosecond)
                - 1us   (1 microsecond)
                - 1ms   (1 millisecond)
                - 1s    (1 second)
                - 1m    (1 minute)
                - 1h    (1 hour)
                - 1d    (1 calendar day)
                - 1w    (1 calendar week)
                - 1mo   (1 calendar month)
                - 1q    (1 calendar quarter)
                - 1y    (1 calendar year)

                Or combine them:
                "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

                By "calendar day", we mean the corresponding time on the next day
                (which may not be 24 hours, due to daylight savings). Similarly for
                "calendar week", "calendar month", "calendar quarter", and
                "calendar year".

        allow_parallel
            Allow the physical plan to optionally evaluate the computation of both
            DataFrames up to the join in parallel.
        force_parallel
            Force the physical plan to evaluate the computation of both DataFrames up to
            the join in parallel.
        coalesce
            Coalescing behavior (merging of `on` / `left_on` / `right_on` columns):

            - True: -> Always coalesce join columns.
            - False: -> Never coalesce join columns.

            Note that joining on any other expressions than `col`
            will turn off coalescing.

        Examples
        --------
        >>> from datetime import date
        >>> gdp = pl.LazyFrame(
        ...     {
        ...         "date": pl.date_range(
        ...             date(2016, 1, 1),
        ...             date(2020, 1, 1),
        ...             "1y",
        ...             eager=True,
        ...         ),
        ...         "gdp": [4164, 4411, 4566, 4696, 4827],
        ...     }
        ... )
        >>> gdp.collect()
        shape: (5, 2)
        ┌────────────┬──────┐
        │ date       ┆ gdp  │
        │ ---        ┆ ---  │
        │ date       ┆ i64  │
        ╞════════════╪══════╡
        │ 2016-01-01 ┆ 4164 │
        │ 2017-01-01 ┆ 4411 │
        │ 2018-01-01 ┆ 4566 │
        │ 2019-01-01 ┆ 4696 │
        │ 2020-01-01 ┆ 4827 │
        └────────────┴──────┘

        >>> population = pl.LazyFrame(
        ...     {
        ...         "date": [date(2016, 3, 1), date(2018, 8, 1), date(2019, 1, 1)],
        ...         "population": [82.19, 82.66, 83.12],
        ...     }
        ... ).sort("date")
        >>> population.collect()
        shape: (3, 2)
        ┌────────────┬────────────┐
        │ date       ┆ population │
        │ ---        ┆ ---        │
        │ date       ┆ f64        │
        ╞════════════╪════════════╡
        │ 2016-03-01 ┆ 82.19      │
        │ 2018-08-01 ┆ 82.66      │
        │ 2019-01-01 ┆ 83.12      │
        └────────────┴────────────┘

        Note how the dates don't quite match. If we join them using `join_asof` and
        `strategy='backward'`, then each date from `population` which doesn't have an
        exact match is matched with the closest earlier date from `gdp`:

        >>> population.join_asof(gdp, on="date", strategy="backward").collect()
        shape: (3, 3)
        ┌────────────┬────────────┬──────┐
        │ date       ┆ population ┆ gdp  │
        │ ---        ┆ ---        ┆ ---  │
        │ date       ┆ f64        ┆ i64  │
        ╞════════════╪════════════╪══════╡
        │ 2016-03-01 ┆ 82.19      ┆ 4164 │
        │ 2018-08-01 ┆ 82.66      ┆ 4566 │
        │ 2019-01-01 ┆ 83.12      ┆ 4696 │
        └────────────┴────────────┴──────┘

        Note how:

        - date `2016-03-01` from `population` is matched with `2016-01-01` from `gdp`;
        - date `2018-08-01` from `population` is matched with `2018-01-01` from `gdp`.

        You can verify this by passing `coalesce=False`:

        >>> population.join_asof(
        ...     gdp, on="date", strategy="backward", coalesce=False
        ... ).collect()
        shape: (3, 4)
        ┌────────────┬────────────┬────────────┬──────┐
        │ date       ┆ population ┆ date_right ┆ gdp  │
        │ ---        ┆ ---        ┆ ---        ┆ ---  │
        │ date       ┆ f64        ┆ date       ┆ i64  │
        ╞════════════╪════════════╪════════════╪══════╡
        │ 2016-03-01 ┆ 82.19      ┆ 2016-01-01 ┆ 4164 │
        │ 2018-08-01 ┆ 82.66      ┆ 2018-01-01 ┆ 4566 │
        │ 2019-01-01 ┆ 83.12      ┆ 2019-01-01 ┆ 4696 │
        └────────────┴────────────┴────────────┴──────┘

        If we instead use `strategy='forward'`, then each date from `population` which
        doesn't have an exact match is matched with the closest later date from `gdp`:

        >>> population.join_asof(gdp, on="date", strategy="forward").collect()
        shape: (3, 3)
        ┌────────────┬────────────┬──────┐
        │ date       ┆ population ┆ gdp  │
        │ ---        ┆ ---        ┆ ---  │
        │ date       ┆ f64        ┆ i64  │
        ╞════════════╪════════════╪══════╡
        │ 2016-03-01 ┆ 82.19      ┆ 4411 │
        │ 2018-08-01 ┆ 82.66      ┆ 4696 │
        │ 2019-01-01 ┆ 83.12      ┆ 4696 │
        └────────────┴────────────┴──────┘

        Note how:

        - date `2016-03-01` from `population` is matched with `2017-01-01` from `gdp`;
        - date `2018-08-01` from `population` is matched with `2019-01-01` from `gdp`.

        Finally, `strategy='nearest'` gives us a mix of the two results above, as each
        date from `population` which doesn't have an exact match is matched with the
        closest date from `gdp`, regardless of whether it's earlier or later:

        >>> population.join_asof(gdp, on="date", strategy="nearest").collect()
        shape: (3, 3)
        ┌────────────┬────────────┬──────┐
        │ date       ┆ population ┆ gdp  │
        │ ---        ┆ ---        ┆ ---  │
        │ date       ┆ f64        ┆ i64  │
        ╞════════════╪════════════╪══════╡
        │ 2016-03-01 ┆ 82.19      ┆ 4164 │
        │ 2018-08-01 ┆ 82.66      ┆ 4696 │
        │ 2019-01-01 ┆ 83.12      ┆ 4696 │
        └────────────┴────────────┴──────┘

        Note how:

        - date `2016-03-01` from `population` is matched with `2016-01-01` from `gdp`;
        - date `2018-08-01` from `population` is matched with `2019-01-01` from `gdp`.

        They `by` argument allows joining on another column first, before the asof join.
        In this example we join by `country` first, then asof join by date, as above.

        >>> gdp_dates = pl.date_range(  # fmt: skip
        ...     date(2016, 1, 1), date(2020, 1, 1), "1y", eager=True
        ... )
        >>> gdp2 = pl.LazyFrame(
        ...     {
        ...         "country": ["Germany"] * 5 + ["Netherlands"] * 5,
        ...         "date": pl.concat([gdp_dates, gdp_dates]),
        ...         "gdp": [4164, 4411, 4566, 4696, 4827, 784, 833, 914, 910, 909],
        ...     }
        ... ).sort("country", "date")
        >>>
        >>> gdp2.collect()
        shape: (10, 3)
        ┌─────────────┬────────────┬──────┐
        │ country     ┆ date       ┆ gdp  │
        │ ---         ┆ ---        ┆ ---  │
        │ str         ┆ date       ┆ i64  │
        ╞═════════════╪════════════╪══════╡
        │ Germany     ┆ 2016-01-01 ┆ 4164 │
        │ Germany     ┆ 2017-01-01 ┆ 4411 │
        │ Germany     ┆ 2018-01-01 ┆ 4566 │
        │ Germany     ┆ 2019-01-01 ┆ 4696 │
        │ Germany     ┆ 2020-01-01 ┆ 4827 │
        │ Netherlands ┆ 2016-01-01 ┆ 784  │
        │ Netherlands ┆ 2017-01-01 ┆ 833  │
        │ Netherlands ┆ 2018-01-01 ┆ 914  │
        │ Netherlands ┆ 2019-01-01 ┆ 910  │
        │ Netherlands ┆ 2020-01-01 ┆ 909  │
        └─────────────┴────────────┴──────┘
        >>> pop2 = pl.LazyFrame(
        ...     {
        ...         "country": ["Germany"] * 3 + ["Netherlands"] * 3,
        ...         "date": [
        ...             date(2016, 3, 1),
        ...             date(2018, 8, 1),
        ...             date(2019, 1, 1),
        ...             date(2016, 3, 1),
        ...             date(2018, 8, 1),
        ...             date(2019, 1, 1),
        ...         ],
        ...         "population": [82.19, 82.66, 83.12, 17.11, 17.32, 17.40],
        ...     }
        ... ).sort("country", "date")
        >>>
        >>> pop2.collect()
        shape: (6, 3)
        ┌─────────────┬────────────┬────────────┐
        │ country     ┆ date       ┆ population │
        │ ---         ┆ ---        ┆ ---        │
        │ str         ┆ date       ┆ f64        │
        ╞═════════════╪════════════╪════════════╡
        │ Germany     ┆ 2016-03-01 ┆ 82.19      │
        │ Germany     ┆ 2018-08-01 ┆ 82.66      │
        │ Germany     ┆ 2019-01-01 ┆ 83.12      │
        │ Netherlands ┆ 2016-03-01 ┆ 17.11      │
        │ Netherlands ┆ 2018-08-01 ┆ 17.32      │
        │ Netherlands ┆ 2019-01-01 ┆ 17.4       │
        └─────────────┴────────────┴────────────┘
        >>> pop2.join_asof(gdp2, by="country", on="date", strategy="nearest").collect()
        shape: (6, 4)
        ┌─────────────┬────────────┬────────────┬──────┐
        │ country     ┆ date       ┆ population ┆ gdp  │
        │ ---         ┆ ---        ┆ ---        ┆ ---  │
        │ str         ┆ date       ┆ f64        ┆ i64  │
        ╞═════════════╪════════════╪════════════╪══════╡
        │ Germany     ┆ 2016-03-01 ┆ 82.19      ┆ 4164 │
        │ Germany     ┆ 2018-08-01 ┆ 82.66      ┆ 4696 │
        │ Germany     ┆ 2019-01-01 ┆ 83.12      ┆ 4696 │
        │ Netherlands ┆ 2016-03-01 ┆ 17.11      ┆ 784  │
        │ Netherlands ┆ 2018-08-01 ┆ 17.32      ┆ 910  │
        │ Netherlands ┆ 2019-01-01 ┆ 17.4       ┆ 910  │
        └─────────────┴────────────┴────────────┴──────┘

        """
        ...

    def join(self, other: LazyFrame, on: str | Expr | Sequence[str | Expr] | None=None, how: JoinStrategy='inner', *, left_on: str | Expr | Sequence[str | Expr] | None=None, right_on: str | Expr | Sequence[str | Expr] | None=None, suffix: str='_right', validate: JoinValidation='m:m', join_nulls: bool=False, coalesce: bool | None=None, allow_parallel: bool=True, force_parallel: bool=False) -> LazyFrame:
        """
        Add a join operation to the Logical Plan.

        Parameters
        ----------
        other
            Lazy DataFrame to join with.
        on
            Join column of both DataFrames. If set, `left_on` and `right_on` should be
            None.
        how : {'inner', 'left', 'right', 'full', 'semi', 'anti', 'cross'}
            Join strategy.

            * *inner*
                Returns rows that have matching values in both tables
            * *left*
                Returns all rows from the left table, and the matched rows from the
                right table
            * *right*
                Returns all rows from the right table, and the matched rows from the
                left table
            * *full*
                Returns all rows when there is a match in either left or right table
            * *cross*
                Returns the Cartesian product of rows from both tables
            * *semi*
                Returns rows from the left table that have a match in the right table.
            * *anti*
                Returns rows from the left table that have no match in the right table.

            .. note::
                A left join preserves the row order of the left DataFrame.
        left_on
            Join column of the left DataFrame.
        right_on
            Join column of the right DataFrame.
        suffix
            Suffix to append to columns with a duplicate name.
        validate: {'m:m', 'm:1', '1:m', '1:1'}
            Checks if join is of specified type.

                * *many_to_many*
                    “m:m”: default, does not result in checks
                * *one_to_one*
                    “1:1”: check if join keys are unique in both left and right datasets
                * *one_to_many*
                    “1:m”: check if join keys are unique in left dataset
                * *many_to_one*
                    “m:1”: check if join keys are unique in right dataset

            .. note::
                This is currently not supported by the streaming engine.

        join_nulls
            Join on null values. By default null values will never produce matches.
        coalesce
            Coalescing behavior (merging of join columns).

            - None: -> join specific.
            - True: -> Always coalesce join columns.
            - False: -> Never coalesce join columns.

            Note that joining on any other expressions than `col`
            will turn off coalescing.
        allow_parallel
            Allow the physical plan to optionally evaluate the computation of both
            DataFrames up to the join in parallel.
        force_parallel
            Force the physical plan to evaluate the computation of both DataFrames up to
            the join in parallel.

        See Also
        --------
        join_asof

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> other_lf = pl.LazyFrame(
        ...     {
        ...         "apple": ["x", "y", "z"],
        ...         "ham": ["a", "b", "d"],
        ...     }
        ... )
        >>> lf.join(other_lf, on="ham").collect()
        shape: (2, 4)
        ┌─────┬─────┬─────┬───────┐
        │ foo ┆ bar ┆ ham ┆ apple │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ i64 ┆ f64 ┆ str ┆ str   │
        ╞═════╪═════╪═════╪═══════╡
        │ 1   ┆ 6.0 ┆ a   ┆ x     │
        │ 2   ┆ 7.0 ┆ b   ┆ y     │
        └─────┴─────┴─────┴───────┘
        >>> lf.join(other_lf, on="ham", how="full").collect()
        shape: (4, 5)
        ┌──────┬──────┬──────┬───────┬───────────┐
        │ foo  ┆ bar  ┆ ham  ┆ apple ┆ ham_right │
        │ ---  ┆ ---  ┆ ---  ┆ ---   ┆ ---       │
        │ i64  ┆ f64  ┆ str  ┆ str   ┆ str       │
        ╞══════╪══════╪══════╪═══════╪═══════════╡
        │ 1    ┆ 6.0  ┆ a    ┆ x     ┆ a         │
        │ 2    ┆ 7.0  ┆ b    ┆ y     ┆ b         │
        │ null ┆ null ┆ null ┆ z     ┆ d         │
        │ 3    ┆ 8.0  ┆ c    ┆ null  ┆ null      │
        └──────┴──────┴──────┴───────┴───────────┘
        >>> lf.join(other_lf, on="ham", how="left", coalesce=True).collect()
        shape: (3, 4)
        ┌─────┬─────┬─────┬───────┐
        │ foo ┆ bar ┆ ham ┆ apple │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ i64 ┆ f64 ┆ str ┆ str   │
        ╞═════╪═════╪═════╪═══════╡
        │ 1   ┆ 6.0 ┆ a   ┆ x     │
        │ 2   ┆ 7.0 ┆ b   ┆ y     │
        │ 3   ┆ 8.0 ┆ c   ┆ null  │
        └─────┴─────┴─────┴───────┘
        >>> lf.join(other_lf, on="ham", how="semi").collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6.0 ┆ a   │
        │ 2   ┆ 7.0 ┆ b   │
        └─────┴─────┴─────┘
        >>> lf.join(other_lf, on="ham", how="anti").collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘
        """
        ...

    @unstable()
    def join_where(self, other: LazyFrame, *predicates: Expr | Iterable[Expr], suffix: str='_right') -> LazyFrame:
        """
        Perform a join based on one or multiple (in)equality predicates.

        This performs an inner join, so only rows where all predicates are true
        are included in the result, and a row from either DataFrame may be included
        multiple times in the result.

        .. note::
            The row order of the input DataFrames is not preserved.

        .. warning::
            This functionality is experimental. It may be
            changed at any point without it being considered a breaking change.

        Parameters
        ----------
        other
            DataFrame to join with.
        *predicates
            (In)Equality condition to join the two tables on.
            When a column name occurs in both tables, the proper suffix must
            be applied in the predicate.
        suffix
            Suffix to append to columns with a duplicate name.

        Examples
        --------
        >>> east = pl.LazyFrame(
        ...     {
        ...         "id": [100, 101, 102],
        ...         "dur": [120, 140, 160],
        ...         "rev": [12, 14, 16],
        ...         "cores": [2, 8, 4],
        ...     }
        ... )
        >>> west = pl.LazyFrame(
        ...     {
        ...         "t_id": [404, 498, 676, 742],
        ...         "time": [90, 130, 150, 170],
        ...         "cost": [9, 13, 15, 16],
        ...         "cores": [4, 2, 1, 4],
        ...     }
        ... )
        >>> east.join_where(
        ...     west,
        ...     pl.col("dur") < pl.col("time"),
        ...     pl.col("rev") < pl.col("cost"),
        ... ).collect()
        shape: (5, 8)
        ┌─────┬─────┬─────┬───────┬──────┬──────┬──────┬─────────────┐
        │ id  ┆ dur ┆ rev ┆ cores ┆ t_id ┆ time ┆ cost ┆ cores_right │
        │ --- ┆ --- ┆ --- ┆ ---   ┆ ---  ┆ ---  ┆ ---  ┆ ---         │
        │ i64 ┆ i64 ┆ i64 ┆ i64   ┆ i64  ┆ i64  ┆ i64  ┆ i64         │
        ╞═════╪═════╪═════╪═══════╪══════╪══════╪══════╪═════════════╡
        │ 100 ┆ 120 ┆ 12  ┆ 2     ┆ 498  ┆ 130  ┆ 13   ┆ 2           │
        │ 100 ┆ 120 ┆ 12  ┆ 2     ┆ 676  ┆ 150  ┆ 15   ┆ 1           │
        │ 100 ┆ 120 ┆ 12  ┆ 2     ┆ 742  ┆ 170  ┆ 16   ┆ 4           │
        │ 101 ┆ 140 ┆ 14  ┆ 8     ┆ 676  ┆ 150  ┆ 15   ┆ 1           │
        │ 101 ┆ 140 ┆ 14  ┆ 8     ┆ 742  ┆ 170  ┆ 16   ┆ 4           │
        └─────┴─────┴─────┴───────┴──────┴──────┴──────┴─────────────┘

        """
        ...

    def with_columns(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> LazyFrame:
        """
        Add columns to this LazyFrame.

        Added columns will replace existing columns with the same name.

        Parameters
        ----------
        *exprs
            Column(s) to add, specified as positional arguments.
            Accepts expression input. Strings are parsed as column names, other
            non-expression inputs are parsed as literals.
        **named_exprs
            Additional columns to add, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        Returns
        -------
        LazyFrame
            A new LazyFrame with the columns added.

        Notes
        -----
        Creating a new LazyFrame using this method does not create a new copy of
        existing data.

        Examples
        --------
        Pass an expression to add it as a new column.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> lf.with_columns((pl.col("a") ** 2).alias("a^2")).collect()
        shape: (4, 4)
        ┌─────┬──────┬───────┬─────┐
        │ a   ┆ b    ┆ c     ┆ a^2 │
        │ --- ┆ ---  ┆ ---   ┆ --- │
        │ i64 ┆ f64  ┆ bool  ┆ i64 │
        ╞═════╪══════╪═══════╪═════╡
        │ 1   ┆ 0.5  ┆ true  ┆ 1   │
        │ 2   ┆ 4.0  ┆ true  ┆ 4   │
        │ 3   ┆ 10.0 ┆ false ┆ 9   │
        │ 4   ┆ 13.0 ┆ true  ┆ 16  │
        └─────┴──────┴───────┴─────┘

        Added columns will replace existing columns with the same name.

        >>> lf.with_columns(pl.col("a").cast(pl.Float64)).collect()
        shape: (4, 3)
        ┌─────┬──────┬───────┐
        │ a   ┆ b    ┆ c     │
        │ --- ┆ ---  ┆ ---   │
        │ f64 ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╡
        │ 1.0 ┆ 0.5  ┆ true  │
        │ 2.0 ┆ 4.0  ┆ true  │
        │ 3.0 ┆ 10.0 ┆ false │
        │ 4.0 ┆ 13.0 ┆ true  │
        └─────┴──────┴───────┘

        Multiple columns can be added using positional arguments.

        >>> lf.with_columns(
        ...     (pl.col("a") ** 2).alias("a^2"),
        ...     (pl.col("b") / 2).alias("b/2"),
        ...     (pl.col("c").not_()).alias("not c"),
        ... ).collect()
        shape: (4, 6)
        ┌─────┬──────┬───────┬─────┬──────┬───────┐
        │ a   ┆ b    ┆ c     ┆ a^2 ┆ b/2  ┆ not c │
        │ --- ┆ ---  ┆ ---   ┆ --- ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ bool  ┆ i64 ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╪═════╪══════╪═══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ 1   ┆ 0.25 ┆ false │
        │ 2   ┆ 4.0  ┆ true  ┆ 4   ┆ 2.0  ┆ false │
        │ 3   ┆ 10.0 ┆ false ┆ 9   ┆ 5.0  ┆ true  │
        │ 4   ┆ 13.0 ┆ true  ┆ 16  ┆ 6.5  ┆ false │
        └─────┴──────┴───────┴─────┴──────┴───────┘

        Multiple columns can also be added by passing a list of expressions.

        >>> lf.with_columns(
        ...     [
        ...         (pl.col("a") ** 2).alias("a^2"),
        ...         (pl.col("b") / 2).alias("b/2"),
        ...         (pl.col("c").not_()).alias("not c"),
        ...     ]
        ... ).collect()
        shape: (4, 6)
        ┌─────┬──────┬───────┬─────┬──────┬───────┐
        │ a   ┆ b    ┆ c     ┆ a^2 ┆ b/2  ┆ not c │
        │ --- ┆ ---  ┆ ---   ┆ --- ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ bool  ┆ i64 ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╪═════╪══════╪═══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ 1   ┆ 0.25 ┆ false │
        │ 2   ┆ 4.0  ┆ true  ┆ 4   ┆ 2.0  ┆ false │
        │ 3   ┆ 10.0 ┆ false ┆ 9   ┆ 5.0  ┆ true  │
        │ 4   ┆ 13.0 ┆ true  ┆ 16  ┆ 6.5  ┆ false │
        └─────┴──────┴───────┴─────┴──────┴───────┘

        Use keyword arguments to easily name your expression inputs.

        >>> lf.with_columns(
        ...     ab=pl.col("a") * pl.col("b"),
        ...     not_c=pl.col("c").not_(),
        ... ).collect()
        shape: (4, 5)
        ┌─────┬──────┬───────┬──────┬───────┐
        │ a   ┆ b    ┆ c     ┆ ab   ┆ not_c │
        │ --- ┆ ---  ┆ ---   ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ bool  ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╪══════╪═══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ 0.5  ┆ false │
        │ 2   ┆ 4.0  ┆ true  ┆ 8.0  ┆ false │
        │ 3   ┆ 10.0 ┆ false ┆ 30.0 ┆ true  │
        │ 4   ┆ 13.0 ┆ true  ┆ 52.0 ┆ false │
        └─────┴──────┴───────┴──────┴───────┘

        Expressions with multiple outputs can automatically be instantiated as Structs
        by enabling the experimental setting `Config.set_auto_structify(True)`:

        >>> with pl.Config(auto_structify=True):
        ...     lf.drop("c").with_columns(
        ...         diffs=pl.col(["a", "b"]).diff().name.suffix("_diff"),
        ...     ).collect()
        shape: (4, 3)
        ┌─────┬──────┬─────────────┐
        │ a   ┆ b    ┆ diffs       │
        │ --- ┆ ---  ┆ ---         │
        │ i64 ┆ f64  ┆ struct[2]   │
        ╞═════╪══════╪═════════════╡
        │ 1   ┆ 0.5  ┆ {null,null} │
        │ 2   ┆ 4.0  ┆ {1,3.5}     │
        │ 3   ┆ 10.0 ┆ {1,6.0}     │
        │ 4   ┆ 13.0 ┆ {1,3.0}     │
        └─────┴──────┴─────────────┘
        """
        ...

    def with_columns_seq(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> LazyFrame:
        """
        Add columns to this LazyFrame.

        Added columns will replace existing columns with the same name.

        This will run all expression sequentially instead of in parallel.
        Use this when the work per expression is cheap.

        Parameters
        ----------
        *exprs
            Column(s) to add, specified as positional arguments.
            Accepts expression input. Strings are parsed as column names, other
            non-expression inputs are parsed as literals.
        **named_exprs
            Additional columns to add, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        Returns
        -------
        LazyFrame
            A new LazyFrame with the columns added.

        See Also
        --------
        with_columns
        """
        ...

    @deprecate_function("Use `pl.concat(..., how='horizontal')` instead.", version='1.0.0')
    def with_context(self, other: Self | list[Self]) -> LazyFrame:
        """
        Add an external context to the computation graph.

        .. deprecated:: 1.0.0
            Use :func:`concat` instead with `how='horizontal'`

        This allows expressions to also access columns from DataFrames
        that are not part of this one.

        Parameters
        ----------
        other
            Lazy DataFrame to join with.

        Examples
        --------
        >>> lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "c", None]})
        >>> lf_other = pl.LazyFrame({"c": ["foo", "ham"]})
        >>> lf.with_context(lf_other).select(  # doctest: +SKIP
        ...     pl.col("b") + pl.col("c").first()
        ... ).collect()
        shape: (3, 1)
        ┌──────┐
        │ b    │
        │ ---  │
        │ str  │
        ╞══════╡
        │ afoo │
        │ cfoo │
        │ null │
        └──────┘

        Fill nulls with the median from another DataFrame:

        >>> train_lf = pl.LazyFrame(
        ...     {"feature_0": [-1.0, 0, 1], "feature_1": [-1.0, 0, 1]}
        ... )
        >>> test_lf = pl.LazyFrame(
        ...     {"feature_0": [-1.0, None, 1], "feature_1": [-1.0, 0, 1]}
        ... )
        >>> test_lf.with_context(  # doctest: +SKIP
        ...     train_lf.select(pl.all().name.suffix("_train"))
        ... ).select(
        ...     pl.col("feature_0").fill_null(pl.col("feature_0_train").median())
        ... ).collect()
        shape: (3, 1)
        ┌───────────┐
        │ feature_0 │
        │ ---       │
        │ f64       │
        ╞═══════════╡
        │ -1.0      │
        │ 0.0       │
        │ 1.0       │
        └───────────┘
        """
        ...

    def drop(self, *columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector], strict: bool=True) -> LazyFrame:
        """
        Remove columns from the DataFrame.

        Parameters
        ----------
        *columns
            Names of the columns that should be removed from the dataframe.
            Accepts column selector input.
        strict
            Validate that all column names exist in the current schema,
            and throw an exception if any do not.

        Examples
        --------
        Drop a single column by passing the name of that column.

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.drop("ham").collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 1   ┆ 6.0 │
        │ 2   ┆ 7.0 │
        │ 3   ┆ 8.0 │
        └─────┴─────┘

        Drop multiple columns by passing a selector.

        >>> import polars.selectors as cs
        >>> lf.drop(cs.numeric()).collect()
        shape: (3, 1)
        ┌─────┐
        │ ham │
        │ --- │
        │ str │
        ╞═════╡
        │ a   │
        │ b   │
        │ c   │
        └─────┘

        Use positional arguments to drop multiple columns.

        >>> lf.drop("foo", "ham").collect()
        shape: (3, 1)
        ┌─────┐
        │ bar │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 6.0 │
        │ 7.0 │
        │ 8.0 │
        └─────┘
        """
        ...

    def rename(self, mapping: dict[str, str] | Callable[[str], str], *, strict: bool=True) -> LazyFrame:
        """
        Rename column names.

        Parameters
        ----------
        mapping
            Key value pairs that map from old name to new name, or a function
            that takes the old name as input and returns the new name.
        strict
            Validate that all column names exist in the current schema,
            and throw an exception if any do not. (Note that this parameter
            is a no-op when passing a function to `mapping`).

        Notes
        -----
        If existing names are swapped (e.g. 'A' points to 'B' and 'B' points to 'A'),
        polars will block projection and predicate pushdowns at this node.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.rename({"foo": "apple"}).collect()
        shape: (3, 3)
        ┌───────┬─────┬─────┐
        │ apple ┆ bar ┆ ham │
        │ ---   ┆ --- ┆ --- │
        │ i64   ┆ i64 ┆ str │
        ╞═══════╪═════╪═════╡
        │ 1     ┆ 6   ┆ a   │
        │ 2     ┆ 7   ┆ b   │
        │ 3     ┆ 8   ┆ c   │
        └───────┴─────┴─────┘
        >>> lf.rename(lambda column_name: "c" + column_name[1:]).collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ coo ┆ car ┆ cam │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 2   ┆ 7   ┆ b   │
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘
        """
        ...

    def reverse(self) -> LazyFrame:
        """
        Reverse the DataFrame.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "key": ["a", "b", "c"],
        ...         "val": [1, 2, 3],
        ...     }
        ... )
        >>> lf.reverse().collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ key ┆ val │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ c   ┆ 3   │
        │ b   ┆ 2   │
        │ a   ┆ 1   │
        └─────┴─────┘
        """
        ...

    def shift(self, n: int | IntoExprColumn=1, *, fill_value: IntoExpr | None=None) -> LazyFrame:
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

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [5, 6, 7, 8],
        ...     }
        ... )
        >>> lf.shift().collect()
        shape: (4, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ i64  ┆ i64  │
        ╞══════╪══════╡
        │ null ┆ null │
        │ 1    ┆ 5    │
        │ 2    ┆ 6    │
        │ 3    ┆ 7    │
        └──────┴──────┘

        Pass a negative value to shift in the opposite direction instead.

        >>> lf.shift(-2).collect()
        shape: (4, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ i64  ┆ i64  │
        ╞══════╪══════╡
        │ 3    ┆ 7    │
        │ 4    ┆ 8    │
        │ null ┆ null │
        │ null ┆ null │
        └──────┴──────┘

        Specify `fill_value` to fill the resulting null values.

        >>> lf.shift(-2, fill_value=100).collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 3   ┆ 7   │
        │ 4   ┆ 8   │
        │ 100 ┆ 100 │
        │ 100 ┆ 100 │
        └─────┴─────┘
        """
        ...

    def slice(self, offset: int, length: int | None=None) -> LazyFrame:
        """
        Get a slice of this DataFrame.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to `None`, all rows starting at the offset
            will be selected.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["x", "y", "z"],
        ...         "b": [1, 3, 5],
        ...         "c": [2, 4, 6],
        ...     }
        ... )
        >>> lf.slice(1, 2).collect()
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ y   ┆ 3   ┆ 4   │
        │ z   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┘
        """
        ...

    def limit(self, n: int=5) -> LazyFrame:
        """
        Get the first `n` rows.

        Alias for :func:`LazyFrame.head`.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5, 6],
        ...         "b": [7, 8, 9, 10, 11, 12],
        ...     }
        ... )
        >>> lf.limit().collect()
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 7   │
        │ 2   ┆ 8   │
        │ 3   ┆ 9   │
        │ 4   ┆ 10  │
        │ 5   ┆ 11  │
        └─────┴─────┘
        >>> lf.limit(2).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 7   │
        │ 2   ┆ 8   │
        └─────┴─────┘
        """
        ...

    def head(self, n: int=5) -> LazyFrame:
        """
        Get the first `n` rows.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5, 6],
        ...         "b": [7, 8, 9, 10, 11, 12],
        ...     }
        ... )
        >>> lf.head().collect()
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 7   │
        │ 2   ┆ 8   │
        │ 3   ┆ 9   │
        │ 4   ┆ 10  │
        │ 5   ┆ 11  │
        └─────┴─────┘
        >>> lf.head(2).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 7   │
        │ 2   ┆ 8   │
        └─────┴─────┘
        """
        ...

    def tail(self, n: int=5) -> LazyFrame:
        """
        Get the last `n` rows.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5, 6],
        ...         "b": [7, 8, 9, 10, 11, 12],
        ...     }
        ... )
        >>> lf.tail().collect()
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 8   │
        │ 3   ┆ 9   │
        │ 4   ┆ 10  │
        │ 5   ┆ 11  │
        │ 6   ┆ 12  │
        └─────┴─────┘
        >>> lf.tail(2).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 5   ┆ 11  │
        │ 6   ┆ 12  │
        └─────┴─────┘
        """
        ...

    def last(self) -> LazyFrame:
        """
        Get the last row of the DataFrame.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 5, 3],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> lf.last().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 3   ┆ 6   │
        └─────┴─────┘
        """
        ...

    def first(self) -> LazyFrame:
        """
        Get the first row of the DataFrame.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> lf.first().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 2   │
        └─────┴─────┘
        """
        ...

    @deprecate_function('Use `select(pl.all().approx_n_unique())` instead.', version='0.20.11')
    def approx_n_unique(self) -> LazyFrame:
        """
        Approximate count of unique values.

        .. deprecated:: 0.20.11
            Use `select(pl.all().approx_n_unique())` instead.

        This is done using the HyperLogLog++ algorithm for cardinality estimation.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.approx_n_unique().collect()  # doctest: +SKIP
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u32 ┆ u32 │
        ╞═════╪═════╡
        │ 4   ┆ 2   │
        └─────┴─────┘
        """
        ...

    def with_row_index(self, name: str='index', offset: int=0) -> LazyFrame:
        """
        Add a row index as the first column in the LazyFrame.

        Parameters
        ----------
        name
            Name of the index column.
        offset
            Start the index at this offset. Cannot be negative.

        Warnings
        --------
        Using this function can have a negative effect on query performance.
        This may, for instance, block predicate pushdown optimization.

        Notes
        -----
        The resulting column does not have any special properties. It is a regular
        column of type `UInt32` (or `UInt64` in `polars-u64-idx`).

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> lf.with_row_index().collect()
        shape: (3, 3)
        ┌───────┬─────┬─────┐
        │ index ┆ a   ┆ b   │
        │ ---   ┆ --- ┆ --- │
        │ u32   ┆ i64 ┆ i64 │
        ╞═══════╪═════╪═════╡
        │ 0     ┆ 1   ┆ 2   │
        │ 1     ┆ 3   ┆ 4   │
        │ 2     ┆ 5   ┆ 6   │
        └───────┴─────┴─────┘
        >>> lf.with_row_index("id", offset=1000).collect()
        shape: (3, 3)
        ┌──────┬─────┬─────┐
        │ id   ┆ a   ┆ b   │
        │ ---  ┆ --- ┆ --- │
        │ u32  ┆ i64 ┆ i64 │
        ╞══════╪═════╪═════╡
        │ 1000 ┆ 1   ┆ 2   │
        │ 1001 ┆ 3   ┆ 4   │
        │ 1002 ┆ 5   ┆ 6   │
        └──────┴─────┴─────┘

        An index column can also be created using the expressions :func:`int_range`
        and :func:`len`.

        >>> lf.select(
        ...     pl.int_range(pl.len(), dtype=pl.UInt32).alias("index"),
        ...     pl.all(),
        ... ).collect()
        shape: (3, 3)
        ┌───────┬─────┬─────┐
        │ index ┆ a   ┆ b   │
        │ ---   ┆ --- ┆ --- │
        │ u32   ┆ i64 ┆ i64 │
        ╞═══════╪═════╪═════╡
        │ 0     ┆ 1   ┆ 2   │
        │ 1     ┆ 3   ┆ 4   │
        │ 2     ┆ 5   ┆ 6   │
        └───────┴─────┴─────┘
        """
        ...

    @deprecate_function("Use `with_row_index` instead. Note that the default column name has changed from 'row_nr' to 'index'.", version='0.20.4')
    def with_row_count(self, name: str='row_nr', offset: int=0) -> LazyFrame:
        """
        Add a column at index 0 that counts the rows.

        .. deprecated:: 0.20.4
            Use :meth:`with_row_index` instead.
            Note that the default column name has changed from 'row_nr' to 'index'.

        Parameters
        ----------
        name
            Name of the column to add.
        offset
            Start the row count at this offset.

        Warnings
        --------
        This can have a negative effect on query performance.
        This may, for instance, block predicate pushdown optimization.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> lf.with_row_count().collect()  # doctest: +SKIP
        shape: (3, 3)
        ┌────────┬─────┬─────┐
        │ row_nr ┆ a   ┆ b   │
        │ ---    ┆ --- ┆ --- │
        │ u32    ┆ i64 ┆ i64 │
        ╞════════╪═════╪═════╡
        │ 0      ┆ 1   ┆ 2   │
        │ 1      ┆ 3   ┆ 4   │
        │ 2      ┆ 5   ┆ 6   │
        └────────┴─────┴─────┘
        """
        ...

    def gather_every(self, n: int, offset: int=0) -> LazyFrame:
        """
        Take every nth row in the LazyFrame and return as a new LazyFrame.

        Parameters
        ----------
        n
            Gather every *n*-th row.
        offset
            Starting index.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [5, 6, 7, 8],
        ...     }
        ... )
        >>> lf.gather_every(2).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 5   │
        │ 3   ┆ 7   │
        └─────┴─────┘
        >>> lf.gather_every(2, offset=1).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 6   │
        │ 4   ┆ 8   │
        └─────┴─────┘
        """
        ...

    def fill_null(self, value: Any | Expr | None=None, strategy: FillNullStrategy | None=None, limit: int | None=None, *, matches_supertype: bool=True) -> LazyFrame:
        """
        Fill null values using the specified value or strategy.

        Parameters
        ----------
        value
            Value used to fill null values.
        strategy : {None, 'forward', 'backward', 'min', 'max', 'mean', 'zero', 'one'}
            Strategy used to fill null values.
        limit
            Number of consecutive null values to fill when using the 'forward' or
            'backward' strategy.
        matches_supertype
            Fill all matching supertypes of the fill `value` literal.

        See Also
        --------
        fill_nan

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, None, 4],
        ...         "b": [0.5, 4, None, 13],
        ...     }
        ... )
        >>> lf.fill_null(99).collect()
        shape: (4, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ f64  │
        ╞═════╪══════╡
        │ 1   ┆ 0.5  │
        │ 2   ┆ 4.0  │
        │ 99  ┆ 99.0 │
        │ 4   ┆ 13.0 │
        └─────┴──────┘
        >>> lf.fill_null(strategy="forward").collect()
        shape: (4, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ f64  │
        ╞═════╪══════╡
        │ 1   ┆ 0.5  │
        │ 2   ┆ 4.0  │
        │ 2   ┆ 4.0  │
        │ 4   ┆ 13.0 │
        └─────┴──────┘

        >>> lf.fill_null(strategy="max").collect()
        shape: (4, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ f64  │
        ╞═════╪══════╡
        │ 1   ┆ 0.5  │
        │ 2   ┆ 4.0  │
        │ 4   ┆ 13.0 │
        │ 4   ┆ 13.0 │
        └─────┴──────┘

        >>> lf.fill_null(strategy="zero").collect()
        shape: (4, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ f64  │
        ╞═════╪══════╡
        │ 1   ┆ 0.5  │
        │ 2   ┆ 4.0  │
        │ 0   ┆ 0.0  │
        │ 4   ┆ 13.0 │
        └─────┴──────┘
        """
        ...

    def fill_nan(self, value: int | float | Expr | None) -> LazyFrame:
        """
        Fill floating point NaN values.

        Parameters
        ----------
        value
            Value to fill the NaN values with.

        Warnings
        --------
        Note that floating point NaNs (Not a Number) are not missing values.
        To replace missing values, use :func:`fill_null`.

        See Also
        --------
        fill_null

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1.5, 2, float("nan"), 4],
        ...         "b": [0.5, 4, float("nan"), 13],
        ...     }
        ... )
        >>> lf.fill_nan(99).collect()
        shape: (4, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ f64  ┆ f64  │
        ╞══════╪══════╡
        │ 1.5  ┆ 0.5  │
        │ 2.0  ┆ 4.0  │
        │ 99.0 ┆ 99.0 │
        │ 4.0  ┆ 13.0 │
        └──────┴──────┘
        """
        ...

    def std(self, ddof: int=1) -> LazyFrame:
        """
        Aggregate the columns in the LazyFrame to their standard deviation value.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.std().collect()
        shape: (1, 2)
        ┌──────────┬─────┐
        │ a        ┆ b   │
        │ ---      ┆ --- │
        │ f64      ┆ f64 │
        ╞══════════╪═════╡
        │ 1.290994 ┆ 0.5 │
        └──────────┴─────┘
        >>> lf.std(ddof=0).collect()
        shape: (1, 2)
        ┌──────────┬──────────┐
        │ a        ┆ b        │
        │ ---      ┆ ---      │
        │ f64      ┆ f64      │
        ╞══════════╪══════════╡
        │ 1.118034 ┆ 0.433013 │
        └──────────┴──────────┘
        """
        ...

    def var(self, ddof: int=1) -> LazyFrame:
        """
        Aggregate the columns in the LazyFrame to their variance value.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.var().collect()
        shape: (1, 2)
        ┌──────────┬──────┐
        │ a        ┆ b    │
        │ ---      ┆ ---  │
        │ f64      ┆ f64  │
        ╞══════════╪══════╡
        │ 1.666667 ┆ 0.25 │
        └──────────┴──────┘
        >>> lf.var(ddof=0).collect()
        shape: (1, 2)
        ┌──────┬────────┐
        │ a    ┆ b      │
        │ ---  ┆ ---    │
        │ f64  ┆ f64    │
        ╞══════╪════════╡
        │ 1.25 ┆ 0.1875 │
        └──────┴────────┘
        """
        ...

    def max(self) -> LazyFrame:
        """
        Aggregate the columns in the LazyFrame to their maximum value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.max().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 4   ┆ 2   │
        └─────┴─────┘
        """
        ...

    def min(self) -> LazyFrame:
        """
        Aggregate the columns in the LazyFrame to their minimum value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.min().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 1   │
        └─────┴─────┘
        """
        ...

    def sum(self) -> LazyFrame:
        """
        Aggregate the columns in the LazyFrame to their sum value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.sum().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 10  ┆ 5   │
        └─────┴─────┘
        """
        ...

    def mean(self) -> LazyFrame:
        """
        Aggregate the columns in the LazyFrame to their mean value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.mean().collect()
        shape: (1, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ f64 ┆ f64  │
        ╞═════╪══════╡
        │ 2.5 ┆ 1.25 │
        └─────┴──────┘
        """
        ...

    def median(self) -> LazyFrame:
        """
        Aggregate the columns in the LazyFrame to their median value.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.median().collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 2.5 ┆ 1.0 │
        └─────┴─────┘
        """
        ...

    def null_count(self) -> LazyFrame:
        """
        Aggregate the columns in the LazyFrame as the sum of their null value count.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, None, 3],
        ...         "bar": [6, 7, None],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> lf.null_count().collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ u32 ┆ u32 ┆ u32 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 0   │
        └─────┴─────┴─────┘
        """
        ...

    def quantile(self, quantile: float | Expr, interpolation: RollingInterpolationMethod='nearest') -> LazyFrame:
        """
        Aggregate the columns in the LazyFrame to their quantile value.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {'nearest', 'higher', 'lower', 'midpoint', 'linear'}
            Interpolation method.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> lf.quantile(0.7).collect()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 3.0 ┆ 1.0 │
        └─────┴─────┘
        """
        ...

    def explode(self, columns: str | Expr | Sequence[str | Expr], *more_columns: str | Expr) -> LazyFrame:
        """
        Explode the DataFrame to long format by exploding the given columns.

        Parameters
        ----------
        columns
            Column names, expressions, or a selector defining them. The underlying
            columns being exploded must be of the `List` or `Array` data type.
        *more_columns
            Additional names of columns to explode, specified as positional arguments.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "letters": ["a", "a", "b", "c"],
        ...         "numbers": [[1], [2, 3], [4, 5], [6, 7, 8]],
        ...     }
        ... )
        >>> lf.explode("numbers").collect()
        shape: (8, 2)
        ┌─────────┬─────────┐
        │ letters ┆ numbers │
        │ ---     ┆ ---     │
        │ str     ┆ i64     │
        ╞═════════╪═════════╡
        │ a       ┆ 1       │
        │ a       ┆ 2       │
        │ a       ┆ 3       │
        │ b       ┆ 4       │
        │ b       ┆ 5       │
        │ c       ┆ 6       │
        │ c       ┆ 7       │
        │ c       ┆ 8       │
        └─────────┴─────────┘
        """
        ...

    def unique(self, subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None=None, *, keep: UniqueKeepStrategy='any', maintain_order: bool=False) -> LazyFrame:
        """
        Drop duplicate rows from this DataFrame.

        Parameters
        ----------
        subset
            Column name(s) or selector(s), to consider when identifying
            duplicate rows. If set to `None` (default), use all columns.
        keep : {'first', 'last', 'any', 'none'}
            Which of the duplicate rows to keep.

            * 'any': Does not give any guarantee of which row is kept.
                     This allows more optimizations.
            * 'none': Don't keep duplicate rows.
            * 'first': Keep first unique row.
            * 'last': Keep last unique row.
        maintain_order
            Keep the same order as the original DataFrame. This is more expensive to
            compute.
            Settings this to `True` blocks the possibility
            to run on the streaming engine.

        Returns
        -------
        LazyFrame
            LazyFrame with unique rows.

        Warnings
        --------
        This method will fail if there is a column of type `List` in the DataFrame or
        subset.

        Notes
        -----
        If you're coming from pandas, this is similar to
        `pandas.DataFrame.drop_duplicates`.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3, 1],
        ...         "bar": ["a", "a", "a", "a"],
        ...         "ham": ["b", "b", "b", "b"],
        ...     }
        ... )
        >>> lf.unique(maintain_order=True).collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ a   ┆ b   │
        │ 2   ┆ a   ┆ b   │
        │ 3   ┆ a   ┆ b   │
        └─────┴─────┴─────┘
        >>> lf.unique(subset=["bar", "ham"], maintain_order=True).collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ a   ┆ b   │
        └─────┴─────┴─────┘
        >>> lf.unique(keep="last", maintain_order=True).collect()
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 2   ┆ a   ┆ b   │
        │ 3   ┆ a   ┆ b   │
        │ 1   ┆ a   ┆ b   │
        └─────┴─────┴─────┘
        """
        ...

    def drop_nulls(self, subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None=None) -> LazyFrame:
        """
        Drop all rows that contain null values.

        The original order of the remaining rows is preserved.

        Parameters
        ----------
        subset
            Column name(s) for which null values are considered.
            If set to `None` (default), use all columns.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, None, 8],
        ...         "ham": ["a", "b", None],
        ...     }
        ... )

        The default behavior of this method is to drop rows where any single
        value of the row is null.

        >>> lf.drop_nulls().collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        This behaviour can be constrained to consider only a subset of columns, as
        defined by name or with a selector. For example, dropping rows if there is
        a null in any of the integer columns:

        >>> import polars.selectors as cs
        >>> lf.drop_nulls(subset=cs.integer()).collect()
        shape: (2, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ i64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 1   ┆ 6   ┆ a    │
        │ 3   ┆ 8   ┆ null │
        └─────┴─────┴──────┘

        This method drops a row if any single value of the row is null.

        Below are some example snippets that show how you could drop null
        values based on other conditions:

        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": [None, None, None, None],
        ...         "b": [1, 2, None, 1],
        ...         "c": [1, None, None, 1],
        ...     }
        ... )
        >>> lf.collect()
        shape: (4, 3)
        ┌──────┬──────┬──────┐
        │ a    ┆ b    ┆ c    │
        │ ---  ┆ ---  ┆ ---  │
        │ null ┆ i64  ┆ i64  │
        ╞══════╪══════╪══════╡
        │ null ┆ 1    ┆ 1    │
        │ null ┆ 2    ┆ null │
        │ null ┆ null ┆ null │
        │ null ┆ 1    ┆ 1    │
        └──────┴──────┴──────┘

        Drop a row only if all values are null:

        >>> lf.filter(~pl.all_horizontal(pl.all().is_null())).collect()
        shape: (3, 3)
        ┌──────┬─────┬──────┐
        │ a    ┆ b   ┆ c    │
        │ ---  ┆ --- ┆ ---  │
        │ null ┆ i64 ┆ i64  │
        ╞══════╪═════╪══════╡
        │ null ┆ 1   ┆ 1    │
        │ null ┆ 2   ┆ null │
        │ null ┆ 1   ┆ 1    │
        └──────┴─────┴──────┘
        """
        ...

    def unpivot(self, on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None=None, *, index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None=None, variable_name: str | None=None, value_name: str | None=None, streamable: bool=True) -> LazyFrame:
        """
        Unpivot a DataFrame from wide to long format.

        Optionally leaves identifiers set.

        This function is useful to massage a DataFrame into a format where one or more
        columns are identifier variables (index) while all other columns, considered
        measured variables (on), are "unpivoted" to the row axis leaving just
        two non-identifier columns, 'variable' and 'value'.

        Parameters
        ----------
        on
            Column(s) or selector(s) to use as values variables; if `on`
            is empty all columns that are not in `index` will be used.
        index
            Column(s) or selector(s) to use as identifier variables.
        variable_name
            Name to give to the `variable` column. Defaults to "variable"
        value_name
            Name to give to the `value` column. Defaults to "value"
        streamable
            deprecated

        Notes
        -----
        If you're coming from pandas, this is similar to `pandas.DataFrame.melt`,
        but with `index` replacing `id_vars` and `on` replacing `value_vars`.
        In other frameworks, you might know this operation as `pivot_longer`.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "a": ["x", "y", "z"],
        ...         "b": [1, 3, 5],
        ...         "c": [2, 4, 6],
        ...     }
        ... )
        >>> import polars.selectors as cs
        >>> lf.unpivot(cs.numeric(), index="a").collect()
        shape: (6, 3)
        ┌─────┬──────────┬───────┐
        │ a   ┆ variable ┆ value │
        │ --- ┆ ---      ┆ ---   │
        │ str ┆ str      ┆ i64   │
        ╞═════╪══════════╪═══════╡
        │ x   ┆ b        ┆ 1     │
        │ y   ┆ b        ┆ 3     │
        │ z   ┆ b        ┆ 5     │
        │ x   ┆ c        ┆ 2     │
        │ y   ┆ c        ┆ 4     │
        │ z   ┆ c        ┆ 6     │
        └─────┴──────────┴───────┘
        """
        ...

    def map_batches(self, function: Callable[[DataFrame], DataFrame], *, predicate_pushdown: bool=True, projection_pushdown: bool=True, slice_pushdown: bool=True, no_optimizations: bool=False, schema: None | SchemaDict=None, validate_output_schema: bool=True, streamable: bool=False) -> LazyFrame:
        """
        Apply a custom function.

        It is important that the function returns a Polars DataFrame.

        Parameters
        ----------
        function
            Lambda/ function to apply.
        predicate_pushdown
            Allow predicate pushdown optimization to pass this node.
        projection_pushdown
            Allow projection pushdown optimization to pass this node.
        slice_pushdown
            Allow slice pushdown optimization to pass this node.
        no_optimizations
            Turn off all optimizations past this point.
        schema
            Output schema of the function, if set to `None` we assume that the schema
            will remain unchanged by the applied function.
        validate_output_schema
            It is paramount that polars' schema is correct. This flag will ensure that
            the output schema of this function will be checked with the expected schema.
            Setting this to `False` will not do this check, but may lead to hard to
            debug bugs.
        streamable
            Whether the function that is given is eligible to be running with the
            streaming engine. That means that the function must produce the same result
            when it is executed in batches or when it is be executed on the full
            dataset.

        Warnings
        --------
        The `schema` of a `LazyFrame` must always be correct. It is up to the caller
        of this function to ensure that this invariant is upheld.

        It is important that the optimization flags are correct. If the custom function
        for instance does an aggregation of a column, `predicate_pushdown` should not
        be allowed, as this prunes rows and will influence your aggregation results.

        Examples
        --------
        >>> lf = (  # doctest: +SKIP
        ...     pl.LazyFrame(
        ...         {
        ...             "a": pl.int_range(-100_000, 0, eager=True),
        ...             "b": pl.int_range(0, 100_000, eager=True),
        ...         }
        ...     )
        ...     .map_batches(lambda x: 2 * x, streamable=True)
        ...     .collect(streaming=True)
        ... )
        shape: (100_000, 2)
        ┌─────────┬────────┐
        │ a       ┆ b      │
        │ ---     ┆ ---    │
        │ i64     ┆ i64    │
        ╞═════════╪════════╡
        │ -200000 ┆ 0      │
        │ -199998 ┆ 2      │
        │ -199996 ┆ 4      │
        │ -199994 ┆ 6      │
        │ …       ┆ …      │
        │ -8      ┆ 199992 │
        │ -6      ┆ 199994 │
        │ -4      ┆ 199996 │
        │ -2      ┆ 199998 │
        └─────────┴────────┘
        """
        ...

    def interpolate(self) -> LazyFrame:
        """
        Interpolate intermediate values. The interpolation method is linear.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "foo": [1, None, 9, 10],
        ...         "bar": [6, 7, 9, None],
        ...         "baz": [1, None, None, 9],
        ...     }
        ... )
        >>> lf.interpolate().collect()
        shape: (4, 3)
        ┌──────┬──────┬──────────┐
        │ foo  ┆ bar  ┆ baz      │
        │ ---  ┆ ---  ┆ ---      │
        │ f64  ┆ f64  ┆ f64      │
        ╞══════╪══════╪══════════╡
        │ 1.0  ┆ 6.0  ┆ 1.0      │
        │ 5.0  ┆ 7.0  ┆ 3.666667 │
        │ 9.0  ┆ 9.0  ┆ 6.333333 │
        │ 10.0 ┆ null ┆ 9.0      │
        └──────┴──────┴──────────┘
        """
        ...

    def unnest(self, columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector], *more_columns: ColumnNameOrSelector) -> LazyFrame:
        """
        Decompose struct columns into separate columns for each of their fields.

        The new columns will be inserted into the DataFrame at the location of the
        struct column.

        Parameters
        ----------
        columns
            Name of the struct column(s) that should be unnested.
        *more_columns
            Additional columns to unnest, specified as positional arguments.

        Examples
        --------
        >>> df = pl.LazyFrame(
        ...     {
        ...         "before": ["foo", "bar"],
        ...         "t_a": [1, 2],
        ...         "t_b": ["a", "b"],
        ...         "t_c": [True, None],
        ...         "t_d": [[1, 2], [3]],
        ...         "after": ["baz", "womp"],
        ...     }
        ... ).select("before", pl.struct(pl.col("^t_.$")).alias("t_struct"), "after")
        >>> df.collect()
        shape: (2, 3)
        ┌────────┬─────────────────────┬───────┐
        │ before ┆ t_struct            ┆ after │
        │ ---    ┆ ---                 ┆ ---   │
        │ str    ┆ struct[4]           ┆ str   │
        ╞════════╪═════════════════════╪═══════╡
        │ foo    ┆ {1,"a",true,[1, 2]} ┆ baz   │
        │ bar    ┆ {2,"b",null,[3]}    ┆ womp  │
        └────────┴─────────────────────┴───────┘
        >>> df.unnest("t_struct").collect()
        shape: (2, 6)
        ┌────────┬─────┬─────┬──────┬───────────┬───────┐
        │ before ┆ t_a ┆ t_b ┆ t_c  ┆ t_d       ┆ after │
        │ ---    ┆ --- ┆ --- ┆ ---  ┆ ---       ┆ ---   │
        │ str    ┆ i64 ┆ str ┆ bool ┆ list[i64] ┆ str   │
        ╞════════╪═════╪═════╪══════╪═══════════╪═══════╡
        │ foo    ┆ 1   ┆ a   ┆ true ┆ [1, 2]    ┆ baz   │
        │ bar    ┆ 2   ┆ b   ┆ null ┆ [3]       ┆ womp  │
        └────────┴─────┴─────┴──────┴───────────┴───────┘
        """
        ...

    def merge_sorted(self, other: LazyFrame, key: str) -> LazyFrame:
        """
        Take two sorted DataFrames and merge them by the sorted key.

        The output of this operation will also be sorted.
        It is the callers responsibility that the frames are sorted
        by that key otherwise the output will not make sense.

        The schemas of both LazyFrames must be equal.

        Parameters
        ----------
        other
            Other DataFrame that must be merged
        key
            Key that is sorted.

        Examples
        --------
        >>> df0 = pl.LazyFrame(
        ...     {"name": ["steve", "elise", "bob"], "age": [42, 44, 18]}
        ... ).sort("age")
        >>> df0.collect()
        shape: (3, 2)
        ┌───────┬─────┐
        │ name  ┆ age │
        │ ---   ┆ --- │
        │ str   ┆ i64 │
        ╞═══════╪═════╡
        │ bob   ┆ 18  │
        │ steve ┆ 42  │
        │ elise ┆ 44  │
        └───────┴─────┘
        >>> df1 = pl.LazyFrame(
        ...     {"name": ["anna", "megan", "steve", "thomas"], "age": [21, 33, 42, 20]}
        ... ).sort("age")
        >>> df1.collect()
        shape: (4, 2)
        ┌────────┬─────┐
        │ name   ┆ age │
        │ ---    ┆ --- │
        │ str    ┆ i64 │
        ╞════════╪═════╡
        │ thomas ┆ 20  │
        │ anna   ┆ 21  │
        │ megan  ┆ 33  │
        │ steve  ┆ 42  │
        └────────┴─────┘
        >>> df0.merge_sorted(df1, key="age").collect()
        shape: (7, 2)
        ┌────────┬─────┐
        │ name   ┆ age │
        │ ---    ┆ --- │
        │ str    ┆ i64 │
        ╞════════╪═════╡
        │ bob    ┆ 18  │
        │ thomas ┆ 20  │
        │ anna   ┆ 21  │
        │ megan  ┆ 33  │
        │ steve  ┆ 42  │
        │ steve  ┆ 42  │
        │ elise  ┆ 44  │
        └────────┴─────┘
        """
        ...

    def set_sorted(self, column: str, *, descending: bool=False) -> LazyFrame:
        """
        Indicate that one or multiple columns are sorted.

        This can speed up future operations.

        Parameters
        ----------
        column
            Columns that are sorted
        descending
            Whether the columns are sorted in descending order.

        Warnings
        --------
        This can lead to incorrect results if the data is NOT sorted!!
        Use with care!

        """
        ...

    @unstable()
    def update(self, other: LazyFrame, on: str | Sequence[str] | None=None, how: Literal['left', 'inner', 'full']='left', *, left_on: str | Sequence[str] | None=None, right_on: str | Sequence[str] | None=None, include_nulls: bool=False) -> LazyFrame:
        """
        Update the values in this `LazyFrame` with the values in `other`.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Parameters
        ----------
        other
            LazyFrame that will be used to update the values
        on
            Column names that will be joined on. If set to `None` (default),
            the implicit row index of each frame is used as a join key.
        how : {'left', 'inner', 'full'}
            * 'left' will keep all rows from the left table; rows may be duplicated
              if multiple rows in the right frame match the left row's key.
            * 'inner' keeps only those rows where the key exists in both frames.
            * 'full' will update existing rows where the key matches while also
              adding any new rows contained in the given frame.
        left_on
           Join column(s) of the left DataFrame.
        right_on
           Join column(s) of the right DataFrame.
        include_nulls
            Overwrite values in the left frame with null values from the right frame.
            If set to `False` (default), null values in the right frame are ignored.

        Notes
        -----
        This is syntactic sugar for a left/inner join, with an optional coalesce when
        `include_nulls = False`.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {
        ...         "A": [1, 2, 3, 4],
        ...         "B": [400, 500, 600, 700],
        ...     }
        ... )
        >>> lf.collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 400 │
        │ 2   ┆ 500 │
        │ 3   ┆ 600 │
        │ 4   ┆ 700 │
        └─────┴─────┘
        >>> new_lf = pl.LazyFrame(
        ...     {
        ...         "B": [-66, None, -99],
        ...         "C": [5, 3, 1],
        ...     }
        ... )

        Update `df` values with the non-null values in `new_df`, by row index:

        >>> lf.update(new_lf).collect()
        shape: (4, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ -66 │
        │ 2   ┆ 500 │
        │ 3   ┆ -99 │
        │ 4   ┆ 700 │
        └─────┴─────┘

        Update `df` values with the non-null values in `new_df`, by row index,
        but only keeping those rows that are common to both frames:

        >>> lf.update(new_lf, how="inner").collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ -66 │
        │ 2   ┆ 500 │
        │ 3   ┆ -99 │
        └─────┴─────┘

        Update `df` values with the non-null values in `new_df`, using a full
        outer join strategy that defines explicit join columns in each frame:

        >>> lf.update(new_lf, left_on=["A"], right_on=["C"], how="full").collect()
        shape: (5, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ -99 │
        │ 2   ┆ 500 │
        │ 3   ┆ 600 │
        │ 4   ┆ 700 │
        │ 5   ┆ -66 │
        └─────┴─────┘

        Update `df` values including null values in `new_df`, using a full
        outer join strategy that defines explicit join columns in each frame:

        >>> lf.update(
        ...     new_lf, left_on="A", right_on="C", how="full", include_nulls=True
        ... ).collect()
        shape: (5, 2)
        ┌─────┬──────┐
        │ A   ┆ B    │
        │ --- ┆ ---  │
        │ i64 ┆ i64  │
        ╞═════╪══════╡
        │ 1   ┆ -99  │
        │ 2   ┆ 500  │
        │ 3   ┆ null │
        │ 4   ┆ 700  │
        │ 5   ┆ -66  │
        └─────┴──────┘
        """
        ...

    def count(self) -> LazyFrame:
        """
        Return the number of non-null elements for each column.

        Examples
        --------
        >>> lf = pl.LazyFrame(
        ...     {"a": [1, 2, 3, 4], "b": [1, 2, 1, None], "c": [None, None, None, None]}
        ... )
        >>> lf.count().collect()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ u32 ┆ u32 ┆ u32 │
        ╞═════╪═════╪═════╡
        │ 4   ┆ 3   ┆ 0   │
        └─────┴─────┴─────┘
        """
        ...

    @deprecate_function('Use `unpivot` instead, with `index` instead of `id_vars` and `on` instead of `value_vars`', version='1.0.0')
    def melt(self, id_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None=None, value_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None=None, variable_name: str | None=None, value_name: str | None=None, *, streamable: bool=True) -> LazyFrame:
        """
        Unpivot a DataFrame from wide to long format.

        Optionally leaves identifiers set.

        This function is useful to massage a DataFrame into a format where one or more
        columns are identifier variables (id_vars) while all other columns, considered
        measured variables (value_vars), are "unpivoted" to the row axis leaving just
        two non-identifier columns, 'variable' and 'value'.

        .. deprecated:: 1.0.0
            Please use :meth:`.unpivot` instead.

        Parameters
        ----------
        id_vars
            Column(s) or selector(s) to use as identifier variables.
        value_vars
            Column(s) or selector(s) to use as values variables; if `value_vars`
            is empty all columns that are not in `id_vars` will be used.
        variable_name
            Name to give to the `variable` column. Defaults to "variable"
        value_name
            Name to give to the `value` column. Defaults to "value"
        streamable
            Allow this node to run in the streaming engine.
            If this runs in streaming, the output of the unpivot operation
            will not have a stable ordering.
        """
        ...

    def _to_metadata(self, columns: None | str | list[str]=None, stats: None | str | list[str]=None) -> DataFrame:
        """
        Get all runtime metadata for each column.

        This is unstable and is meant for debugging purposes.
        """
        ...
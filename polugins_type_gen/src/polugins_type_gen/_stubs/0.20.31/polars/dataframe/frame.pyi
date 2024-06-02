#: version 0.20.31
import P
import deltalake
import deltalake.table
import jax
import np as np
import npt
import pa as pa
import pd as pd
import torch
from _io import BytesIO

from polars.polars import PyDataFrame
from pathlib import Path
from polars._utils.construction.dataframe import (
    arrow_to_pydf as arrow_to_pydf,
    dataframe_to_pydf as dataframe_to_pydf,
    dict_to_pydf as dict_to_pydf,
    iterable_to_pydf as iterable_to_pydf,
    numpy_to_pydf as numpy_to_pydf,
    pandas_to_pydf as pandas_to_pydf,
    sequence_to_pydf as sequence_to_pydf,
    series_to_pydf as series_to_pydf,
)
from polars._utils.convert import parse_as_duration_string as parse_as_duration_string
from polars._utils.deprecation import (
    deprecate_function as deprecate_function,
    deprecate_nonkeyword_arguments as deprecate_nonkeyword_arguments,
    deprecate_parameter_as_positional as deprecate_parameter_as_positional,
    deprecate_renamed_function as deprecate_renamed_function,
    deprecate_renamed_parameter as deprecate_renamed_parameter,
    deprecate_saturating as deprecate_saturating,
    issue_deprecation_warning as issue_deprecation_warning,
)
from polars._utils.getitem import get_df_item_by_key as get_df_item_by_key
from polars._utils.parse_expr_input import parse_as_expression as parse_as_expression
from polars._utils.unstable import (
    issue_unstable_warning as issue_unstable_warning,
    unstable as unstable,
)
from polars._utils.various import (
    is_bool_sequence as is_bool_sequence,
    normalize_filepath as normalize_filepath,
    parse_version as parse_version,
    scale_bytes as scale_bytes,
    warn_null_comparison as warn_null_comparison,
)
from polars._utils.wrap import wrap_expr as wrap_expr, wrap_ldf as wrap_ldf, wrap_s as wrap_s
from polars.dataframe._html import NotebookFormatter as NotebookFormatter
from polars.dataframe.group_by import (
    DynamicGroupBy as DynamicGroupBy,
    GroupBy as GroupBy,
    RollingGroupBy as RollingGroupBy,
)
from polars.datatypes.classes import (
    Boolean as Boolean,
    Float32 as Float32,
    Float64 as Float64,
    Int32 as Int32,
    Int64 as Int64,
    Object as Object,
    String as String,
    Struct as Struct,
    UInt16 as UInt16,
    UInt32 as UInt32,
    UInt64 as UInt64,
)
from polars.dependencies import (
    _check_for_numpy as _check_for_numpy,
    _check_for_pandas as _check_for_pandas,
    _check_for_pyarrow as _check_for_pyarrow,
    hvplot as hvplot,
    import_optional as import_optional,
)
from polars.exceptions import (
    ModuleUpgradeRequired as ModuleUpgradeRequired,
    NoRowsReturnedError as NoRowsReturnedError,
    TooManyRowsReturnedError as TooManyRowsReturnedError,
)
from polars.functions.col import col as col
from polars.functions.lit import lit as lit
from polars.selectors import (
    _expand_selector_dicts as _expand_selector_dicts,
    _expand_selectors as _expand_selectors,
)
from typing import (
    Any,
    Callable,
    ClassVar as _ClassVar,
    Collection,
    IO,
    Iterable,
    Iterator,
    JaxExportType,
    Mapping,
    NoReturn,
    Sequence,
    TorchExportType,
)

TYPE_CHECKING: bool
INTEGER_DTYPES: frozenset
N_INFER_DEFAULT: int
_HVPLOT_AVAILABLE: bool
_PANDAS_AVAILABLE: bool
_PYARROW_AVAILABLE: bool
_dtype_str_repr: builtin_function_or_method
_write_clipboard_string: builtin_function_or_method

class DataFrame:
    _accessors: _ClassVar[set] = ...
    columns: list[str]
    def __init__(
        self, data: FrameInitTypes | None = ..., schema: SchemaDefinition | None = ...
    ) -> None: ...
    @classmethod
    def deserialize(cls, source: str | Path | IOBase) -> Self:
        """
        Read a serialized DataFrame from a file.

        .. versionadded:: 0.20.31

        Parameters
        ----------
        source
            Path to a file or a file-like object (by file-like object, we refer to
            objects that have a `read()` method, such as a file handler (e.g.
            via builtin `open` function) or `BytesIO`).

        See Also
        --------
        DataFrame.serialize

        Examples
        --------
        >>> import io
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        >>> json = df.serialize()
        >>> pl.DataFrame.deserialize(io.StringIO(json))
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 1   ┆ 4.0 │
        │ 2   ┆ 5.0 │
        │ 3   ┆ 6.0 │
        └─────┴─────┘
        """
    @classmethod
    def _from_pydf(cls, py_df: PyDataFrame) -> Self:
        """Construct Polars DataFrame from FFI PyDataFrame object."""
    @classmethod
    def _from_arrow(
        cls, data: pa.Table | pa.RecordBatch, schema: SchemaDefinition | None = ...
    ) -> Self:
        """
        Construct a DataFrame from an Arrow table.

        This operation will be zero copy for the most part. Types that are not
        supported by Polars may be cast to the closest supported type.

        Parameters
        ----------
        data : arrow Table, RecordBatch, or sequence of sequences
            Data representing an Arrow Table or RecordBatch.
        schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
            The DataFrame schema may be declared in several ways:

            * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
            * As a list of column names; in this case types are automatically inferred.
            * As a list of (name,type) pairs; this is equivalent to the dictionary form.

            If you supply a list of column names that does not match the names in the
            underlying data, the names given here will overwrite them. The number
            of names given in the schema should match the underlying data dimensions.
        schema_overrides : dict, default None
            Support type specification or override of one or more columns; note that
            any dtypes inferred from the columns param will be overridden.
        rechunk : bool, default True
            Make sure that all data is in contiguous memory.
        """
    @classmethod
    def _from_pandas(cls, data: pd.DataFrame, schema: SchemaDefinition | None = ...) -> Self:
        """
        Construct a Polars DataFrame from a pandas DataFrame.

        Parameters
        ----------
        data : pandas DataFrame
            Two-dimensional data represented as a pandas DataFrame.
        schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict
            The DataFrame schema may be declared in several ways:

            * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.
            * As a list of column names; in this case types are automatically inferred.
            * As a list of (name,type) pairs; this is equivalent to the dictionary form.

            If you supply a list of column names that does not match the names in the
            underlying data, the names given here will overwrite them. The number
            of names given in the schema should match the underlying data dimensions.
        schema_overrides : dict, default None
            Support type specification or override of one or more columns; note that
            any dtypes inferred from the columns param will be overridden.
        rechunk : bool, default True
            Make sure that all data is in contiguous memory.
        nan_to_null : bool, default True
            If the data contains NaN values they will be converted to null/None.
        include_index : bool, default False
            Load any non-default pandas indexes as columns.
        """
    def _replace(self, column: str, new_column: Series) -> Self:
        """Replace a column by a new Series (in place)."""
    def __array__(
        self, dtype: npt.DTypeLike | None = ..., copy: bool | None = ...
    ) -> np.ndarray[Any, Any]:
        """
        Return a NumPy ndarray with the given data type.

        This method ensures a Polars DataFrame can be treated as a NumPy ndarray.
        It enables `np.asarray` and NumPy universal functions.

        See the NumPy documentation for more information:
        https://numpy.org/doc/stable/user/basics.interoperability.html#the-array-method
        """
    def __dataframe__(self, nan_as_null: bool = ..., allow_copy: bool = ...) -> PolarsDataFrame:
        """
        Convert to a dataframe object implementing the dataframe interchange protocol.

        Parameters
        ----------
        nan_as_null
            Overwrite null values in the data with `NaN`.

            .. warning::
                This functionality has not been implemented and the parameter will be
                removed in a future version.
                Setting this to `True` will raise a `NotImplementedError`.
        allow_copy
            Allow memory to be copied to perform the conversion. If set to `False`,
            causes conversions that are not zero-copy to fail.

        Notes
        -----
        Details on the Python dataframe interchange protocol:
        https://data-apis.org/dataframe-protocol/latest/index.html

        Examples
        --------
        Convert a Polars DataFrame to a generic dataframe object and access some
        properties.

        >>> df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["x", "y"]})
        >>> dfi = df.__dataframe__()
        >>> dfi.num_rows()
        2
        >>> dfi.get_column(1).dtype
        (<DtypeKind.FLOAT: 2>, 64, \'g\', \'=\')
        """
    def _comp(self, other: Any, op: ComparisonOperator) -> DataFrame:
        """Compare a DataFrame with another object."""
    def _compare_to_other_df(self, other: DataFrame, op: ComparisonOperator) -> DataFrame:
        """Compare a DataFrame with another DataFrame."""
    def _compare_to_non_df(self, other: Any, op: ComparisonOperator) -> DataFrame:
        """Compare a DataFrame with a non-DataFrame object."""
    def _div(self, other: Any) -> DataFrame: ...
    def _cast_all_from_to(
        self, df: DataFrame, from_: frozenset[PolarsDataType], to: PolarsDataType
    ) -> DataFrame: ...
    def __floordiv__(self, other: DataFrame | Series | int | float) -> DataFrame: ...
    def __truediv__(self, other: DataFrame | Series | int | float) -> DataFrame: ...
    def __bool__(self) -> NoReturn: ...
    def __eq__(self, other: Any) -> DataFrame: ...
    def __ne__(self, other: Any) -> DataFrame: ...
    def __gt__(self, other: Any) -> DataFrame: ...
    def __lt__(self, other: Any) -> DataFrame: ...
    def __ge__(self, other: Any) -> DataFrame: ...
    def __le__(self, other: Any) -> DataFrame: ...
    def __mul__(self, other: DataFrame | Series | int | float) -> Self: ...
    def __rmul__(self, other: DataFrame | Series | int | float) -> Self: ...
    def __add__(self, other: DataFrame | Series | int | float | bool | str) -> DataFrame: ...
    def __radd__(self, other: DataFrame | Series | int | float | bool | str) -> DataFrame: ...
    def __sub__(self, other: DataFrame | Series | int | float) -> Self: ...
    def __mod__(self, other: DataFrame | Series | int | float) -> Self: ...
    def __contains__(self, key: str) -> bool: ...
    def __iter__(self) -> Iterator[Series]: ...
    def __reversed__(self) -> Iterator[Series]: ...
    def __getitem__(
        self,
        key: SingleIndexSelector
        | SingleColSelector
        | MultiColSelector
        | MultiIndexSelector
        | tuple[SingleIndexSelector, SingleColSelector]
        | tuple[SingleIndexSelector, MultiColSelector]
        | tuple[MultiIndexSelector, SingleColSelector]
        | tuple[MultiIndexSelector, MultiColSelector],
    ) -> DataFrame | Series | Any:
        """Get part of the DataFrame as a new DataFrame, Series, or scalar."""
    def __setitem__(
        self, key: str | Sequence[int] | Sequence[str] | tuple[Any, str | int], value: Any
    ) -> None: ...
    def __len__(self) -> int: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: None = ...) -> Self: ...
    def _ipython_key_completions_(self) -> list[str]: ...
    def _repr_html_(self) -> str:
        """
        Format output data in HTML for display in Jupyter Notebooks.

        Output rows and columns can be modified by setting the following ENVIRONMENT
        variables:

        * POLARS_FMT_MAX_COLS: set the number of columns
        * POLARS_FMT_MAX_ROWS: set the number of rows
        """
    def item(self, row: int | None = ..., column: int | str | None = ...) -> Any:
        """
        Return the DataFrame as a scalar, or return the element at the given row/column.

        Parameters
        ----------
        row
            Optional row index.
        column
            Optional column index or name.

        See Also
        --------
        row: Get the values of a single row, either by index or by predicate.

        Notes
        -----
        If row/col not provided, this is equivalent to `df[0,0]`, with a check that
        the shape is (1,1). With row/col, this is equivalent to `df[row,col]`.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> df.select((pl.col("a") * pl.col("b")).sum()).item()
        32
        >>> df.item(1, 1)
        5
        >>> df.item(2, "b")
        6
        """
    def to_arrow(self) -> pa.Table:
        """
        Collect the underlying arrow arrays in an Arrow Table.

        This operation is mostly zero copy.

        Data types that do copy:
            - CategoricalType

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3, 4, 5, 6], "bar": ["a", "b", "c", "d", "e", "f"]}
        ... )
        >>> df.to_arrow()
        pyarrow.Table
        foo: int64
        bar: large_string
        ----
        foo: [[1,2,3,4,5,6]]
        bar: [["a","b","c","d","e","f"]]
        """
    def to_dict(self) -> dict[str, Series] | dict[str, list[Any]]:
        """
        Convert DataFrame to a dictionary mapping column name to values.

        Parameters
        ----------
        as_series
            True -> Values are Series
            False -> Values are List[Any]

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2, 3, 4, 5],
        ...         "fruits": ["banana", "banana", "apple", "apple", "banana"],
        ...         "B": [5, 4, 3, 2, 1],
        ...         "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        ...         "optional": [28, 300, None, 2, -30],
        ...     }
        ... )
        >>> df
        shape: (5, 5)
        ┌─────┬────────┬─────┬────────┬──────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   ┆ optional │
        │ --- ┆ ---    ┆ --- ┆ ---    ┆ ---      │
        │ i64 ┆ str    ┆ i64 ┆ str    ┆ i64      │
        ╞═════╪════════╪═════╪════════╪══════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle ┆ 28       │
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 300      │
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ null     │
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2        │
        │ 5   ┆ banana ┆ 1   ┆ beetle ┆ -30      │
        └─────┴────────┴─────┴────────┴──────────┘
        >>> df.to_dict(as_series=False)
        {\'A\': [1, 2, 3, 4, 5],
        \'fruits\': [\'banana\', \'banana\', \'apple\', \'apple\', \'banana\'],
        \'B\': [5, 4, 3, 2, 1],
        \'cars\': [\'beetle\', \'audi\', \'beetle\', \'beetle\', \'beetle\'],
        \'optional\': [28, 300, None, 2, -30]}
        >>> df.to_dict(as_series=True)
        {\'A\': shape: (5,)
        Series: \'A\' [i64]
        [
            1
            2
            3
            4
            5
        ], \'fruits\': shape: (5,)
        Series: \'fruits\' [str]
        [
            "banana"
            "banana"
            "apple"
            "apple"
            "banana"
        ], \'B\': shape: (5,)
        Series: \'B\' [i64]
        [
            5
            4
            3
            2
            1
        ], \'cars\': shape: (5,)
        Series: \'cars\' [str]
        [
            "beetle"
            "audi"
            "beetle"
            "beetle"
            "beetle"
        ], \'optional\': shape: (5,)
        Series: \'optional\' [i64]
        [
            28
            300
            null
            2
            -30
        ]}
        """
    def to_dicts(self) -> list[dict[str, Any]]:
        """
        Convert every row to a dictionary of Python-native values.

        Notes
        -----
        If you have `ns`-precision temporal values you should be aware that Python
        natively only supports up to `μs`-precision; `ns`-precision values will be
        truncated to microseconds on conversion to Python. If this matters to your
        use-case you should export to a different format (such as Arrow or NumPy).

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df.to_dicts()
        [{\'foo\': 1, \'bar\': 4}, {\'foo\': 2, \'bar\': 5}, {\'foo\': 3, \'bar\': 6}]
        """
    def to_numpy(self) -> np.ndarray[Any, Any]:
        """
        Convert this DataFrame to a NumPy ndarray.

        This operation copies data only when necessary. The conversion is zero copy when
        all of the following hold:

        - The DataFrame is fully contiguous in memory, with all Series back-to-back and
          all Series consisting of a single chunk.
        - The data type is an integer or float.
        - The DataFrame contains no null values.
        - The `order` parameter is set to `fortran` (default).
        - The `writable` parameter is set to `False` (default).

        Parameters
        ----------
        structured
            Return a `structured array`_ with a data type that corresponds to the
            DataFrame schema. If set to `False` (default), a 2D ndarray is
            returned instead.

            .. _structured array: https://numpy.org/doc/stable/user/basics.rec.html
        order
            The index order of the returned NumPy array, either C-like or
            Fortran-like. In general, using the Fortran-like index order is faster.
            However, the C-like order might be more appropriate to use for downstream
            applications to prevent cloning data, e.g. when reshaping into a
            one-dimensional array.
        writable
            Ensure the resulting array is writable. This will force a copy of the data
            if the array was created without copy, as the underlying Arrow data is
            immutable.
        allow_copy
            Allow memory to be copied to perform the conversion. If set to `False`,
            causes conversions that are not zero-copy to fail.

        use_pyarrow
            Use `pyarrow.Array.to_numpy
            <https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.to_numpy>`_

            function for the conversion to NumPy if necessary.

            .. deprecated:: 0.20.28
                Polars now uses its native engine by default for conversion to NumPy.

        Examples
        --------
        Numeric data without nulls can be converted without copying data in some cases.
        The resulting array will not be writable.

        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> arr = df.to_numpy()
        >>> arr
        array([[1],
               [2],
               [3]])
        >>> arr.flags.writeable
        False

        Set `writable=True` to force data copy to make the array writable.

        >>> df.to_numpy(writable=True).flags.writeable
        True

        If the DataFrame contains different numeric data types, the resulting data type
        will be the supertype. This requires data to be copied. Integer types with
        nulls are cast to a float type with `nan` representing a null value.

        >>> df = pl.DataFrame({"a": [1, 2, None], "b": [4.0, 5.0, 6.0]})
        >>> df.to_numpy()
        array([[ 1.,  4.],
               [ 2.,  5.],
               [nan,  6.]])

        Set `allow_copy=False` to raise an error if data would be copied.

        >>> s.to_numpy(allow_copy=False)  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        RuntimeError: copy not allowed: cannot convert to a NumPy array without copying data

        Polars defaults to F-contiguous order. Use `order="c"` to force the resulting
        array to be C-contiguous.

        >>> df.to_numpy(order="c").flags.c_contiguous
        True

        DataFrames with mixed types will result in an array with an object dtype.

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.5, 7.0, 8.5],
        ...         "ham": ["a", "b", "c"],
        ...     },
        ...     schema_overrides={"foo": pl.UInt8, "bar": pl.Float32},
        ... )
        >>> df.to_numpy()
        array([[1, 6.5, \'a\'],
               [2, 7.0, \'b\'],
               [3, 8.5, \'c\']], dtype=object)

        Set `structured=True` to convert to a structured array, which can better
        preserve individual column data such as name and data type.

        >>> df.to_numpy(structured=True)
        array([(1, 6.5, \'a\'), (2, 7. , \'b\'), (3, 8.5, \'c\')],
              dtype=[(\'foo\', \'u1\'), (\'bar\', \'<f4\'), (\'ham\', \'<U1\')])
        """
    def to_jax(self, return_type: JaxExportType = ...) -> jax.Array | dict[str, jax.Array]:
        """
        Convert DataFrame to a Jax Array, or dict of Jax Arrays.

        .. versionadded:: 0.20.27

        .. warning::
            This functionality is currently considered **unstable**. It may be
            changed at any point without it being considered a breaking change.

        Parameters
        ----------
        return_type : {"array", "dict"}
            Set return type; a Jax Array, or dict of Jax Arrays.
        device
            Specify the jax `Device` on which the array will be created; can provide
            a string (such as "cpu", "gpu", or "tpu") in which case the device is
            retrieved as `jax.devices(string)[0]`. For more specific control you
            can supply the instantiated `Device` directly. If None, arrays are
            created on the default device.
        label
            One or more column names, expressions, or selectors that label the feature
            data; results in a `{"label": ..., "features": ...}` dict being returned
            when `return_type` is "dict" instead of a `{"col": array, }` dict.
        features
            One or more column names, expressions, or selectors that contain the feature
            data; if omitted, all columns that are not designated as part of the label
            are used. Only applies when `return_type` is "dict".
        dtype
            Unify the dtype of all returned arrays; this casts any column that is
            not already of the required dtype before converting to Array. Note that
            export will be single-precision (32bit) unless the Jax config/environment
            directs otherwise (eg: "jax_enable_x64" was set True in the config object
            at startup, or "JAX_ENABLE_X64" is set to "1" in the environment).
        order : {"c", "fortran"}
            The index order of the returned Jax array, either C-like (row-major) or
            Fortran-like (column-major).

        See Also
        --------
        to_dummies
        to_numpy
        to_torch

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "lbl": [0, 1, 2, 3],
        ...         "feat1": [1, 0, 0, 1],
        ...         "feat2": [1.5, -0.5, 0.0, -2.25],
        ...     }
        ... )

        Standard return type (2D Array), on the standard device:

        >>> df.to_jax()
        Array([[ 0.  ,  1.  ,  1.5 ],
               [ 1.  ,  0.  , -0.5 ],
               [ 2.  ,  0.  ,  0.  ],
               [ 3.  ,  1.  , -2.25]], dtype=float32)

        Create the Array on the default GPU device:

        >>> a = df.to_jax(device="gpu")  # doctest: +SKIP
        >>> a.device()  # doctest: +SKIP
        GpuDevice(id=0, process_index=0)

        Create the Array on a specific GPU device:

        >>> gpu_device = jax.devices("gpu")[1])  # doctest: +SKIP
        >>> a = df.to_jax(device=gpu_device)  # doctest: +SKIP
        >>> a.device()  # doctest: +SKIP
        GpuDevice(id=1, process_index=0)

        As a dictionary of individual Arrays:

        >>> df.to_jax("dict")
        {\'lbl\': Array([0, 1, 2, 3], dtype=int32),
         \'feat1\': Array([1, 0, 0, 1], dtype=int32),
         \'feat2\': Array([ 1.5 , -0.5 ,  0.  , -2.25], dtype=float32)}

        As a "label" and "features" dictionary; note that as "features" is not
        declared, it defaults to all the columns that are not in "label":

        >>> df.to_jax("dict", label="lbl")
        {\'label\': Array([[0],
                [1],
                [2],
                [3]], dtype=int32),
         \'features\': Array([[ 1.  ,  1.5 ],
                [ 0.  , -0.5 ],
                [ 0.  ,  0.  ],
                [ 1.  , -2.25]], dtype=float32)}

        As a "label" and "features" dictionary where each is designated using
        a col or selector expression (which can also be used to cast the data
        if the label and features are better-represented with different dtypes):

        >>> import polars.selectors as cs
        >>> df.to_jax(
        ...     return_type="dict",
        ...     features=cs.float(),
        ...     label=pl.col("lbl").cast(pl.UInt8),
        ... )
        {\'label\': Array([[0],
                [1],
                [2],
                [3]], dtype=uint8),
         \'features\': Array([[ 1.5 ],
                [-0.5 ],
                [ 0.  ],
                [-2.25]], dtype=float32)}
        """
    def to_torch(
        self, return_type: TorchExportType = ...
    ) -> torch.Tensor | dict[str, torch.Tensor] | PolarsDataset:
        """
        Convert DataFrame to a PyTorch Tensor, Dataset, or dict of Tensors.

        .. versionadded:: 0.20.23

        .. warning::
            This functionality is currently considered **unstable**. It may be
            changed at any point without it being considered a breaking change.

        Parameters
        ----------
        return_type : {"tensor", "dataset", "dict"}
            Set return type; a PyTorch Tensor, PolarsDataset (a frame-specialized
            TensorDataset), or dict of Tensors.
        label
            One or more column names, expressions, or selectors that label the feature
            data; when `return_type` is "dataset", the PolarsDataset will return
            `(features, label)` tensor tuples for each row. Otherwise, it returns
            `(features,)` tensor tuples where the feature contains all the row data.
        features
            One or more column names, expressions, or selectors that contain the feature
            data; if omitted, all columns that are not designated as part of the label
            are used.
        dtype
            Unify the dtype of all returned tensors; this casts any column that is
            not of the required dtype before converting to Tensor. This includes
            the label column *unless* the label is an expression (such as
            `pl.col("label_column").cast(pl.Int16)`).

        See Also
        --------
        to_dummies
        to_jax
        to_numpy

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "lbl": [0, 1, 2, 3],
        ...         "feat1": [1, 0, 0, 1],
        ...         "feat2": [1.5, -0.5, 0.0, -2.25],
        ...     }
        ... )

        Standard return type (Tensor), with f32 supertype:

        >>> df.to_torch(dtype=pl.Float32)
        tensor([[ 0.0000,  1.0000,  1.5000],
                [ 1.0000,  0.0000, -0.5000],
                [ 2.0000,  0.0000,  0.0000],
                [ 3.0000,  1.0000, -2.2500]])

        As a dictionary of individual Tensors:

        >>> df.to_torch("dict")
        {\'lbl\': tensor([0, 1, 2, 3]),
         \'feat1\': tensor([1, 0, 0, 1]),
         \'feat2\': tensor([ 1.5000, -0.5000,  0.0000, -2.2500], dtype=torch.float64)}

        As a "label" and "features" dictionary; note that as "features" is not
        declared, it defaults to all the columns that are not in "label":

        >>> df.to_torch("dict", label="lbl", dtype=pl.Float32)
        {\'label\': tensor([[0.],
                 [1.],
                 [2.],
                 [3.]]),
         \'features\': tensor([[ 1.0000,  1.5000],
                 [ 0.0000, -0.5000],
                 [ 0.0000,  0.0000],
                 [ 1.0000, -2.2500]])}

        As a PolarsDataset, with f64 supertype:

        >>> ds = df.to_torch("dataset", dtype=pl.Float64)
        >>> ds[3]
        (tensor([ 3.0000,  1.0000, -2.2500], dtype=torch.float64),)
        >>> ds[:2]
        (tensor([[ 0.0000,  1.0000,  1.5000],
                 [ 1.0000,  0.0000, -0.5000]], dtype=torch.float64),)
        >>> ds[[0, 3]]
        (tensor([[ 0.0000,  1.0000,  1.5000],
                 [ 3.0000,  1.0000, -2.2500]], dtype=torch.float64),)

        As a convenience the PolarsDataset can opt in to half-precision data
        for experimentation (usually this would be set on the model/pipeline):

        >>> list(ds.half())
        [(tensor([0.0000, 1.0000, 1.5000], dtype=torch.float16),),
         (tensor([ 1.0000,  0.0000, -0.5000], dtype=torch.float16),),
         (tensor([2., 0., 0.], dtype=torch.float16),),
         (tensor([ 3.0000,  1.0000, -2.2500], dtype=torch.float16),)]

        Pass PolarsDataset to a DataLoader, designating the label:

        >>> from torch.utils.data import DataLoader
        >>> ds = df.to_torch("dataset", label="lbl")
        >>> dl = DataLoader(ds, batch_size=2)
        >>> batches = list(dl)
        >>> batches[0]
        [tensor([[ 1.0000,  1.5000],
                 [ 0.0000, -0.5000]], dtype=torch.float64), tensor([0, 1])]

        Note that labels can be given as expressions, allowing them to have
        a dtype independent of the feature columns (multi-column labels are
        supported).

        >>> ds = df.to_torch(
        ...     return_type="dataset",
        ...     dtype=pl.Float32,
        ...     label=pl.col("lbl").cast(pl.Int16),
        ... )
        >>> ds[:2]
        (tensor([[ 1.0000,  1.5000],
                 [ 0.0000, -0.5000]]), tensor([0, 1], dtype=torch.int16))

        Easily integrate with (for example) scikit-learn and other datasets:

        >>> from sklearn.datasets import fetch_california_housing  # doctest: +SKIP
        >>> housing = fetch_california_housing()  # doctest: +SKIP
        >>> df = pl.DataFrame(
        ...     data=housing.data,
        ...     schema=housing.feature_names,
        ... ).with_columns(
        ...     Target=housing.target,
        ... )  # doctest: +SKIP
        >>> train = df.to_torch("dataset", label="Target")  # doctest: +SKIP
        >>> loader = DataLoader(
        ...     train,
        ...     shuffle=True,
        ...     batch_size=64,
        ... )  # doctest: +SKIP
        """
    def to_pandas(self, **kwargs: Any) -> pd.DataFrame:
        """
        Convert this DataFrame to a pandas DataFrame.

        This operation copies data if `use_pyarrow_extension_array` is not enabled.

        Parameters
        ----------
        use_pyarrow_extension_array
            Use PyArrow-backed extension arrays instead of NumPy arrays for the columns
            of the pandas DataFrame. This allows zero copy operations and preservation
            of null values. Subsequent operations on the resulting pandas DataFrame may
            trigger conversion to NumPy if those operations are not supported by PyArrow
            compute functions.
        **kwargs
            Additional keyword arguments to be passed to
            :meth:`pyarrow.Table.to_pandas`.

        Returns
        -------
        :class:`pandas.DataFrame`

        Notes
        -----
        This operation requires that both :mod:`pandas` and :mod:`pyarrow` are
        installed.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.to_pandas()
           foo  bar ham
        0    1  6.0   a
        1    2  7.0   b
        2    3  8.0   c

        Null values in numeric columns are converted to `NaN`.

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, None],
        ...         "bar": [6.0, None, 8.0],
        ...         "ham": [None, "b", "c"],
        ...     }
        ... )
        >>> df.to_pandas()
           foo  bar   ham
        0  1.0  6.0  None
        1  2.0  NaN     b
        2  NaN  8.0     c

        Pass `use_pyarrow_extension_array=True` to get a pandas DataFrame with columns
        backed by PyArrow extension arrays. This will preserve null values.

        >>> df.to_pandas(use_pyarrow_extension_array=True)
            foo   bar   ham
        0     1   6.0  <NA>
        1     2  <NA>     b
        2  <NA>   8.0     c
        >>> _.dtypes
        foo           int64[pyarrow]
        bar          double[pyarrow]
        ham    large_string[pyarrow]
        dtype: object
        """
    def _to_pandas_with_object_columns(self, **kwargs: Any) -> pd.DataFrame: ...
    def _to_pandas_without_object_columns(self, df: DataFrame, **kwargs: Any) -> pd.DataFrame: ...
    def to_series(self, index: int = ...) -> Series:
        """
        Select column as Series at index location.

        Parameters
        ----------
        index
            Location of selection.

        See Also
        --------
        get_column

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.to_series(1)
        shape: (3,)
        Series: \'bar\' [i64]
        [
                6
                7
                8
        ]
        """
    def to_init_repr(self, n: int = ...) -> str:
        """
        Convert DataFrame to instantiatable string representation.

        Parameters
        ----------
        n
            Only use first n rows.

        See Also
        --------
        polars.Series.to_init_repr
        polars.from_repr

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     [
        ...         pl.Series("foo", [1, 2, 3], dtype=pl.UInt8),
        ...         pl.Series("bar", [6.0, 7.0, 8.0], dtype=pl.Float32),
        ...         pl.Series("ham", ["a", "b", "c"], dtype=pl.String),
        ...     ]
        ... )
        >>> print(df.to_init_repr())
        pl.DataFrame(
            [
                pl.Series("foo", [1, 2, 3], dtype=pl.UInt8),
                pl.Series("bar", [6.0, 7.0, 8.0], dtype=pl.Float32),
                pl.Series("ham", [\'a\', \'b\', \'c\'], dtype=pl.String),
            ]
        )

        >>> df_from_str_repr = eval(df.to_init_repr())
        >>> df_from_str_repr
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ u8  ┆ f32 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6.0 ┆ a   │
        │ 2   ┆ 7.0 ┆ b   │
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘
        """
    def serialize(self, file: IOBase | str | Path | None = ...) -> str | None:
        """
        Serialize this DataFrame to a file or string in JSON format.

        .. versionadded:: 0.20.31

        Parameters
        ----------
        file
            File path or writable file-like object to which the result will be written.
            If set to `None` (default), the output is returned as a string instead.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...     }
        ... )
        >>> df.serialize()
        \'{"columns":[{"name":"foo","datatype":"Int64","bit_settings":"","values":[1,2,3]},{"name":"bar","datatype":"Int64","bit_settings":"","values":[6,7,8]}]}\'
        """
    def write_json(self, file: IOBase | str | Path | None = ...) -> str | None:
        """
        Serialize to JSON representation.

        Parameters
        ----------
        file
            File path or writable file-like object to which the result will be written.
            If set to `None` (default), the output is returned as a string instead.
        row_oriented
            Write to row oriented json. This is slower, but more common.

        pretty
            Pretty serialize json.

            .. deprecated:: 0.20.31
                The `pretty` functionality for `write_json` will be removed in the next
                breaking release. Use :meth:`serialize` to serialize the DataFrame in
                the regular JSON format.


        See Also
        --------
        DataFrame.write_ndjson

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...     }
        ... )
        >>> df.write_json(row_oriented=True)
        \'[{"foo":1,"bar":6},{"foo":2,"bar":7},{"foo":3,"bar":8}]\'
        """
    def write_ndjson(self, file: IOBase | str | Path | None = ...) -> str | None:
        """
        Serialize to newline delimited JSON representation.

        Parameters
        ----------
        file
            File path or writable file-like object to which the result will be written.
            If set to `None` (default), the output is returned as a string instead.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...     }
        ... )
        >>> df.write_ndjson()
        \'{"foo":1,"bar":6}\\n{"foo":2,"bar":7}\\n{"foo":3,"bar":8}\\n\'
        """
    def write_csv(self, file: str | Path | IO[str] | IO[bytes] | None = ...) -> str | None:
        """
        Write to comma-separated values (CSV) file.

        Parameters
        ----------
        file
            File path or writable file-like object to which the result will be written.
            If set to `None` (default), the output is returned as a string instead.
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
            precision is inferred from the maximum timeunit found in the frame\'s
            Datetime cols (if any).
        date_format
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate.
        time_format
            A format string, with the specifiers defined by the
            `chrono <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            Rust crate.
        float_precision
            Number of decimal places to write, applied to both `Float32` and
            `Float64` datatypes.
        null_value
            A string representing null values (defaulting to the empty string).
        quote_style : {\'necessary\', \'always\', \'non_numeric\', \'never\'}
            Determines the quoting strategy used.

            - necessary (default): This puts quotes around fields only when necessary.
              They are necessary when fields contain a quote,
              separator or record terminator.
              Quotes are also necessary when writing an empty record
              (which is indistinguishable from a record with one empty field).
              This is the default.
            - always: This puts quotes around every field. Always.
            - never: This never puts quotes around fields, even if that results in
              invalid CSV data (e.g.: by not quoting strings containing the separator).
            - non_numeric: This puts quotes around all fields that are non-numeric.
              Namely, when writing a field that does not parse as a valid float
              or integer, then quotes will be used even if they aren`t strictly
              necessary.

        Examples
        --------
        >>> import pathlib
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> path: pathlib.Path = dirpath / "new_file.csv"
        >>> df.write_csv(path, separator=",")
        """
    def write_clipboard(self, **kwargs: Any) -> None:
        """
        Copy `DataFrame` in csv format to the system clipboard with `write_csv`.

        Useful for pasting into Excel or other similar spreadsheet software.

        Parameters
        ----------
        separator
            Separate CSV fields with this symbol.
        kwargs
            Additional arguments to pass to `write_csv`.

        See Also
        --------
        polars.read_clipboard: Read a DataFrame from the clipboard.
        write_csv: Write to comma-separated values (CSV) file.
        """
    def write_avro(
        self, file: str | Path | IO[bytes], compression: AvroCompression = ..., name: str = ...
    ) -> None:
        """
        Write to Apache Avro file.

        Parameters
        ----------
        file
            File path or writable file-like object to which the data will be written.
        compression : {\'uncompressed\', \'snappy\', \'deflate\'}
            Compression method. Defaults to "uncompressed".
        name
            Schema name. Defaults to empty string.

        Examples
        --------
        >>> import pathlib
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> path: pathlib.Path = dirpath / "new_file.avro"
        >>> df.write_avro(path)
        """
    def write_excel(
        self, workbook: Workbook | IO[bytes] | Path | str | None = ..., worksheet: str | None = ...
    ) -> Workbook:
        """
        Write frame data to a table in an Excel workbook/worksheet.

        Parameters
        ----------
        workbook : Workbook
            String name or path of the workbook to create, BytesIO object to write
            into, or an open `xlsxwriter.Workbook` object that has not been closed.
            If None, writes to a `dataframe.xlsx` workbook in the working directory.
        worksheet : str
            Name of target worksheet; if None, writes to "Sheet1" when creating a new
            workbook (note that writing to an existing workbook requires a valid
            existing -or new- worksheet name).
        position : {str, tuple}
            Table position in Excel notation (eg: "A1"), or a (row,col) integer tuple.
        table_style : {str, dict}
            A named Excel table style, such as "Table Style Medium 4", or a dictionary
            of `{"key":value,}` options containing one or more of the following keys:
            "style", "first_column", "last_column", "banded_columns, "banded_rows".
        table_name : str
            Name of the output table object in the worksheet; can then be referred to
            in the sheet by formulae/charts, or by subsequent `xlsxwriter` operations.
        column_formats : dict
            A `{colname(s):str,}` or `{selector:str,}` dictionary for applying an
            Excel format string to the given columns. Formats defined here (such as
            "dd/mm/yyyy", "0.00%", etc) will override any defined in `dtype_formats`.
        dtype_formats : dict
            A `{dtype:str,}` dictionary that sets the default Excel format for the
            given dtype. (This can be overridden on a per-column basis by the
            `column_formats` param). It is also valid to use dtype groups such as
            `pl.FLOAT_DTYPES` as the dtype/format key, to simplify setting uniform
            integer and float formats.
        conditional_formats : dict
            A dictionary of colname (or selector) keys to a format str, dict, or list
            that defines conditional formatting options for the specified columns.

            * If supplying a string typename, should be one of the valid `xlsxwriter`
              types such as "3_color_scale", "data_bar", etc.
            * If supplying a dictionary you can make use of any/all `xlsxwriter`
              supported options, including icon sets, formulae, etc.
            * Supplying multiple columns as a tuple/key will apply a single format
              across all columns - this is effective in creating a heatmap, as the
              min/max values will be determined across the entire range, not per-column.
            * Finally, you can also supply a list made up from the above options
              in order to apply *more* than one conditional format to the same range.
        header_format : dict
            A `{key:value,}` dictionary of `xlsxwriter` format options to apply
            to the table header row, such as `{"bold":True, "font_color":"#702963"}`.
        column_totals : {bool, list, dict}
            Add a column-total row to the exported table.

            * If True, all numeric columns will have an associated total using "sum".
            * If passing a string, it must be one of the valid total function names
              and all numeric columns will have an associated total using that function.
            * If passing a list of colnames, only those given will have a total.
            * For more control, pass a `{colname:funcname,}` dict.

            Valid total function names are "average", "count_nums", "count", "max",
            "min", "std_dev", "sum", and "var".
        column_widths : {dict, int}
            A `{colname:int,}` or `{selector:int,}` dict or a single integer that
            sets (or overrides if autofitting) table column widths, in integer pixel
            units. If given as an integer the same value is used for all table columns.
        row_totals : {dict, bool}
            Add a row-total column to the right-hand side of the exported table.

            * If True, a column called "total" will be added at the end of the table
              that applies a "sum" function row-wise across all numeric columns.
            * If passing a list/sequence of column names, only the matching columns
              will participate in the sum.
            * Can also pass a `{colname:columns,}` dictionary to create one or
              more total columns with distinct names, referencing different columns.
        row_heights : {dict, int}
            An int or `{row_index:int,}` dictionary that sets the height of the given
            rows (if providing a dictionary) or all rows (if providing an integer) that
            intersect with the table body (including any header and total row) in
            integer pixel units. Note that `row_index` starts at zero and will be
            the header row (unless `include_header` is False).
        sparklines : dict
            A `{colname:list,}` or `{colname:dict,}` dictionary defining one or more
            sparklines to be written into a new column in the table.

            * If passing a list of colnames (used as the source of the sparkline data)
              the default sparkline settings are used (eg: line chart with no markers).
            * For more control an `xlsxwriter`-compliant options dict can be supplied,
              in which case three additional polars-specific keys are available:
              "columns", "insert_before", and "insert_after". These allow you to define
              the source columns and position the sparkline(s) with respect to other
              table columns. If no position directive is given, sparklines are added to
              the end of the table (eg: to the far right) in the order they are given.
        formulas : dict
            A `{colname:formula,}` or `{colname:dict,}` dictionary defining one or
            more formulas to be written into a new column in the table. Note that you
            are strongly advised to use structured references in your formulae wherever
            possible to make it simple to reference columns by name.

            * If providing a string formula (such as "=[@colx]*[@coly]") the column will
              be added to the end of the table (eg: to the far right), after any default
              sparklines and before any row_totals.
            * For the most control supply an options dictionary with the following keys:
              "formula" (mandatory), one of "insert_before" or "insert_after", and
              optionally "return_dtype". The latter is used to appropriately format the
              output of the formula and allow it to participate in row/column totals.
        float_precision : int
            Default number of decimals displayed for floating point columns (note that
            this is purely a formatting directive; the actual values are not rounded).
        include_header : bool
            Indicate if the table should be created with a header row.
        autofilter : bool
            If the table has headers, provide autofilter capability.
        autofit : bool
            Calculate individual column widths from the data.
        hidden_columns : list
             A list or selector representing table columns to hide in the worksheet.
        hide_gridlines : bool
            Do not display any gridlines on the output worksheet.
        sheet_zoom : int
            Set the default zoom level of the output worksheet.
        freeze_panes : str | (str, int, int) | (int, int) | (int, int, int, int)
            Freeze workbook panes.

            * If (row, col) is supplied, panes are split at the top-left corner of the
              specified cell, which are 0-indexed. Thus, to freeze only the top row,
              supply (1, 0).
            * Alternatively, cell notation can be used to supply the cell. For example,
              "A2" indicates the split occurs at the top-left of cell A2, which is the
              equivalent of (1, 0).
            * If (row, col, top_row, top_col) are supplied, the panes are split based on
              the `row` and `col`, and the scrolling region is initialized to begin at
              the `top_row` and `top_col`. Thus, to freeze only the top row and have the
              scrolling region begin at row 10, column D (5th col), supply (1, 0, 9, 4).
              Using cell notation for (row, col), supplying ("A2", 9, 4) is equivalent.

        Notes
        -----
        * A list of compatible `xlsxwriter` format property names can be found here:
          https://xlsxwriter.readthedocs.io/format.html#format-methods-and-format-properties

        * Conditional formatting dictionaries should provide xlsxwriter-compatible
          definitions; polars will take care of how they are applied on the worksheet
          with respect to the relative sheet/column position. For supported options,
          see: https://xlsxwriter.readthedocs.io/working_with_conditional_formats.html

        * Similarly, sparkline option dictionaries should contain xlsxwriter-compatible
          key/values, as well as a mandatory polars "columns" key that defines the
          sparkline source data; these source columns should all be adjacent. Two other
          polars-specific keys are available to help define where the sparkline appears
          in the table: "insert_after", and "insert_before". The value associated with
          these keys should be the name of a column in the exported table.
          https://xlsxwriter.readthedocs.io/working_with_sparklines.html

        * Formula dictionaries *must* contain a key called "formula", and then optional
          "insert_after", "insert_before", and/or "return_dtype" keys. These additional
          keys allow the column to be injected into the table at a specific location,
          and/or to define the return type of the formula (eg: "Int64", "Float64", etc).
          Formulas that refer to table columns should use Excel\'s structured references
          syntax to ensure the formula is applied correctly and is table-relative.
          https://support.microsoft.com/en-us/office/using-structured-references-with-excel-tables-f5ed2452-2337-4f71-bed3-c8ae6d2b276e

        Examples
        --------
        Instantiate a basic DataFrame:

        >>> from random import uniform
        >>> from datetime import date
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "dtm": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        ...         "num": [uniform(-500, 500), uniform(-500, 500), uniform(-500, 500)],
        ...         "val": [10_000, 20_000, 30_000],
        ...     }
        ... )

        Export to "dataframe.xlsx" (the default workbook name, if not specified) in the
        working directory, add column totals ("sum" by default) on all numeric columns,
        then autofit:

        >>> df.write_excel(column_totals=True, autofit=True)  # doctest: +SKIP

        Write frame to a specific location on the sheet, set a named table style,
        apply US-style date formatting, increase default float precision, apply a
        non-default total function to a single column, autofit:

        >>> df.write_excel(  # doctest: +SKIP
        ...     position="B4",
        ...     table_style="Table Style Light 16",
        ...     dtype_formats={pl.Date: "mm/dd/yyyy"},
        ...     column_totals={"num": "average"},
        ...     float_precision=6,
        ...     autofit=True,
        ... )

        Write the same frame to a named worksheet twice, applying different styles
        and conditional formatting to each table, adding table titles using explicit
        xlsxwriter integration:

        >>> from xlsxwriter import Workbook
        >>> with Workbook("multi_frame.xlsx") as wb:  # doctest: +SKIP
        ...     # basic/default conditional formatting
        ...     df.write_excel(
        ...         workbook=wb,
        ...         worksheet="data",
        ...         position=(3, 1),  # specify position as (row,col) coordinates
        ...         conditional_formats={"num": "3_color_scale", "val": "data_bar"},
        ...         table_style="Table Style Medium 4",
        ...     )
        ...
        ...     # advanced conditional formatting, custom styles
        ...     df.write_excel(
        ...         workbook=wb,
        ...         worksheet="data",
        ...         position=(len(df) + 7, 1),
        ...         table_style={
        ...             "style": "Table Style Light 4",
        ...             "first_column": True,
        ...         },
        ...         conditional_formats={
        ...             "num": {
        ...                 "type": "3_color_scale",
        ...                 "min_color": "#76933c",
        ...                 "mid_color": "#c4d79b",
        ...                 "max_color": "#ebf1de",
        ...             },
        ...             "val": {
        ...                 "type": "data_bar",
        ...                 "data_bar_2010": True,
        ...                 "bar_color": "#9bbb59",
        ...                 "bar_negative_color_same": True,
        ...                 "bar_negative_border_color_same": True,
        ...             },
        ...         },
        ...         column_formats={"num": "#,##0.000;[White]-#,##0.000"},
        ...         column_widths={"val": 125},
        ...         autofit=True,
        ...     )
        ...
        ...     # add some table titles (with a custom format)
        ...     ws = wb.get_worksheet_by_name("data")
        ...     fmt_title = wb.add_format(
        ...         {
        ...             "font_color": "#4f6228",
        ...             "font_size": 12,
        ...             "italic": True,
        ...             "bold": True,
        ...         }
        ...     )
        ...     ws.write(2, 1, "Basic/default conditional formatting", fmt_title)
        ...     ws.write(len(df) + 6, 1, "Customised conditional formatting", fmt_title)

        Export a table containing two different types of sparklines. Use default
        options for the "trend" sparkline and customized options (and positioning)
        for the "+/-" win_loss sparkline, with non-default integer dtype formatting,
        column totals, a subtle two-tone heatmap and hidden worksheet gridlines:

        >>> df = pl.DataFrame(
        ...     {
        ...         "id": ["aaa", "bbb", "ccc", "ddd", "eee"],
        ...         "q1": [100, 55, -20, 0, 35],
        ...         "q2": [30, -10, 15, 60, 20],
        ...         "q3": [-50, 0, 40, 80, 80],
        ...         "q4": [75, 55, 25, -10, -55],
        ...     }
        ... )
        >>> df.write_excel(  # doctest: +SKIP
        ...     table_style="Table Style Light 2",
        ...     # apply accounting format to all flavours of integer
        ...     dtype_formats={pl.INTEGER_DTYPES: "#,##0_);(#,##0)"},
        ...     sparklines={
        ...         # default options; just provide source cols
        ...         "trend": ["q1", "q2", "q3", "q4"],
        ...         # customized sparkline type, with positioning directive
        ...         "+/-": {
        ...             "columns": ["q1", "q2", "q3", "q4"],
        ...             "insert_after": "id",
        ...             "type": "win_loss",
        ...         },
        ...     },
        ...     conditional_formats={
        ...         # create a unified multi-column heatmap
        ...         ("q1", "q2", "q3", "q4"): {
        ...             "type": "2_color_scale",
        ...             "min_color": "#95b3d7",
        ...             "max_color": "#ffffff",
        ...         },
        ...     },
        ...     column_totals=["q1", "q2", "q3", "q4"],
        ...     row_totals=True,
        ...     hide_gridlines=True,
        ... )

        Export a table containing an Excel formula-based column that calculates a
        standardised Z-score, showing use of structured references in conjunction
        with positioning directives, column totals, and custom formatting.

        >>> df = pl.DataFrame(
        ...     {
        ...         "id": ["a123", "b345", "c567", "d789", "e101"],
        ...         "points": [99, 45, 50, 85, 35],
        ...     }
        ... )
        >>> df.write_excel(  # doctest: +SKIP
        ...     table_style={
        ...         "style": "Table Style Medium 15",
        ...         "first_column": True,
        ...     },
        ...     column_formats={
        ...         "id": {"font": "Consolas"},
        ...         "points": {"align": "center"},
        ...         "z-score": {"align": "center"},
        ...     },
        ...     column_totals="average",
        ...     formulas={
        ...         "z-score": {
        ...             # use structured references to refer to the table columns and \'totals\' row
        ...             "formula": "=STANDARDIZE([@points], [[#Totals],[points]], STDEV([points]))",
        ...             "insert_after": "points",
        ...             "return_dtype": pl.Float64,
        ...         }
        ...     },
        ...     hide_gridlines=True,
        ...     sheet_zoom=125,
        ... )
        """
    def write_ipc(
        self, file: str | Path | IO[bytes] | None, compression: IpcCompression = ...
    ) -> BytesIO | None:
        """
        Write to Arrow IPC binary stream or Feather file.

        See "File or Random Access format" in https://arrow.apache.org/docs/python/ipc.html.

        Parameters
        ----------
        file
            Path or writable file-like object to which the IPC data will be
            written. If set to `None`, the output is returned as a BytesIO object.
        compression : {\'uncompressed\', \'lz4\', \'zstd\'}
            Compression method. Defaults to "uncompressed".
        future
            Setting this to `True` will write Polars\' internal data structures that
            might not be available by other Arrow implementations.

            .. warning::
                This functionality is considered **unstable**. It may be changed
                at any point without it being considered a breaking change.

        Examples
        --------
        >>> import pathlib
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> path: pathlib.Path = dirpath / "new_file.arrow"
        >>> df.write_ipc(path)
        """
    def write_ipc_stream(
        self, file: str | Path | IO[bytes] | None, compression: IpcCompression = ...
    ) -> BytesIO | None:
        """
        Write to Arrow IPC record batch stream.

        See "Streaming format" in https://arrow.apache.org/docs/python/ipc.html.

        Parameters
        ----------
        file
            Path or writable file-like object to which the IPC record batch data will
            be written. If set to `None`, the output is returned as a BytesIO object.
        compression : {\'uncompressed\', \'lz4\', \'zstd\'}
            Compression method. Defaults to "uncompressed".

        Examples
        --------
        >>> import pathlib
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> path: pathlib.Path = dirpath / "new_file.arrow"
        >>> df.write_ipc_stream(path)
        """
    def write_parquet(self, file: str | Path | BytesIO) -> None:
        """
        Write to Apache Parquet file.

        Parameters
        ----------
        file
            File path or writable file-like object to which the result will be written.
        compression : {\'lz4\', \'uncompressed\', \'snappy\', \'gzip\', \'lzo\', \'brotli\', \'zstd\'}
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
        row_group_size
            Size of the row groups in number of rows. Defaults to 512^2 rows.
        data_page_size
            Size of the data page in bytes. Defaults to 1024^2 bytes.
        use_pyarrow
            Use C++ parquet implementation vs Rust parquet implementation.
            At the moment C++ supports more features.
        pyarrow_options
            Arguments passed to `pyarrow.parquet.write_table`.

            If you pass `partition_cols` here, the dataset will be written
            using `pyarrow.parquet.write_to_dataset`.
            The `partition_cols` parameter leads to write the dataset to a directory.
            Similar to Spark\'s partitioned datasets.

        Examples
        --------
        >>> import pathlib
        >>>
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> path: pathlib.Path = dirpath / "new_file.parquet"
        >>> df.write_parquet(path)

        We can use pyarrow with use_pyarrow_write_to_dataset=True
        to write partitioned datasets. The following example will
        write the first row to ../watermark=1/*.parquet and the
        other rows to ../watermark=2/*.parquet.

        >>> df = pl.DataFrame({"a": [1, 2, 3], "watermark": [1, 2, 2]})
        >>> path: pathlib.Path = dirpath / "partitioned_object"
        >>> df.write_parquet(
        ...     path,
        ...     use_pyarrow=True,
        ...     pyarrow_options={"partition_cols": ["watermark"]},
        ... )
        """
    def write_database(self, table_name: str, connection: ConnectionOrCursor | str) -> int:
        """
        Write the data in a Polars DataFrame to a database.

        .. versionadded:: 0.20.26
            Support for instantiated connection objects in addition to URI strings, and
            a new `engine_options` parameter.

        Parameters
        ----------
        table_name
            Schema-qualified name of the table to create or append to in the target
            SQL database. If your table name contains special characters, it should
            be quoted.
        connection
            An existing SQLAlchemy or ADBC connection against the target database, or
            a URI string that will be used to instantiate such a connection, such as:

            * "postgresql://user:pass@server:port/database"
            * "sqlite:////path/to/database.db"
        if_table_exists : {\'append\', \'replace\', \'fail\'}
            The insert mode:

            * \'replace\' will create a new database table, overwriting an existing one.
            * \'append\' will append to an existing table.
            * \'fail\' will fail if table already exists.
        engine : {\'sqlalchemy\', \'adbc\'}
            Select the engine to use for writing frame data; only necessary when
            supplying a URI string (defaults to \'sqlalchemy\' if unset)
        engine_options
            Additional options to pass to the engine\'s associated insert method:

            * "sqlalchemy" - currently inserts using Pandas\' `to_sql` method, though
              this will eventually be phased out in favour of a native solution.
            * "adbc" - inserts using the ADBC cursor\'s `adbc_ingest` method.

        Examples
        --------
        Insert into a temporary table using a PostgreSQL URI and the ADBC engine:

        >>> df.write_database(
        ...     table_name="target_table",
        ...     connection="postgresql://user:pass@server:port/database",
        ...     engine="adbc",
        ...     engine_options={"temporary": True},
        ... )  # doctest: +SKIP

        Insert into a table using a `pyodbc` SQLAlchemy connection to SQL Server
        that was instantiated with "fast_executemany=True" to improve performance:

        >>> pyodbc_uri = (
        ...     "mssql+pyodbc://user:pass@server:1433/test?"
        ...     "driver=ODBC+Driver+18+for+SQL+Server"
        ... )
        >>> engine = create_engine(pyodbc_uri, fast_executemany=True)  # doctest: +SKIP
        >>> df.write_database(
        ...     table_name="target_table",
        ...     connection=engine,
        ... )  # doctest: +SKIP

        Returns
        -------
        int
            The number of rows affected, if the driver provides this information.
            Otherwise, returns -1.
        """
    def write_delta(
        self, target: str | Path | deltalake.DeltaTable
    ) -> deltalake.table.TableMerger | None:
        """
        Write DataFrame as delta table.

        Parameters
        ----------
        target
            URI of a table or a DeltaTable object.
        mode : {\'error\', \'append\', \'overwrite\', \'ignore\', \'merge\'}
            How to handle existing data.

            - If \'error\', throw an error if the table already exists (default).
            - If \'append\', will add new data.
            - If \'overwrite\', will replace table with new data.
            - If \'ignore\', will not write anything if table already exists.
            - If \'merge\', return a `TableMerger` object to merge data from the DataFrame
              with the existing data.
        overwrite_schema
            If True, allows updating the schema of the table.

            .. deprecated:: 0.20.14
                Use the parameter `delta_write_options` instead and pass
                `{"schema_mode": "overwrite"}`.
        storage_options
            Extra options for the storage backends supported by `deltalake`.
            For cloud storages, this may include configurations for authentication etc.

            - See a list of supported storage options for S3 `here <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html#variants>`__.
            - See a list of supported storage options for GCS `here <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html#variants>`__.
            - See a list of supported storage options for Azure `here <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html#variants>`__.
        delta_write_options
            Additional keyword arguments while writing a Delta lake Table.
            See a list of supported write options `here <https://delta-io.github.io/delta-rs/api/delta_writer/#deltalake.write_deltalake>`__.
        delta_merge_options
            Keyword arguments which are required to `MERGE` a Delta lake Table.
            See a list of supported merge options `here <https://delta-io.github.io/delta-rs/api/delta_table/#deltalake.DeltaTable.merge>`__.

        Raises
        ------
        TypeError
            If the DataFrame contains unsupported data types.
        ArrowInvalidError
            If the DataFrame contains data types that could not be cast to their
            primitive type.
        TableNotFoundError
            If the delta table doesn\'t exist and MERGE action is triggered

        Notes
        -----
        The Polars data types :class:`Null` and :class:`Time` are not supported
        by the delta protocol specification and will raise a TypeError. Columns
        using The :class:`Categorical` data type will be converted to
        normal (non-categorical) strings when written.

        Polars columns are always nullable. To write data to a delta table with
        non-nullable columns, a custom pyarrow schema has to be passed to the
        `delta_write_options`. See the last example below.

        Examples
        --------
        Write a dataframe to the local filesystem as a Delta Lake table.

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> table_path = "/path/to/delta-table/"
        >>> df.write_delta(table_path)  # doctest: +SKIP

        Append data to an existing Delta Lake table on the local filesystem.
        Note that this will fail if the schema of the new data does not match the
        schema of the existing table.

        >>> df.write_delta(table_path, mode="append")  # doctest: +SKIP

        Overwrite a Delta Lake table as a new version.
        If the schemas of the new and old data are the same, specifying the
        `schema_mode` is not required.

        >>> existing_table_path = "/path/to/delta-table/"
        >>> df.write_delta(
        ...     existing_table_path,
        ...     mode="overwrite",
        ...     delta_write_options={"schema_mode": "overwrite"},
        ... )  # doctest: +SKIP

        Write a DataFrame as a Delta Lake table to a cloud object store like S3.

        >>> table_path = "s3://bucket/prefix/to/delta-table/"
        >>> df.write_delta(
        ...     table_path,
        ...     storage_options={
        ...         "AWS_REGION": "THE_AWS_REGION",
        ...         "AWS_ACCESS_KEY_ID": "THE_AWS_ACCESS_KEY_ID",
        ...         "AWS_SECRET_ACCESS_KEY": "THE_AWS_SECRET_ACCESS_KEY",
        ...     },
        ... )  # doctest: +SKIP

        Write DataFrame as a Delta Lake table with non-nullable columns.

        >>> import pyarrow as pa
        >>> existing_table_path = "/path/to/delta-table/"
        >>> df.write_delta(
        ...     existing_table_path,
        ...     delta_write_options={
        ...         "schema": pa.schema([pa.field("foo", pa.int64(), nullable=False)])
        ...     },
        ... )  # doctest: +SKIP

        Merge the DataFrame with an existing Delta Lake table.
        For all `TableMerger` methods, check the deltalake docs
        `here <https://delta-io.github.io/delta-rs/api/delta_table/delta_table_merger/>`__.

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> table_path = "/path/to/delta-table/"
        >>> (
        ...     df.write_delta(
        ...         "table_path",
        ...         mode="merge",
        ...         delta_merge_options={
        ...             "predicate": "s.foo = t.foo",
        ...             "source_alias": "s",
        ...             "target_alias": "t",
        ...         },
        ...     )
        ...     .when_matched_update_all()
        ...     .when_not_matched_insert_all()
        ...     .execute()
        ... )  # doctest: +SKIP
        """
    def estimated_size(self, unit: SizeUnit = ...) -> int | float:
        """
        Return an estimation of the total (heap) allocated size of the `DataFrame`.

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
        >>> df = pl.DataFrame(
        ...     {
        ...         "x": list(reversed(range(1_000_000))),
        ...         "y": [v / 1000 for v in range(1_000_000)],
        ...         "z": [str(v) for v in range(1_000_000)],
        ...     },
        ...     schema=[("x", pl.UInt32), ("y", pl.Float64), ("z", pl.String)],
        ... )
        >>> df.estimated_size()
        17888890
        >>> df.estimated_size("mb")
        17.0601749420166
        """
    def transpose(self) -> Self:
        """
        Transpose a DataFrame over the diagonal.

        Parameters
        ----------
        include_header
            If set, the column names will be added as first column.
        header_name
            If `include_header` is set, this determines the name of the column that will
            be inserted.
        column_names
            Optional iterable yielding strings or a string naming an existing column.
            These will name the value (non-header) columns in the transposed data.

        Notes
        -----
        This is a very expensive operation. Perhaps you can do it differently.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> df.transpose(include_header=True)
        shape: (2, 4)
        ┌────────┬──────────┬──────────┬──────────┐
        │ column ┆ column_0 ┆ column_1 ┆ column_2 │
        │ ---    ┆ ---      ┆ ---      ┆ ---      │
        │ str    ┆ i64      ┆ i64      ┆ i64      │
        ╞════════╪══════════╪══════════╪══════════╡
        │ a      ┆ 1        ┆ 2        ┆ 3        │
        │ b      ┆ 4        ┆ 5        ┆ 6        │
        └────────┴──────────┴──────────┴──────────┘

        Replace the auto-generated column names with a list

        >>> df.transpose(include_header=False, column_names=["x", "y", "z"])
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ x   ┆ y   ┆ z   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 2   ┆ 3   │
        │ 4   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┘

        Include the header as a separate column

        >>> df.transpose(
        ...     include_header=True, header_name="foo", column_names=["x", "y", "z"]
        ... )
        shape: (2, 4)
        ┌─────┬─────┬─────┬─────┐
        │ foo ┆ x   ┆ y   ┆ z   │
        │ --- ┆ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 2   ┆ 3   │
        │ b   ┆ 4   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┴─────┘

        Replace the auto-generated column with column names from a generator function

        >>> def name_generator():
        ...     base_name = "my_column_"
        ...     count = 0
        ...     while True:
        ...         yield f"{base_name}{count}"
        ...         count += 1
        >>> df.transpose(include_header=False, column_names=name_generator())
        shape: (2, 3)
        ┌─────────────┬─────────────┬─────────────┐
        │ my_column_0 ┆ my_column_1 ┆ my_column_2 │
        │ ---         ┆ ---         ┆ ---         │
        │ i64         ┆ i64         ┆ i64         │
        ╞═════════════╪═════════════╪═════════════╡
        │ 1           ┆ 2           ┆ 3           │
        │ 4           ┆ 5           ┆ 6           │
        └─────────────┴─────────────┴─────────────┘

        Use an existing column as the new column names

        >>> df = pl.DataFrame(dict(id=["i", "j", "k"], a=[1, 2, 3], b=[4, 5, 6]))
        >>> df.transpose(column_names="id")
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ i   ┆ j   ┆ k   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 2   ┆ 3   │
        │ 4   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┘
        >>> df.transpose(include_header=True, header_name="new_id", column_names="id")
        shape: (2, 4)
        ┌────────┬─────┬─────┬─────┐
        │ new_id ┆ i   ┆ j   ┆ k   │
        │ ---    ┆ --- ┆ --- ┆ --- │
        │ str    ┆ i64 ┆ i64 ┆ i64 │
        ╞════════╪═════╪═════╪═════╡
        │ a      ┆ 1   ┆ 2   ┆ 3   │
        │ b      ┆ 4   ┆ 5   ┆ 6   │
        └────────┴─────┴─────┴─────┘
        """
    def reverse(self) -> DataFrame:
        """
        Reverse the DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "key": ["a", "b", "c"],
        ...         "val": [1, 2, 3],
        ...     }
        ... )
        >>> df.reverse()
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
    def rename(self, mapping: dict[str, str] | Callable[[str], str]) -> DataFrame:
        """
        Rename column names.

        Parameters
        ----------
        mapping
            Key value pairs that map from old name to new name, or a function
            that takes the old name as input and returns the new name.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
        ... )
        >>> df.rename({"foo": "apple"})
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
        >>> df.rename(lambda column_name: "c" + column_name[1:])
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
    def insert_column(self, index: int, column: Series) -> Self:
        """
        Insert a Series at a certain column index.

        This operation is in place.

        Parameters
        ----------
        index
            Index at which to insert the new `Series` column.
        column
            `Series` to insert.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> s = pl.Series("baz", [97, 98, 99])
        >>> df.insert_column(1, s)
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ baz ┆ bar │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 97  ┆ 4   │
        │ 2   ┆ 98  ┆ 5   │
        │ 3   ┆ 99  ┆ 6   │
        └─────┴─────┴─────┘

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> s = pl.Series("d", [-2.5, 15, 20.5, 0])
        >>> df.insert_column(3, s)
        shape: (4, 4)
        ┌─────┬──────┬───────┬──────┐
        │ a   ┆ b    ┆ c     ┆ d    │
        │ --- ┆ ---  ┆ ---   ┆ ---  │
        │ i64 ┆ f64  ┆ bool  ┆ f64  │
        ╞═════╪══════╪═══════╪══════╡
        │ 1   ┆ 0.5  ┆ true  ┆ -2.5 │
        │ 2   ┆ 4.0  ┆ true  ┆ 15.0 │
        │ 3   ┆ 10.0 ┆ false ┆ 20.5 │
        │ 4   ┆ 13.0 ┆ true  ┆ 0.0  │
        └─────┴──────┴───────┴──────┘
        """
    def filter(
        self,
        *predicates: IntoExprColumn
        | Iterable[IntoExprColumn]
        | bool
        | list[bool]
        | np.ndarray[Any, Any],
        **constraints: Any,
    ) -> DataFrame:
        """
        Filter the rows in the DataFrame based on one or more predicate expressions.

        The original order of the remaining rows is preserved.

        Rows where the filter does not evaluate to True are discarded, including nulls.

        Parameters
        ----------
        predicates
            Expression(s) that evaluates to a boolean Series.
        constraints
            Column filters; use `name = value` to filter columns by the supplied value.
            Each constraint will behave the same as `pl.col(name).eq(value)`, and
            will be implicitly joined with the other filter conditions using `&`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )

        Filter on one condition:

        >>> df.filter(pl.col("foo") > 1)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 2   ┆ 7   ┆ b   │
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        Filter on multiple conditions, combined with and/or operators:

        >>> df.filter((pl.col("foo") < 3) & (pl.col("ham") == "a"))
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        >>> df.filter((pl.col("foo") == 1) | (pl.col("ham") == "c"))
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘

        Provide multiple filters using `*args` syntax:

        >>> df.filter(
        ...     pl.col("foo") <= 2,
        ...     ~pl.col("ham").is_in(["b", "c"]),
        ... )
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘

        Provide multiple filters using `**kwargs` syntax:

        >>> df.filter(foo=2, ham="b")
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 2   ┆ 7   ┆ b   │
        └─────┴─────┴─────┘
        """
    def glimpse(self) -> str | None:
        """
        Return a dense preview of the DataFrame.

        The formatting shows one line per column so that wide dataframes display
        cleanly. Each line shows the column name, the data type, and the first
        few values.

        Parameters
        ----------
        max_items_per_column
            Maximum number of items to show per column.
        max_colname_length
            Maximum length of the displayed column names; values that exceed this
            value are truncated with a trailing ellipsis.
        return_as_string
            If True, return the preview as a string instead of printing to stdout.

        See Also
        --------
        describe, head, tail

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1.0, 2.8, 3.0],
        ...         "b": [4, 5, None],
        ...         "c": [True, False, True],
        ...         "d": [None, "b", "c"],
        ...         "e": ["usd", "eur", None],
        ...         "f": [date(2020, 1, 1), date(2021, 1, 2), date(2022, 1, 1)],
        ...     }
        ... )
        >>> df.glimpse()
        Rows: 3
        Columns: 6
        $ a  <f64> 1.0, 2.8, 3.0
        $ b  <i64> 4, 5, None
        $ c <bool> True, False, True
        $ d  <str> None, \'b\', \'c\'
        $ e  <str> \'usd\', \'eur\', None
        $ f <date> 2020-01-01, 2021-01-02, 2022-01-01
        """
    def describe(self, percentiles: Sequence[float] | float | None = ...) -> DataFrame:
        """
        Summary statistics for a DataFrame.

        Parameters
        ----------
        percentiles
            One or more percentiles to include in the summary statistics.
            All values must be in the range `[0, 1]`.

        interpolation : {\'nearest\', \'higher\', \'lower\', \'midpoint\', \'linear\'}
            Interpolation method used when calculating percentiles.

        Notes
        -----
        The median is included by default as the 50% percentile.

        Warnings
        --------
        We do not guarantee the output of `describe` to be stable. It will show
        statistics that we deem informative, and may be updated in the future.
        Using `describe` programmatically (versus interactive exploration) is
        not recommended for this reason.

        See Also
        --------
        glimpse

        Examples
        --------
        >>> from datetime import date, time
        >>> df = pl.DataFrame(
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

        >>> df.describe()
        shape: (9, 7)
        ┌────────────┬──────────┬──────────┬──────────┬──────┬────────────┬──────────┐
        │ statistic  ┆ float    ┆ int      ┆ bool     ┆ str  ┆ date       ┆ time     │
        │ ---        ┆ ---      ┆ ---      ┆ ---      ┆ ---  ┆ ---        ┆ ---      │
        │ str        ┆ f64      ┆ f64      ┆ f64      ┆ str  ┆ str        ┆ str      │
        ╞════════════╪══════════╪══════════╪══════════╪══════╪════════════╪══════════╡
        │ count      ┆ 3.0      ┆ 2.0      ┆ 3.0      ┆ 3    ┆ 3          ┆ 3        │
        │ null_count ┆ 0.0      ┆ 1.0      ┆ 0.0      ┆ 0    ┆ 0          ┆ 0        │
        │ mean       ┆ 2.266667 ┆ 45.0     ┆ 0.666667 ┆ null ┆ 2021-07-02 ┆ 16:07:10 │
        │ std        ┆ 1.101514 ┆ 7.071068 ┆ null     ┆ null ┆ null       ┆ null     │
        │ min        ┆ 1.0      ┆ 40.0     ┆ 0.0      ┆ xx   ┆ 2020-01-01 ┆ 10:20:30 │
        │ 25%        ┆ 2.8      ┆ 40.0     ┆ null     ┆ null ┆ 2021-07-05 ┆ 14:45:50 │
        │ 50%        ┆ 2.8      ┆ 50.0     ┆ null     ┆ null ┆ 2021-07-05 ┆ 14:45:50 │
        │ 75%        ┆ 3.0      ┆ 50.0     ┆ null     ┆ null ┆ 2022-12-31 ┆ 23:15:10 │
        │ max        ┆ 3.0      ┆ 50.0     ┆ 1.0      ┆ zz   ┆ 2022-12-31 ┆ 23:15:10 │
        └────────────┴──────────┴──────────┴──────────┴──────┴────────────┴──────────┘

        Customize which percentiles are displayed, applying linear interpolation:

        >>> with pl.Config(tbl_rows=12):
        ...     df.describe(
        ...         percentiles=[0.1, 0.3, 0.5, 0.7, 0.9],
        ...         interpolation="linear",
        ...     )
        shape: (11, 7)
        ┌────────────┬──────────┬──────────┬──────────┬──────┬────────────┬──────────┐
        │ statistic  ┆ float    ┆ int      ┆ bool     ┆ str  ┆ date       ┆ time     │
        │ ---        ┆ ---      ┆ ---      ┆ ---      ┆ ---  ┆ ---        ┆ ---      │
        │ str        ┆ f64      ┆ f64      ┆ f64      ┆ str  ┆ str        ┆ str      │
        ╞════════════╪══════════╪══════════╪══════════╪══════╪════════════╪══════════╡
        │ count      ┆ 3.0      ┆ 2.0      ┆ 3.0      ┆ 3    ┆ 3          ┆ 3        │
        │ null_count ┆ 0.0      ┆ 1.0      ┆ 0.0      ┆ 0    ┆ 0          ┆ 0        │
        │ mean       ┆ 2.266667 ┆ 45.0     ┆ 0.666667 ┆ null ┆ 2021-07-02 ┆ 16:07:10 │
        │ std        ┆ 1.101514 ┆ 7.071068 ┆ null     ┆ null ┆ null       ┆ null     │
        │ min        ┆ 1.0      ┆ 40.0     ┆ 0.0      ┆ xx   ┆ 2020-01-01 ┆ 10:20:30 │
        │ 10%        ┆ 1.36     ┆ 41.0     ┆ null     ┆ null ┆ 2020-04-20 ┆ 11:13:34 │
        │ 30%        ┆ 2.08     ┆ 43.0     ┆ null     ┆ null ┆ 2020-11-26 ┆ 12:59:42 │
        │ 50%        ┆ 2.8      ┆ 45.0     ┆ null     ┆ null ┆ 2021-07-05 ┆ 14:45:50 │
        │ 70%        ┆ 2.88     ┆ 47.0     ┆ null     ┆ null ┆ 2022-02-07 ┆ 18:09:34 │
        │ 90%        ┆ 2.96     ┆ 49.0     ┆ null     ┆ null ┆ 2022-09-13 ┆ 21:33:18 │
        │ max        ┆ 3.0      ┆ 50.0     ┆ 1.0      ┆ zz   ┆ 2022-12-31 ┆ 23:15:10 │
        └────────────┴──────────┴──────────┴──────────┴──────┴────────────┴──────────┘
        """
    def get_column_index(self, name: str) -> int:
        """
        Find the index of a column by name.

        Parameters
        ----------
        name
            Name of the column to find.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
        ... )
        >>> df.get_column_index("ham")
        2
        """
    def replace_column(self, index: int, column: Series) -> Self:
        """
        Replace a column at an index location.

        This operation is in place.

        Parameters
        ----------
        index
            Column index.
        column
            Series that will replace the column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> s = pl.Series("apple", [10, 20, 30])
        >>> df.replace_column(0, s)
        shape: (3, 3)
        ┌───────┬─────┬─────┐
        │ apple ┆ bar ┆ ham │
        │ ---   ┆ --- ┆ --- │
        │ i64   ┆ i64 ┆ str │
        ╞═══════╪═════╪═════╡
        │ 10    ┆ 6   ┆ a   │
        │ 20    ┆ 7   ┆ b   │
        │ 30    ┆ 8   ┆ c   │
        └───────┴─────┴─────┘
        """
    def sort(self, by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr) -> DataFrame:
        """
        Sort the dataframe by the given columns.

        Parameters
        ----------
        by
            Column(s) to sort by. Accepts expression input. Strings are parsed as column
            names.
        *more_by
            Additional columns to sort by, specified as positional arguments.
        descending
            Sort in descending order. When sorting by multiple columns, can be specified
            per column by passing a sequence of booleans.
        nulls_last
            Place null values last; can specify a single boolean applying to all columns
            or a sequence of booleans for per-column control.
        multithreaded
            Sort using multiple threads.
        maintain_order
            Whether the order should be maintained if elements are equal.

        Examples
        --------
        Pass a single column name to sort by that column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": [6.0, 5.0, 4.0],
        ...         "c": ["a", "c", "b"],
        ...     }
        ... )
        >>> df.sort("a")
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

        >>> df.sort(pl.col("a") + pl.col("b") * 2, nulls_last=True)
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

        >>> df.sort(["c", "a"], descending=True)
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

        >>> df.sort("c", "a", descending=[False, True])
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
    def sql(self, query: str) -> Self:
        """
        Execute a SQL query against the DataFrame.

        .. versionadded:: 0.20.24

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
        * The SQL query executes in lazy mode before being collected and returned
          as a DataFrame.

        See Also
        --------
        SQLContext

        Examples
        --------
        >>> from datetime import date
        >>> df1 = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["zz", "yy", "xx"],
        ...         "c": [date(1999, 12, 31), date(2010, 10, 10), date(2077, 8, 8)],
        ...     }
        ... )

        Query the DataFrame using SQL:

        >>> df1.sql("SELECT c, b FROM self WHERE a > 1")
        shape: (2, 2)
        ┌────────────┬─────┐
        │ c          ┆ b   │
        │ ---        ┆ --- │
        │ date       ┆ str │
        ╞════════════╪═════╡
        │ 2010-10-10 ┆ yy  │
        │ 2077-08-08 ┆ xx  │
        └────────────┴─────┘

        Apply transformations to a DataFrame using SQL, aliasing "self" to "frame".

        >>> df1.sql(
        ...     query=\'\'\'
        ...         SELECT
        ...             a,
        ...             (a % 2 == 0) AS a_is_even,
        ...             CONCAT_WS(\':\', b, b) AS b_b,
        ...             EXTRACT(year FROM c) AS year,
        ...             0::float4 AS "zero",
        ...         FROM frame
        ...     \'\'\',
        ...     table_name="frame",
        ... )
        shape: (3, 5)
        ┌─────┬───────────┬───────┬──────┬──────┐
        │ a   ┆ a_is_even ┆ b_b   ┆ year ┆ zero │
        │ --- ┆ ---       ┆ ---   ┆ ---  ┆ ---  │
        │ i64 ┆ bool      ┆ str   ┆ i32  ┆ f32  │
        ╞═════╪═══════════╪═══════╪══════╪══════╡
        │ 1   ┆ false     ┆ zz:zz ┆ 1999 ┆ 0.0  │
        │ 2   ┆ true      ┆ yy:yy ┆ 2010 ┆ 0.0  │
        │ 3   ┆ false     ┆ xx:xx ┆ 2077 ┆ 0.0  │
        └─────┴───────────┴───────┴──────┴──────┘
        """
    def top_k(self, k: int) -> DataFrame:
        """
        Return the `k` largest rows.

        Parameters
        ----------
        k
            Number of rows to return.
        by
            Column(s) used to determine the top rows.
            Accepts expression input. Strings are parsed as column names.
        descending
            Consider the `k` smallest elements of the `by` column(s) (instead of the `k`
            largest). This can be specified per column by passing a sequence of
            booleans.

        nulls_last
            Place null values last.

            .. deprecated:: 0.20.31
                This parameter will be removed in the next breaking release.
                Null values will be considered lowest priority and will only be
                included if `k` is larger than the number of non-null elements.

        maintain_order
            Whether the order should be maintained if elements are equal.
            Note that if `true` streaming is not possible and performance might be
            worse since this requires a stable search.

            .. deprecated:: 0.20.31
                This parameter will be removed in the next breaking release.
                There will be no guarantees about the order of the output.

        See Also
        --------
        bottom_k

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [2, 1, 1, 3, 2, 1],
        ...     }
        ... )

        Get the rows which contain the 4 largest values in column b.

        >>> df.top_k(4, by="b")
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

        >>> df.top_k(4, by=["b", "a"])
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
    def bottom_k(self, k: int) -> DataFrame:
        """
        Return the `k` smallest rows.

        Parameters
        ----------
        k
            Number of rows to return.
        by
            Column(s) used to determine the bottom rows.
            Accepts expression input. Strings are parsed as column names.
        descending
            Consider the `k` largest elements of the `by` column(s) (instead of the `k`
            smallest). This can be specified per column by passing a sequence of
            booleans.

        nulls_last
            Place null values last.

            .. deprecated:: 0.20.31
                This parameter will be removed in the next breaking release.
                Null values will be considered lowest priority and will only be
                included if `k` is larger than the number of non-null elements.

        maintain_order
            Whether the order should be maintained if elements are equal.
            Note that if `true` streaming is not possible and performance might be
            worse since this requires a stable search.

            .. deprecated:: 0.20.31
                This parameter will be removed in the next breaking release.
                There will be no guarantees about the order of the output.

        See Also
        --------
        top_k

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "b", "c"],
        ...         "b": [2, 1, 1, 3, 2, 1],
        ...     }
        ... )

        Get the rows which contain the 4 smallest values in column b.

        >>> df.bottom_k(4, by="b")
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

        >>> df.bottom_k(4, by=["a", "b"])
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
    def equals(self, other: DataFrame) -> bool:
        """
        Check whether the DataFrame is equal to another DataFrame.

        Parameters
        ----------
        other
            DataFrame to compare with.
        null_equal
            Consider null values as equal.

        See Also
        --------
        assert_frame_equal

        Examples
        --------
        >>> df1 = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df2 = pl.DataFrame(
        ...     {
        ...         "foo": [3, 2, 1],
        ...         "bar": [8.0, 7.0, 6.0],
        ...         "ham": ["c", "b", "a"],
        ...     }
        ... )
        >>> df1.equals(df1)
        True
        >>> df1.equals(df2)
        False
        """
    def replace(self, column: str, new_column: Series) -> Self:
        """
        Replace a column by a new Series.

        .. deprecated:: 0.19.0
            Use :meth:`with_columns` instead, e.g.
            `df = df.with_columns(new_column.alias(column_name))`.

        Parameters
        ----------
        column
            Column to replace.
        new_column
            New column to insert.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> s = pl.Series([10, 20, 30])
        >>> df.replace("foo", s)  # works in-place!  # doctest: +SKIP
        shape: (3, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 10  ┆ 4   │
        │ 20  ┆ 5   │
        │ 30  ┆ 6   │
        └─────┴─────┘
        """
    def slice(self, offset: int, length: int | None = ...) -> Self:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.slice(1, 2)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 2   ┆ 7.0 ┆ b   │
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘
        """
    def head(self, n: int = ...) -> Self:
        """
        Get the first `n` rows.

        Parameters
        ----------
        n
            Number of rows to return. If a negative value is passed, return all rows
            except the last `abs(n)`.

        See Also
        --------
        tail, glimpse, slice

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> df.head(3)
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

        Pass a negative value to get all rows `except` the last `abs(n)`.

        >>> df.head(-3)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 2   ┆ 7   ┆ b   │
        └─────┴─────┴─────┘
        """
    def tail(self, n: int = ...) -> Self:
        """
        Get the last `n` rows.

        Parameters
        ----------
        n
            Number of rows to return. If a negative value is passed, return all rows
            except the first `abs(n)`.

        See Also
        --------
        head, slice

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 4, 5],
        ...         "bar": [6, 7, 8, 9, 10],
        ...         "ham": ["a", "b", "c", "d", "e"],
        ...     }
        ... )
        >>> df.tail(3)
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8   ┆ c   │
        │ 4   ┆ 9   ┆ d   │
        │ 5   ┆ 10  ┆ e   │
        └─────┴─────┴─────┘

        Pass a negative value to get all rows `except` the first `abs(n)`.

        >>> df.tail(-3)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 4   ┆ 9   ┆ d   │
        │ 5   ┆ 10  ┆ e   │
        └─────┴─────┴─────┘
        """
    def limit(self, n: int = ...) -> Self:
        """
        Get the first `n` rows.

        Alias for :func:`DataFrame.head`.

        Parameters
        ----------
        n
            Number of rows to return. If a negative value is passed, return all rows
            except the last `abs(n)`.

        See Also
        --------
        head
        """
    def drop_nulls(
        self, subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = ...
    ) -> DataFrame:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, None, 8],
        ...         "ham": ["a", "b", None],
        ...     }
        ... )

        The default behavior of this method is to drop rows where any single
        value of the row is null.

        >>> df.drop_nulls()
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
        >>> df.drop_nulls(subset=cs.integer())
        shape: (2, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ i64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 1   ┆ 6   ┆ a    │
        │ 3   ┆ 8   ┆ null │
        └─────┴─────┴──────┘

        Below are some additional examples that show how to drop null
        values based on other conditions.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [None, None, None, None],
        ...         "b": [1, 2, None, 1],
        ...         "c": [1, None, None, 1],
        ...     }
        ... )
        >>> df
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

        >>> df.filter(~pl.all_horizontal(pl.all().is_null()))
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

        Drop a column if all values are null:

        >>> df[[s.name for s in df if not (s.null_count() == df.height)]]
        shape: (4, 2)
        ┌──────┬──────┐
        │ b    ┆ c    │
        │ ---  ┆ ---  │
        │ i64  ┆ i64  │
        ╞══════╪══════╡
        │ 1    ┆ 1    │
        │ 2    ┆ null │
        │ null ┆ null │
        │ 1    ┆ 1    │
        └──────┴──────┘
        """
    def pipe(
        self, function: Callable[Concatenate[DataFrame, P], T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
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

        Notes
        -----
        It is recommended to use LazyFrame when piping operations, in order
        to fully take advantage of query optimization and parallelization.
        See :meth:`df.lazy() <polars.DataFrame.lazy>`.

        Examples
        --------
        >>> def cast_str_to_int(data, col_name):
        ...     return data.with_columns(pl.col(col_name).cast(pl.Int64))
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["10", "20", "30", "40"]})
        >>> df.pipe(cast_str_to_int, col_name="b")
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

        >>> df = pl.DataFrame({"b": [1, 2], "a": [3, 4]})
        >>> df
        shape: (2, 2)
        ┌─────┬─────┐
        │ b   ┆ a   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        └─────┴─────┘
        >>> df.pipe(lambda tdf: tdf.select(sorted(tdf.columns)))
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
    def with_row_index(self, name: str = ..., offset: int = ...) -> Self:
        """
        Add a row index as the first column in the DataFrame.

        Parameters
        ----------
        name
            Name of the index column.
        offset
            Start the index at this offset. Cannot be negative.

        Notes
        -----
        The resulting column does not have any special properties. It is a regular
        column of type `UInt32` (or `UInt64` in `polars-u64-idx`).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> df.with_row_index()
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
        >>> df.with_row_index("id", offset=1000)
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

        >>> df.select(
        ...     pl.int_range(pl.len(), dtype=pl.UInt32).alias("index"),
        ...     pl.all(),
        ... )
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
    def with_row_count(self, name: str = ..., offset: int = ...) -> Self:
        """
        Add a column at index 0 that counts the rows.

        .. deprecated:: 0.20.4
            Use :meth:`with_row_index` instead.
            Note that the default column name has changed from \'row_nr\' to \'index\'.

        Parameters
        ----------
        name
            Name of the column to add.
        offset
            Start the row count at this offset. Default = 0

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> df.with_row_count()  # doctest: +SKIP
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
    def group_by(self, *by: IntoExpr | Iterable[IntoExpr], **named_by: IntoExpr) -> GroupBy:
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
            Settings this to `True` blocks the possibility
            to run on the streaming engine.

            .. note::
                Within each group, the order of rows is always preserved, regardless
                of this argument.
        **named_by
            Additional columns to group by, specified as keyword arguments.
            The columns will be renamed to the keyword used.

        Returns
        -------
        GroupBy
            Object which can be used to perform aggregations.

        Examples
        --------
        Group by one column and call `agg` to compute the grouped sum of another
        column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "c"],
        ...         "b": [1, 2, 1, 3, 3],
        ...         "c": [5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> df.group_by("a").agg(pl.col("b").sum())  # doctest: +IGNORE_RESULT
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

        >>> df.group_by("a", maintain_order=True).agg(pl.col("c"))
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

        >>> df.group_by(["a", "b"]).agg(pl.max("c"))  # doctest: +IGNORE_RESULT
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

        >>> df.group_by("a", pl.col("b") // 2).agg(pl.col("c").mean())  # doctest: +SKIP
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

        The `GroupBy` object returned by this method is iterable, returning the name
        and data of each group.

        >>> for name, data in df.group_by("a"):  # doctest: +SKIP
        ...     print(name)
        ...     print(data)
        a
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 5   │
        │ a   ┆ 1   ┆ 3   │
        └─────┴─────┴─────┘
        b
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 2   ┆ 4   │
        │ b   ┆ 3   ┆ 2   │
        └─────┴─────┴─────┘
        c
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ c   ┆ 3   ┆ 1   │
        └─────┴─────┴─────┘
        """
    def rolling(self, index_column: IntoExpr) -> RollingGroupBy:
        """
        Create rolling groups based on a temporal or integer column.

        Different from a `group_by_dynamic` the windows are now determined by the
        individual values and are not of constant intervals. For constant intervals use
        :func:`DataFrame.group_by_dynamic`.

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

            In case of a rolling operation on indices, dtype needs to be one of
            {UInt32, UInt64, Int32, Int64}. Note that the first three get temporarily
            cast to Int64, so if performance matters use an Int64 column.
        period
            Length of the window - must be non-negative.
        offset
            Offset of the window. Default is `-period`.
        closed : {\'right\', \'left\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).
        group_by
            Also group by this column/these columns
        check_sorted
            Check whether `index_column` is sorted (or, if `group_by` is given,
            check whether it\'s sorted within each group).
            When the `group_by` argument is given, polars can not check sortedness
            by the metadata and has to do a full scan on the index column to
            verify data is sorted. This is expensive. If you are sure the
            data within the groups is sorted, you can set this to `False`.
            Doing so incorrectly will lead to incorrect output

            .. deprecated:: 0.20.31
                Sortedness is now verified in a quick manner, you can safely remove
                this argument.

        Returns
        -------
        RollingGroupBy
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
        >>> df = pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]}).with_columns(
        ...     pl.col("dt").str.strptime(pl.Datetime).set_sorted()
        ... )
        >>> out = df.rolling(index_column="dt", period="2d").agg(
        ...     [
        ...         pl.sum("a").alias("sum_a"),
        ...         pl.min("a").alias("min_a"),
        ...         pl.max("a").alias("max_a"),
        ...     ]
        ... )
        >>> assert out["sum_a"].to_list() == [3, 10, 15, 24, 11, 1]
        >>> assert out["max_a"].to_list() == [3, 7, 7, 9, 9, 1]
        >>> assert out["min_a"].to_list() == [3, 3, 3, 3, 2, 1]
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
    def group_by_dynamic(self, index_column: IntoExpr) -> DynamicGroupBy:
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
            length of the window, if None it will equal \'every\'
        offset
            offset of the window, does not take effect if `start_by` is \'datapoint\'.
            Defaults to negative `every`.
        truncate
            truncate the time value to the window lower bound

            .. deprecated:: 0.19.4
                Use `label` instead.
        include_boundaries
            Add the lower and upper bound of the window to the "_lower_boundary" and
            "_upper_boundary" columns. This will impact performance because it\'s harder to
            parallelize
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).
        label : {\'left\', \'right\', \'datapoint\'}
            Define which label to use for the window:

            - \'left\': lower boundary of the window
            - \'right\': upper boundary of the window
            - \'datapoint\': the first value of the index column in the given window.
              If you don\'t need the label to be at one of the boundaries, choose this
              option for maximum performance
        group_by
            Also group by this column/these columns
        start_by : {\'window\', \'datapoint\', \'monday\', \'tuesday\', \'wednesday\', \'thursday\', \'friday\', \'saturday\', \'sunday\'}
            The strategy to determine the start of the first window by.

            * \'window\': Start by taking the earliest timestamp, truncating it with
              `every`, and then adding `offset`.
              Note that weekly windows start on Monday.
            * \'datapoint\': Start from the first encountered data point.
            * a day of the week (only takes effect if `every` contains `\'w\'`):

              * \'monday\': Start the window on the Monday before the first data point.
              * \'tuesday\': Start the window on the Tuesday before the first data point.
              * ...
              * \'sunday\': Start the window on the Sunday before the first data point.

              The resulting window is then shifted back until the earliest datapoint
              is in or in front of it.
        check_sorted
            Check whether `index_column` is sorted (or, if `group_by` is given,
            check whether it\'s sorted within each group).
            When the `group_by` argument is given, polars can not check sortedness
            by the metadata and has to do a full scan on the index column to
            verify data is sorted. This is expensive. If you are sure the
            data within the groups is sorted, you can set this to `False`.
            Doing so incorrectly will lead to incorrect output

            .. deprecated:: 0.20.31
                Sortedness is now verified in a quick manner, you can safely remove
                this argument.

        Returns
        -------
        DynamicGroupBy
            Object you can call `.agg` on to aggregate by groups, the result
            of which will be sorted by `index_column` (but note that if `group_by` columns are
            passed, it will only be sorted within each group).

        See Also
        --------
        rolling

        Notes
        -----
        1) If you\'re coming from pandas, then

           .. code-block:: python

               # polars
               df.group_by_dynamic("ts", every="1d").agg(pl.col("value").sum())

           is equivalent to

           .. code-block:: python

               # pandas
               df.set_index("ts").resample("D")["value"].sum().reset_index()

           though note that, unlike pandas, polars doesn\'t add extra rows for empty
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
        >>> df = pl.DataFrame(
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
        >>> df
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

        Group by windows of 1 hour starting at 2021-12-16 00:00:00.

        >>> df.group_by_dynamic("time", every="1h", closed="right").agg(pl.col("n"))
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

        >>> df.group_by_dynamic(
        ...     "time", every="1h", include_boundaries=True, closed="right"
        ... ).agg(pl.col("n").mean())
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

        >>> df.group_by_dynamic("time", every="1h", closed="left").agg(pl.col("n"))
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

        >>> df.group_by_dynamic("time", every="1h", closed="both").agg(pl.col("n"))
        shape: (5, 2)
        ┌─────────────────────┬───────────┐
        │ time                ┆ n         │
        │ ---                 ┆ ---       │
        │ datetime[μs]        ┆ list[i64] │
        ╞═════════════════════╪═══════════╡
        │ 2021-12-15 23:00:00 ┆ [0]       │
        │ 2021-12-16 00:00:00 ┆ [0, 1, 2] │
        │ 2021-12-16 01:00:00 ┆ [2, 3, 4] │
        │ 2021-12-16 02:00:00 ┆ [4, 5, 6] │
        │ 2021-12-16 03:00:00 ┆ [6]       │
        └─────────────────────┴───────────┘

        Dynamic group bys can also be combined with grouping on normal keys

        >>> df = df.with_columns(groups=pl.Series(["a", "a", "a", "b", "b", "a", "a"]))
        >>> df
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
        >>> df.group_by_dynamic(
        ...     "time",
        ...     every="1h",
        ...     closed="both",
        ...     group_by="groups",
        ...     include_boundaries=True,
        ... ).agg(pl.col("n"))
        shape: (7, 5)
        ┌────────┬─────────────────────┬─────────────────────┬─────────────────────┬───────────┐
        │ groups ┆ _lower_boundary     ┆ _upper_boundary     ┆ time                ┆ n         │
        │ ---    ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---       │
        │ str    ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ list[i64] │
        ╞════════╪═════════════════════╪═════════════════════╪═════════════════════╪═══════════╡
        │ a      ┆ 2021-12-15 23:00:00 ┆ 2021-12-16 00:00:00 ┆ 2021-12-15 23:00:00 ┆ [0]       │
        │ a      ┆ 2021-12-16 00:00:00 ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 00:00:00 ┆ [0, 1, 2] │
        │ a      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ [2]       │
        │ a      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ [5, 6]    │
        │ a      ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 04:00:00 ┆ 2021-12-16 03:00:00 ┆ [6]       │
        │ b      ┆ 2021-12-16 01:00:00 ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 01:00:00 ┆ [3, 4]    │
        │ b      ┆ 2021-12-16 02:00:00 ┆ 2021-12-16 03:00:00 ┆ 2021-12-16 02:00:00 ┆ [4]       │
        └────────┴─────────────────────┴─────────────────────┴─────────────────────┴───────────┘

        Dynamic group by on an index column

        >>> df = pl.DataFrame(
        ...     {
        ...         "idx": pl.int_range(0, 6, eager=True),
        ...         "A": ["A", "A", "B", "B", "B", "C"],
        ...     }
        ... )
        >>> (
        ...     df.group_by_dynamic(
        ...         "idx",
        ...         every="2i",
        ...         period="3i",
        ...         include_boundaries=True,
        ...         closed="right",
        ...     ).agg(pl.col("A").alias("A_agg_list"))
        ... )
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
    def upsample(self, time_column: str) -> Self:
        """
        Upsample a DataFrame at a regular frequency.

        The `every` and `offset` arguments are created with
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

        - "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds


        By "calendar day", we mean the corresponding time on the next day (which may
        not be 24 hours, due to daylight savings). Similarly for "calendar week",
        "calendar month", "calendar quarter", and "calendar year".

        Parameters
        ----------
        time_column
            Time column will be used to determine a date_range.
            Note that this column has to be sorted for the output to make sense.
        every
            Interval will start \'every\' duration.
        offset
            Change the start of the date_range by this offset.

            .. deprecated:: 0.20.19
                This argument is deprecated and will be removed in the next breaking
                release. Instead, chain `upsample` with `dt.offset_by`.
        group_by
            First group by these columns and then upsample for every group.
        maintain_order
            Keep the ordering predictable. This is slower.

        Returns
        -------
        DataFrame
            Result will be sorted by `time_column` (but note that if `group_by` columns
            are passed, it will only be sorted within each group).

        Examples
        --------
        Upsample a DataFrame by a certain interval.

        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "time": [
        ...             datetime(2021, 2, 1),
        ...             datetime(2021, 4, 1),
        ...             datetime(2021, 5, 1),
        ...             datetime(2021, 6, 1),
        ...         ],
        ...         "groups": ["A", "B", "A", "B"],
        ...         "values": [0, 1, 2, 3],
        ...     }
        ... ).set_sorted("time")
        >>> df.upsample(
        ...     time_column="time", every="1mo", group_by="groups", maintain_order=True
        ... ).select(pl.all().forward_fill())
        shape: (7, 3)
        ┌─────────────────────┬────────┬────────┐
        │ time                ┆ groups ┆ values │
        │ ---                 ┆ ---    ┆ ---    │
        │ datetime[μs]        ┆ str    ┆ i64    │
        ╞═════════════════════╪════════╪════════╡
        │ 2021-02-01 00:00:00 ┆ A      ┆ 0      │
        │ 2021-03-01 00:00:00 ┆ A      ┆ 0      │
        │ 2021-04-01 00:00:00 ┆ A      ┆ 0      │
        │ 2021-05-01 00:00:00 ┆ A      ┆ 2      │
        │ 2021-04-01 00:00:00 ┆ B      ┆ 1      │
        │ 2021-05-01 00:00:00 ┆ B      ┆ 1      │
        │ 2021-06-01 00:00:00 ┆ B      ┆ 3      │
        └─────────────────────┴────────┴────────┘
        """
    def join_asof(self, other: DataFrame) -> DataFrame:
        """
        Perform an asof join.

        This is similar to a left-join except that we match on nearest key rather than
        equal keys.

        Both DataFrames must be sorted by the asof_join key.

        For each row in the left DataFrame:

          - A "backward" search selects the last row in the right DataFrame whose
            \'on\' key is less than or equal to the left\'s key.

          - A "forward" search selects the first row in the right DataFrame whose
            \'on\' key is greater than or equal to the left\'s key.

          - A "nearest" search selects the last row in the right DataFrame whose value
            is nearest to the left\'s key. String keys are not currently supported for a
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
            join on these columns before doing asof join
        by_left
            join on these columns before doing asof join
        by_right
            join on these columns before doing asof join
        strategy : {\'backward\', \'forward\', \'nearest\'}
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

        Examples
        --------
        >>> from datetime import date
        >>> gdp = pl.DataFrame(
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
        >>> gdp
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

        >>> population = pl.DataFrame(
        ...     {
        ...         "date": [date(2016, 3, 1), date(2018, 8, 1), date(2019, 1, 1)],
        ...         "population": [82.19, 82.66, 83.12],
        ...     }
        ... ).sort("date")
        >>> population
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

        Note how the dates don\'t quite match. If we join them using `join_asof` and
        `strategy=\'backward\'`, then each date from `population` which doesn\'t have an
        exact match is matched with the closest earlier date from `gdp`:

        >>> population.join_asof(gdp, on="date", strategy="backward")
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

        If we instead use `strategy=\'forward\'`, then each date from `population` which
        doesn\'t have an exact match is matched with the closest later date from `gdp`:

        >>> population.join_asof(gdp, on="date", strategy="forward")
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

        Finally, `strategy=\'nearest\'` gives us a mix of the two results above, as each
        date from `population` which doesn\'t have an exact match is matched with the
        closest date from `gdp`, regardless of whether it\'s earlier or later:

        >>> population.join_asof(gdp, on="date", strategy="nearest")
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
        >>> gdp2 = pl.DataFrame(
        ...     {
        ...         "country": ["Germany"] * 5 + ["Netherlands"] * 5,
        ...         "date": pl.concat([gdp_dates, gdp_dates]),
        ...         "gdp": [4164, 4411, 4566, 4696, 4827, 784, 833, 914, 910, 909],
        ...     }
        ... ).sort("country", "date")
        >>>
        >>> gdp2
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
        >>> pop2 = pl.DataFrame(
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
        >>> pop2
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
        >>> pop2.join_asof(gdp2, by="country", on="date", strategy="nearest")
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
    def join(
        self,
        other: DataFrame,
        on: str | Expr | Sequence[str | Expr] | None = ...,
        how: JoinStrategy = ...,
    ) -> DataFrame:
        """
        Join in SQL-like fashion.

        Parameters
        ----------
        other
            DataFrame to join with.
        on
            Name(s) of the join columns in both DataFrames.
        how : {\'inner\', \'left\', \'full\', \'semi\', \'anti\', \'cross\'}
            Join strategy.

            * *inner*
                Returns rows that have matching values in both tables
            * *left*
                Returns all rows from the left table, and the matched rows from the
                right table
            * *full*
                 Returns all rows when there is a match in either left or right table
            * *cross*
                 Returns the Cartesian product of rows from both tables
            * *semi*
                 Filter rows that have a match in the right table.
            * *anti*
                 Filter rows that do not have a match in the right table.

            .. note::
                A left join preserves the row order of the left DataFrame.
        left_on
            Name(s) of the left join column(s).
        right_on
            Name(s) of the right join column(s).
        suffix
            Suffix to append to columns with a duplicate name.
        validate: {\'m:m\', \'m:1\', \'1:m\', \'1:1\'}
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

        Returns
        -------
        DataFrame

        See Also
        --------
        join_asof

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> other_df = pl.DataFrame(
        ...     {
        ...         "apple": ["x", "y", "z"],
        ...         "ham": ["a", "b", "d"],
        ...     }
        ... )
        >>> df.join(other_df, on="ham")
        shape: (2, 4)
        ┌─────┬─────┬─────┬───────┐
        │ foo ┆ bar ┆ ham ┆ apple │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ i64 ┆ f64 ┆ str ┆ str   │
        ╞═════╪═════╪═════╪═══════╡
        │ 1   ┆ 6.0 ┆ a   ┆ x     │
        │ 2   ┆ 7.0 ┆ b   ┆ y     │
        └─────┴─────┴─────┴───────┘

        >>> df.join(other_df, on="ham", how="full")
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

        >>> df.join(other_df, on="ham", how="left", coalesce=True)
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

        >>> df.join(other_df, on="ham", how="semi")
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6.0 ┆ a   │
        │ 2   ┆ 7.0 ┆ b   │
        └─────┴─────┴─────┘

        >>> df.join(other_df, on="ham", how="anti")
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ f64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8.0 ┆ c   │
        └─────┴─────┴─────┘

        Notes
        -----
        For joining on columns with categorical data, see :class:`polars.StringCache`.
        """
    def map_rows(
        self, function: Callable[[tuple[Any, ...]], Any], return_dtype: PolarsDataType | None = ...
    ) -> DataFrame:
        """
        Apply a custom/user-defined function (UDF) over the rows of the DataFrame.

        .. warning::
            This method is much slower than the native expressions API.
            Only use it if you cannot implement your logic otherwise.

        The UDF will receive each row as a tuple of values: `udf(row)`.

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
            Output type of the operation. If none given, Polars tries to infer the type.
        inference_size
            Only used in the case when the custom function returns rows.
            This uses the first `n` rows to determine the output schema.

        Notes
        -----
        * The frame-level `map_rows` cannot track column names (as the UDF is a
          black-box that may arbitrarily drop, rearrange, transform, or add new
          columns); if you want to apply a UDF such that column names are preserved,
          you should use the expression-level `map_elements` syntax instead.

        * If your function is expensive and you don\'t want it to be called more than
          once for a given input, consider applying an `@lru_cache` decorator to it.
          If your data is suitable you may achieve *significant* speedups.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [-1, 5, 8]})

        Return a DataFrame by mapping each row to a tuple:

        >>> df.map_rows(lambda t: (t[0] * 2, t[1] * 3))
        shape: (3, 2)
        ┌──────────┬──────────┐
        │ column_0 ┆ column_1 │
        │ ---      ┆ ---      │
        │ i64      ┆ i64      │
        ╞══════════╪══════════╡
        │ 2        ┆ -3       │
        │ 4        ┆ 15       │
        │ 6        ┆ 24       │
        └──────────┴──────────┘

        However, it is much better to implement this with a native expression:

        >>> df.select(
        ...     pl.col("foo") * 2,
        ...     pl.col("bar") * 3,
        ... )  # doctest: +IGNORE_RESULT

        Return a DataFrame with a single column by mapping each row to a scalar:

        >>> df.map_rows(lambda t: (t[0] * 2 + t[1]))  # doctest: +SKIP
        shape: (3, 1)
        ┌───────┐
        │ apply │
        │ ---   │
        │ i64   │
        ╞═══════╡
        │ 1     │
        │ 9     │
        │ 14    │
        └───────┘

        In this case it is better to use the following native expression:

        >>> df.select(pl.col("foo") * 2 + pl.col("bar"))  # doctest: +IGNORE_RESULT
        """
    def hstack(self, columns: list[Series] | DataFrame) -> Self:
        """
        Return a new DataFrame grown horizontally by stacking multiple Series to it.

        Parameters
        ----------
        columns
            Series to stack.
        in_place
            Modify in place.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> x = pl.Series("apple", [10, 20, 30])
        >>> df.hstack([x])
        shape: (3, 4)
        ┌─────┬─────┬─────┬───────┐
        │ foo ┆ bar ┆ ham ┆ apple │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ i64 ┆ i64 ┆ str ┆ i64   │
        ╞═════╪═════╪═════╪═══════╡
        │ 1   ┆ 6   ┆ a   ┆ 10    │
        │ 2   ┆ 7   ┆ b   ┆ 20    │
        │ 3   ┆ 8   ┆ c   ┆ 30    │
        └─────┴─────┴─────┴───────┘
        """
    def vstack(self, other: DataFrame) -> Self:
        """
        Grow this DataFrame vertically by stacking a DataFrame to it.

        Parameters
        ----------
        other
            DataFrame to stack.
        in_place
            Modify in place.

        See Also
        --------
        extend

        Examples
        --------
        >>> df1 = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2],
        ...         "bar": [6, 7],
        ...         "ham": ["a", "b"],
        ...     }
        ... )
        >>> df2 = pl.DataFrame(
        ...     {
        ...         "foo": [3, 4],
        ...         "bar": [8, 9],
        ...         "ham": ["c", "d"],
        ...     }
        ... )
        >>> df1.vstack(df2)
        shape: (4, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        │ 2   ┆ 7   ┆ b   │
        │ 3   ┆ 8   ┆ c   │
        │ 4   ┆ 9   ┆ d   │
        └─────┴─────┴─────┘
        """
    def extend(self, other: DataFrame) -> Self:
        """
        Extend the memory backed by this `DataFrame` with the values from `other`.

        Different from `vstack` which adds the chunks from `other` to the chunks of
        this `DataFrame`, `extend` appends the data from `other` to the underlying
        memory locations and thus may cause a reallocation.

        If this does not cause a reallocation, the resulting data structure will not
        have any extra chunks and thus will yield faster queries.

        Prefer `extend` over `vstack` when you want to do a query after a single
        append. For instance, during online operations where you add `n` rows and rerun
        a query.

        Prefer `vstack` over `extend` when you want to append many times before
        doing a query. For instance, when you read in multiple files and want to store
        them in a single `DataFrame`. In the latter case, finish the sequence of
        `vstack` operations with a `rechunk`.

        Parameters
        ----------
        other
            DataFrame to vertically add.

        Warnings
        --------
        This method modifies the dataframe in-place. The dataframe is returned for
        convenience only.

        See Also
        --------
        vstack

        Examples
        --------
        >>> df1 = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df2 = pl.DataFrame({"foo": [10, 20, 30], "bar": [40, 50, 60]})
        >>> df1.extend(df2)
        shape: (6, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        │ 10  ┆ 40  │
        │ 20  ┆ 50  │
        │ 30  ┆ 60  │
        └─────┴─────┘
        """
    def drop(self, *columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector]) -> DataFrame:
        """
        Remove columns from the dataframe.

        Parameters
        ----------
        *columns
            Names of the columns that should be removed from the dataframe.
            Accepts column selector input.

        Examples
        --------
        Drop a single column by passing the name of that column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.drop("ham")
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

        Drop multiple columns by passing a list of column names.

        >>> df.drop(["bar", "ham"])
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

        Drop multiple columns by passing a selector.

        >>> import polars.selectors as cs
        >>> df.drop(cs.numeric())
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

        >>> df.drop("foo", "ham")
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
    def drop_in_place(self, name: str) -> Series:
        """
        Drop a single column in-place and return the dropped column.

        Parameters
        ----------
        name
            Name of the column to drop.

        Returns
        -------
        Series
            The dropped column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.drop_in_place("ham")
        shape: (3,)
        Series: \'ham\' [str]
        [
            "a"
            "b"
            "c"
        ]
        """
    def cast(
        self,
        dtypes: Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType] | PolarsDataType,
    ) -> DataFrame:
        """
        Cast DataFrame column(s) to the specified dtype(s).

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
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6.0, 7.0, 8.0],
        ...         "ham": [date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)],
        ...     }
        ... )

        Cast specific frame columns to the specified dtypes:

        >>> df.cast({"foo": pl.Float32, "bar": pl.UInt8})
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

        >>> df.cast({pl.Date: pl.Datetime})
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
        >>> df.cast({cs.numeric(): pl.UInt32, cs.temporal(): pl.String})
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

        >>> df.cast(pl.String).to_dict(as_series=False)
        {\'foo\': [\'1\', \'2\', \'3\'],
         \'bar\': [\'6.0\', \'7.0\', \'8.0\'],
         \'ham\': [\'2020-01-02\', \'2021-03-04\', \'2022-05-06\']}
        """
    def clear(self, n: int = ...) -> Self:
        """
        Create an empty (n=0) or `n`-row null-filled (n>0) copy of the DataFrame.

        Returns a `n`-row null-filled DataFrame with an identical schema.
        `n` can be greater than the current number of rows in the DataFrame.

        Parameters
        ----------
        n
            Number of (null-filled) rows to return in the cleared frame.

        See Also
        --------
        clone : Cheap deepcopy/clone.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... )
        >>> df.clear()
        shape: (0, 3)
        ┌─────┬─────┬──────┐
        │ a   ┆ b   ┆ c    │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ f64 ┆ bool │
        ╞═════╪═════╪══════╡
        └─────┴─────┴──────┘

        >>> df.clear(n=2)
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
    def clone(self) -> Self:
        """
        Create a copy of this DataFrame.

        This is a cheap operation that does not copy data.

        See Also
        --------
        clear : Create an empty copy of the current DataFrame, with identical
            schema but no data.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> df.clone()
        shape: (4, 3)
        ┌─────┬──────┬───────┐
        │ a   ┆ b    ┆ c     │
        │ --- ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ bool  │
        ╞═════╪══════╪═══════╡
        │ 1   ┆ 0.5  ┆ true  │
        │ 2   ┆ 4.0  ┆ true  │
        │ 3   ┆ 10.0 ┆ false │
        │ 4   ┆ 13.0 ┆ true  │
        └─────┴──────┴───────┘
        """
    def get_columns(self) -> list[Series]:
        """
        Get the DataFrame as a List of Series.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df.get_columns()
        [shape: (3,)
        Series: \'foo\' [i64]
        [
                1
                2
                3
        ], shape: (3,)
        Series: \'bar\' [i64]
        [
                4
                5
                6
        ]]

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> df.get_columns()
        [shape: (4,)
        Series: \'a\' [i64]
        [
            1
            2
            3
            4
        ], shape: (4,)
        Series: \'b\' [f64]
        [
            0.5
            4.0
            10.0
            13.0
        ], shape: (4,)
        Series: \'c\' [bool]
        [
            true
            true
            false
            true
        ]]
        """
    def get_column(self, name: str) -> Series:
        """
        Get a single column by name.

        Parameters
        ----------
        name : str
            Name of the column to retrieve.

        Returns
        -------
        Series

        See Also
        --------
        to_series

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df.get_column("foo")
        shape: (3,)
        Series: \'foo\' [i64]
        [
                1
                2
                3
        ]
        """
    def fill_null(
        self,
        value: Any | Expr | None = ...,
        strategy: FillNullStrategy | None = ...,
        limit: int | None = ...,
    ) -> DataFrame:
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
        matches_supertype
            Fill all matching supertype of the fill `value`.

        Returns
        -------
        DataFrame
            DataFrame with None values replaced by the filling strategy.

        See Also
        --------
        fill_nan

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 4],
        ...         "b": [0.5, 4, None, 13],
        ...     }
        ... )
        >>> df.fill_null(99)
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
        >>> df.fill_null(strategy="forward")
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

        >>> df.fill_null(strategy="max")
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

        >>> df.fill_null(strategy="zero")
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
    def fill_nan(self, value: Expr | int | float | None) -> DataFrame:
        """
        Fill floating point NaN values by an Expression evaluation.

        Parameters
        ----------
        value
            Value with which to replace NaN values.

        Returns
        -------
        DataFrame
            DataFrame with NaN values replaced by the given value.

        Warnings
        --------
        Note that floating point NaNs (Not a Number) are not missing values.
        To replace missing values, use :func:`fill_null`.

        See Also
        --------
        fill_null

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1.5, 2, float("nan"), 4],
        ...         "b": [0.5, 4, float("nan"), 13],
        ...     }
        ... )
        >>> df.fill_nan(99)
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
    def explode(
        self, columns: str | Expr | Sequence[str | Expr], *more_columns: str | Expr
    ) -> DataFrame:
        """
        Explode the dataframe to long format by exploding the given columns.

        Parameters
        ----------
        columns
            Column names, expressions, or a selector defining them. The underlying
            columns being exploded must be of the `List` or `Array` data type.
        *more_columns
            Additional names of columns to explode, specified as positional arguments.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "letters": ["a", "a", "b", "c"],
        ...         "numbers": [[1], [2, 3], [4, 5], [6, 7, 8]],
        ...     }
        ... )
        >>> df
        shape: (4, 2)
        ┌─────────┬───────────┐
        │ letters ┆ numbers   │
        │ ---     ┆ ---       │
        │ str     ┆ list[i64] │
        ╞═════════╪═══════════╡
        │ a       ┆ [1]       │
        │ a       ┆ [2, 3]    │
        │ b       ┆ [4, 5]    │
        │ c       ┆ [6, 7, 8] │
        └─────────┴───────────┘
        >>> df.explode("numbers")
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
    def pivot(self) -> Self:
        """
        Create a spreadsheet-style pivot table as a DataFrame.

        Only available in eager mode. See "Examples" section below for how to do a
        "lazy pivot" if you know the unique column values in advance.

        Parameters
        ----------
        values
            Column values to aggregate. If None, all remaining columns will be used.
        index
            One or multiple keys to group by.
        columns
            Name of the column(s) whose values will be used as the header of the output
            DataFrame.
        aggregate_function
            Choose from:

            - None: no aggregation takes place, will raise error if multiple values are in group.
            - A predefined aggregate function string, one of
              {\'min\', \'max\', \'first\', \'last\', \'sum\', \'mean\', \'median\', \'len\'}
            - An expression to do the aggregation.
        maintain_order
            Sort the grouped keys so that the output order is predictable.
        sort_columns
            Sort the transposed columns by name. Default is by order of discovery.
        separator
            Used as separator/delimiter in generated column names in case of multiple value columns.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["one", "one", "two", "two", "one", "two"],
        ...         "bar": ["y", "y", "y", "x", "x", "x"],
        ...         "baz": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df.pivot(index="foo", columns="bar", values="baz", aggregate_function="sum")
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ y   ┆ x   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ one ┆ 3   ┆ 5   │
        │ two ┆ 3   ┆ 10  │
        └─────┴─────┴─────┘

        Pivot using selectors to determine the index/values/columns:

        >>> import polars.selectors as cs
        >>> df.pivot(
        ...     index=cs.string(),
        ...     columns=cs.string(),
        ...     values=cs.numeric(),
        ...     aggregate_function="sum",
        ...     sort_columns=True,
        ... ).sort(
        ...     by=cs.string(),
        ... )
        shape: (4, 6)
        ┌─────┬─────┬─────────────┬─────────────┬─────────────┬─────────────┐
        │ foo ┆ bar ┆ {"one","x"} ┆ {"one","y"} ┆ {"two","x"} ┆ {"two","y"} │
        │ --- ┆ --- ┆ ---         ┆ ---         ┆ ---         ┆ ---         │
        │ str ┆ str ┆ i64         ┆ i64         ┆ i64         ┆ i64         │
        ╞═════╪═════╪═════════════╪═════════════╪═════════════╪═════════════╡
        │ one ┆ x   ┆ 5           ┆ null        ┆ null        ┆ null        │
        │ one ┆ y   ┆ null        ┆ 3           ┆ null        ┆ null        │
        │ two ┆ x   ┆ null        ┆ null        ┆ 10          ┆ null        │
        │ two ┆ y   ┆ null        ┆ null        ┆ null        ┆ 3           │
        └─────┴─────┴─────────────┴─────────────┴─────────────┴─────────────┘

        Run an expression as aggregation function

        >>> df = pl.DataFrame(
        ...     {
        ...         "col1": ["a", "a", "a", "b", "b", "b"],
        ...         "col2": ["x", "x", "x", "x", "y", "y"],
        ...         "col3": [6, 7, 3, 2, 5, 7],
        ...     }
        ... )
        >>> df.pivot(
        ...     index="col1",
        ...     columns="col2",
        ...     values="col3",
        ...     aggregate_function=pl.element().tanh().mean(),
        ... )
        shape: (2, 3)
        ┌──────┬──────────┬──────────┐
        │ col1 ┆ x        ┆ y        │
        │ ---  ┆ ---      ┆ ---      │
        │ str  ┆ f64      ┆ f64      │
        ╞══════╪══════════╪══════════╡
        │ a    ┆ 0.998347 ┆ null     │
        │ b    ┆ 0.964028 ┆ 0.999954 │
        └──────┴──────────┴──────────┘

        Note that `pivot` is only available in eager mode. If you know the unique
        column values in advance, you can use :meth:`polars.LazyFrame.groupby` to
        get the same result as above in lazy mode:

        >>> index = pl.col("col1")
        >>> columns = pl.col("col2")
        >>> values = pl.col("col3")
        >>> unique_column_values = ["x", "y"]
        >>> aggregate_function = lambda col: col.tanh().mean()
        >>> df.lazy().group_by(index).agg(
        ...     aggregate_function(values.filter(columns == value)).alias(value)
        ...     for value in unique_column_values
        ... ).collect()  # doctest: +IGNORE_RESULT
        shape: (2, 3)
        ┌──────┬──────────┬──────────┐
        │ col1 ┆ x        ┆ y        │
        │ ---  ┆ ---      ┆ ---      │
        │ str  ┆ f64      ┆ f64      │
        ╞══════╪══════════╪══════════╡
        │ a    ┆ 0.998347 ┆ null     │
        │ b    ┆ 0.964028 ┆ 0.999954 │
        └──────┴──────────┴──────────┘

        Using a custom `separator` in generated column names:

        >>> df = pl.DataFrame(
        ...     {
        ...         "ix": [1, 1, 2, 2, 1, 2],
        ...         "col": ["a", "a", "a", "a", "b", "b"],
        ...         "foo": [0, 1, 2, 2, 7, 1],
        ...         "bar": [0, 2, 0, 0, 9, 4],
        ...     }
        ... )
        >>> df.pivot(
        ...     index="ix",
        ...     columns="col",
        ...     values=["foo", "bar"],
        ...     aggregate_function="sum",
        ...     separator="/",
        ... )
        shape: (2, 5)
        ┌─────┬───────────┬───────────┬───────────┬───────────┐
        │ ix  ┆ foo/col/a ┆ foo/col/b ┆ bar/col/a ┆ bar/col/b │
        │ --- ┆ ---       ┆ ---       ┆ ---       ┆ ---       │
        │ i64 ┆ i64       ┆ i64       ┆ i64       ┆ i64       │
        ╞═════╪═══════════╪═══════════╪═══════════╪═══════════╡
        │ 1   ┆ 1         ┆ 7         ┆ 2         ┆ 9         │
        │ 2   ┆ 4         ┆ 1         ┆ 0         ┆ 4         │
        └─────┴───────────┴───────────┴───────────┴───────────┘
        """
    def melt(
        self,
        id_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = ...,
        value_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = ...,
        variable_name: str | None = ...,
        value_name: str | None = ...,
    ) -> Self:
        """
        Unpivot a DataFrame from wide to long format.

        Optionally leaves identifiers set.

        This function is useful to massage a DataFrame into a format where one or more
        columns are identifier variables (id_vars) while all other columns, considered
        measured variables (value_vars), are "unpivoted" to the row axis leaving just
        two non-identifier columns, \'variable\' and \'value\'.

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

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["x", "y", "z"],
        ...         "b": [1, 3, 5],
        ...         "c": [2, 4, 6],
        ...     }
        ... )
        >>> import polars.selectors as cs
        >>> df.melt(id_vars="a", value_vars=cs.numeric())
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
    def unstack(
        self,
        step: int,
        how: UnstackDirection = ...,
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = ...,
        fill_values: list[Any] | None = ...,
    ) -> DataFrame:
        """
        Unstack a long table to a wide form without doing an aggregation.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        This can be much faster than a pivot, because it can skip the grouping phase.

        Parameters
        ----------
        step
            Number of rows in the unstacked frame.
        how : { \'vertical\', \'horizontal\' }
            Direction of the unstack.
        columns
            Column name(s) or selector(s) to include in the operation.
            If set to `None` (default), use all columns.
        fill_values
            Fill values that don\'t fit the new size with this value.

        Examples
        --------
        >>> from string import ascii_uppercase
        >>> df = pl.DataFrame(
        ...     {
        ...         "x": list(ascii_uppercase[0:8]),
        ...         "y": pl.int_range(1, 9, eager=True),
        ...     }
        ... ).with_columns(
        ...     z=pl.int_ranges(pl.col("y"), pl.col("y") + 2, dtype=pl.UInt8),
        ... )
        >>> df
        shape: (8, 3)
        ┌─────┬─────┬──────────┐
        │ x   ┆ y   ┆ z        │
        │ --- ┆ --- ┆ ---      │
        │ str ┆ i64 ┆ list[u8] │
        ╞═════╪═════╪══════════╡
        │ A   ┆ 1   ┆ [1, 2]   │
        │ B   ┆ 2   ┆ [2, 3]   │
        │ C   ┆ 3   ┆ [3, 4]   │
        │ D   ┆ 4   ┆ [4, 5]   │
        │ E   ┆ 5   ┆ [5, 6]   │
        │ F   ┆ 6   ┆ [6, 7]   │
        │ G   ┆ 7   ┆ [7, 8]   │
        │ H   ┆ 8   ┆ [8, 9]   │
        └─────┴─────┴──────────┘
        >>> df.unstack(step=4, how="vertical")
        shape: (4, 6)
        ┌─────┬─────┬─────┬─────┬──────────┬──────────┐
        │ x_0 ┆ x_1 ┆ y_0 ┆ y_1 ┆ z_0      ┆ z_1      │
        │ --- ┆ --- ┆ --- ┆ --- ┆ ---      ┆ ---      │
        │ str ┆ str ┆ i64 ┆ i64 ┆ list[u8] ┆ list[u8] │
        ╞═════╪═════╪═════╪═════╪══════════╪══════════╡
        │ A   ┆ E   ┆ 1   ┆ 5   ┆ [1, 2]   ┆ [5, 6]   │
        │ B   ┆ F   ┆ 2   ┆ 6   ┆ [2, 3]   ┆ [6, 7]   │
        │ C   ┆ G   ┆ 3   ┆ 7   ┆ [3, 4]   ┆ [7, 8]   │
        │ D   ┆ H   ┆ 4   ┆ 8   ┆ [4, 5]   ┆ [8, 9]   │
        └─────┴─────┴─────┴─────┴──────────┴──────────┘
        >>> df.unstack(step=2, how="horizontal")
        shape: (4, 6)
        ┌─────┬─────┬─────┬─────┬──────────┬──────────┐
        │ x_0 ┆ x_1 ┆ y_0 ┆ y_1 ┆ z_0      ┆ z_1      │
        │ --- ┆ --- ┆ --- ┆ --- ┆ ---      ┆ ---      │
        │ str ┆ str ┆ i64 ┆ i64 ┆ list[u8] ┆ list[u8] │
        ╞═════╪═════╪═════╪═════╪══════════╪══════════╡
        │ A   ┆ B   ┆ 1   ┆ 2   ┆ [1, 2]   ┆ [2, 3]   │
        │ C   ┆ D   ┆ 3   ┆ 4   ┆ [3, 4]   ┆ [4, 5]   │
        │ E   ┆ F   ┆ 5   ┆ 6   ┆ [5, 6]   ┆ [6, 7]   │
        │ G   ┆ H   ┆ 7   ┆ 8   ┆ [7, 8]   ┆ [8, 9]   │
        └─────┴─────┴─────┴─────┴──────────┴──────────┘
        >>> import polars.selectors as cs
        >>> df.unstack(step=5, columns=cs.numeric(), fill_values=0)
        shape: (5, 2)
        ┌─────┬─────┐
        │ y_0 ┆ y_1 │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 6   │
        │ 2   ┆ 7   │
        │ 3   ┆ 8   │
        │ 4   ┆ 0   │
        │ 5   ┆ 0   │
        └─────┴─────┘
        """
    def partition_by(
        self,
        by: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
        *more_by: ColumnNameOrSelector,
    ) -> list[Self] | dict[Any, Self]:
        """
        Group by the given columns and return the groups as separate dataframes.

        Parameters
        ----------
        by
            Column name(s) or selector(s) to group by.
        *more_by
            Additional names of columns to group by, specified as positional arguments.
        maintain_order
            Ensure that the order of the groups is consistent with the input data.
            This is slower than a default partition by operation.
        include_key
            Include the columns used to partition the DataFrame in the output.
        as_dict
            Return a dictionary instead of a list. The dictionary keys are tuples of
            the distinct group values that identify each group. If a single string
            was passed to `by`, the keys are a single value instead of a tuple.

        Examples
        --------
        Pass a single column name to partition by that column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "b", "a", "b", "c"],
        ...         "b": [1, 2, 1, 3, 3],
        ...         "c": [5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> df.partition_by("a")  # doctest: +IGNORE_RESULT
        [shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 5   │
        │ a   ┆ 1   ┆ 3   │
        └─────┴─────┴─────┘,
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 2   ┆ 4   │
        │ b   ┆ 3   ┆ 2   │
        └─────┴─────┴─────┘,
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ c   ┆ 3   ┆ 1   │
        └─────┴─────┴─────┘]

        Partition by multiple columns by either passing a list of column names, or by
        specifying each column name as a positional argument.

        >>> df.partition_by("a", "b")  # doctest: +IGNORE_RESULT
        [shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 5   │
        │ a   ┆ 1   ┆ 3   │
        └─────┴─────┴─────┘,
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 2   ┆ 4   │
        └─────┴─────┴─────┘,
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 3   ┆ 2   │
        └─────┴─────┴─────┘,
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ c   ┆ 3   ┆ 1   │
        └─────┴─────┴─────┘]

        Return the partitions as a dictionary by specifying `as_dict=True`.

        >>> import polars.selectors as cs
        >>> df.partition_by(cs.string(), as_dict=True)  # doctest: +IGNORE_RESULT
        {(\'a\',): shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ a   ┆ 1   ┆ 5   │
        │ a   ┆ 1   ┆ 3   │
        └─────┴─────┴─────┘,
        (\'b\',): shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ b   ┆ 2   ┆ 4   │
        │ b   ┆ 3   ┆ 2   │
        └─────┴─────┴─────┘,
        (\'c\',): shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ c   ┆ 3   ┆ 1   │
        └─────┴─────┴─────┘}
        """
    def shift(self, n: int = ...) -> DataFrame:
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

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [5, 6, 7, 8],
        ...     }
        ... )
        >>> df.shift()
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

        >>> df.shift(-2)
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

        >>> df.shift(-2, fill_value=100)
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
    def is_duplicated(self) -> Series:
        """
        Get a mask of all duplicated rows in this DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 1],
        ...         "b": ["x", "y", "z", "x"],
        ...     }
        ... )
        >>> df.is_duplicated()
        shape: (4,)
        Series: \'\' [bool]
        [
                true
                false
                false
                true
        ]

        This mask can be used to visualize the duplicated lines like this:

        >>> df.filter(df.is_duplicated())
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 1   ┆ x   │
        │ 1   ┆ x   │
        └─────┴─────┘
        """
    def is_unique(self) -> Series:
        """
        Get a mask of all unique rows in this DataFrame.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 1],
        ...         "b": ["x", "y", "z", "x"],
        ...     }
        ... )
        >>> df.is_unique()
        shape: (4,)
        Series: \'\' [bool]
        [
                false
                true
                true
                false
        ]

        This mask can be used to visualize the unique lines like this:

        >>> df.filter(df.is_unique())
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 2   ┆ y   │
        │ 3   ┆ z   │
        └─────┴─────┘
        """
    def lazy(self) -> LazyFrame:
        """
        Start a lazy query from this point. This returns a `LazyFrame` object.

        Operations on a `LazyFrame` are not executed until this is requested by either
        calling:

        * :meth:`.fetch() <polars.LazyFrame.fetch>`
            (run on a small number of rows)
        * :meth:`.collect() <polars.LazyFrame.collect>`
            (run on all data)
        * :meth:`.describe_plan() <polars.LazyFrame.describe_plan>`
            (print unoptimized query plan)
        * :meth:`.describe_optimized_plan() <polars.LazyFrame.describe_optimized_plan>`
            (print optimized query plan)
        * :meth:`.show_graph() <polars.LazyFrame.show_graph>`
            (show (un)optimized query plan as graphviz graph)

        Lazy operations are advised because they allow for query optimization and more
        parallelization.

        Returns
        -------
        LazyFrame

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [None, 2, 3, 4],
        ...         "b": [0.5, None, 2.5, 13],
        ...         "c": [True, True, False, None],
        ...     }
        ... )
        >>> df.lazy()  # doctest: +ELLIPSIS
        <LazyFrame at ...>
        """
    def select(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr) -> DataFrame:
        """
        Select columns from this DataFrame.

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

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.select("foo")
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

        >>> df.select(["foo", "bar"])
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

        >>> df.select(pl.col("foo"), pl.col("bar") + 1)
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

        >>> df.select(threshold=pl.when(pl.col("foo") > 2).then(10).otherwise(0))
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
        ...     df.select(
        ...         is_odd=(pl.col(pl.INTEGER_DTYPES) % 2).name.suffix("_is_odd"),
        ...     )
        shape: (3, 1)
        ┌───────────┐
        │ is_odd    │
        │ ---       │
        │ struct[2] │
        ╞═══════════╡
        │ {1,0}     │
        │ {0,1}     │
        │ {1,0}     │
        └───────────┘
        """
    def select_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrame:
        """
        Select columns from this DataFrame.

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
    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrame:
        """
        Add columns to this DataFrame.

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
        DataFrame
            A new DataFrame with the columns added.

        Notes
        -----
        Creating a new DataFrame using this method does not create a new copy of
        existing data.

        Examples
        --------
        Pass an expression to add it as a new column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> df.with_columns((pl.col("a") ** 2).alias("a^2"))
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

        >>> df.with_columns(pl.col("a").cast(pl.Float64))
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

        Multiple columns can be added by passing a list of expressions.

        >>> df.with_columns(
        ...     [
        ...         (pl.col("a") ** 2).alias("a^2"),
        ...         (pl.col("b") / 2).alias("b/2"),
        ...         (pl.col("c").not_()).alias("not c"),
        ...     ]
        ... )
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

        Multiple columns also can be added using positional arguments instead of a list.

        >>> df.with_columns(
        ...     (pl.col("a") ** 2).alias("a^2"),
        ...     (pl.col("b") / 2).alias("b/2"),
        ...     (pl.col("c").not_()).alias("not c"),
        ... )
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

        >>> df.with_columns(
        ...     ab=pl.col("a") * pl.col("b"),
        ...     not_c=pl.col("c").not_(),
        ... )
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

        Expressions with multiple outputs can be automatically instantiated as Structs
        by enabling the setting `Config.set_auto_structify(True)`:

        >>> with pl.Config(auto_structify=True):
        ...     df.drop("c").with_columns(
        ...         diffs=pl.col(["a", "b"]).diff().name.suffix("_diff"),
        ...     )
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
    def with_columns_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> DataFrame:
        """
        Add columns to this DataFrame.

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
        DataFrame
            A new DataFrame with the columns added.

        See Also
        --------
        with_columns
        """
    def n_chunks(self, strategy: str = ...) -> int | list[int]:
        """
        Get number of chunks used by the ChunkedArrays of this DataFrame.

        Parameters
        ----------
        strategy : {\'first\', \'all\'}
            Return the number of chunks of the \'first\' column,
            or \'all\' columns in this DataFrame.


        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [0.5, 4, 10, 13],
        ...         "c": [True, True, False, True],
        ...     }
        ... )
        >>> df.n_chunks()
        1
        >>> df.n_chunks(strategy="all")
        [1, 1, 1]
        """
    def max(self, axis: int | None = ...) -> Self | Series:
        """
        Aggregate the columns of this DataFrame to their maximum value.

        Parameters
        ----------
        axis
            Either 0 (vertical) or 1 (horizontal).

            .. deprecated:: 0.19.14
                This argument will be removed in a future version. This method will only
                support vertical aggregation, as if `axis` were set to `0`.
                To perform horizontal aggregation, use :meth:`max_horizontal`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.max()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8   ┆ c   │
        └─────┴─────┴─────┘
        """
    def max_horizontal(self) -> Series:
        """
        Get the maximum value horizontally across columns.

        Returns
        -------
        Series
            A Series named `"max"`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [4.0, 5.0, 6.0],
        ...     }
        ... )
        >>> df.max_horizontal()
        shape: (3,)
        Series: \'max\' [f64]
        [
                4.0
                5.0
                6.0
        ]
        """
    def min(self, axis: int | None = ...) -> Self | Series:
        """
        Aggregate the columns of this DataFrame to their minimum value.

        Parameters
        ----------
        axis
            Either 0 (vertical) or 1 (horizontal).

            .. deprecated:: 0.19.14
                This argument will be removed in a future version. This method will only
                support vertical aggregation, as if `axis` were set to `0`.
                To perform horizontal aggregation, use :meth:`min_horizontal`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.min()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 6   ┆ a   │
        └─────┴─────┴─────┘
        """
    def min_horizontal(self) -> Series:
        """
        Get the minimum value horizontally across columns.

        Returns
        -------
        Series
            A Series named `"min"`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [4.0, 5.0, 6.0],
        ...     }
        ... )
        >>> df.min_horizontal()
        shape: (3,)
        Series: \'min\' [f64]
        [
                1.0
                2.0
                3.0
        ]
        """
    def sum(self) -> Self | Series:
        """
        Aggregate the columns of this DataFrame to their sum value.

        Parameters
        ----------
        axis
            Either 0 (vertical) or 1 (horizontal).

            .. deprecated:: 0.19.14
                This argument will be removed in a future version. This method will only
                support vertical aggregation, as if `axis` were set to `0`.
                To perform horizontal aggregation, use :meth:`sum_horizontal`.
        null_strategy : {\'ignore\', \'propagate\'}
            This argument is only used if `axis == 1`.

            .. deprecated:: 0.19.14
                This argument will be removed in a future version.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.sum()
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ i64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 6   ┆ 21  ┆ null │
        └─────┴─────┴──────┘
        """
    def sum_horizontal(self) -> Series:
        """
        Sum all values horizontally across columns.

        Parameters
        ----------
        ignore_nulls
            Ignore null values (default).
            If set to `False`, any null value in the input will lead to a null output.

        Returns
        -------
        Series
            A Series named `"sum"`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [4.0, 5.0, 6.0],
        ...     }
        ... )
        >>> df.sum_horizontal()
        shape: (3,)
        Series: \'sum\' [f64]
        [
                5.0
                7.0
                9.0
        ]
        """
    def mean(self) -> Self | Series:
        """
        Aggregate the columns of this DataFrame to their mean value.

        Parameters
        ----------
        axis
            Either 0 (vertical) or 1 (horizontal).

            .. deprecated:: 0.19.14
                This argument will be removed in a future version. This method will only
                support vertical aggregation, as if `axis` were set to `0`.
                To perform horizontal aggregation, use :meth:`mean_horizontal`.
        null_strategy : {\'ignore\', \'propagate\'}
            This argument is only used if `axis == 1`.

            .. deprecated:: 0.19.14
                This argument will be removed in a future version.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...         "spam": [True, False, None],
        ...     }
        ... )
        >>> df.mean()
        shape: (1, 4)
        ┌─────┬─────┬──────┬──────┐
        │ foo ┆ bar ┆ ham  ┆ spam │
        │ --- ┆ --- ┆ ---  ┆ ---  │
        │ f64 ┆ f64 ┆ str  ┆ f64  │
        ╞═════╪═════╪══════╪══════╡
        │ 2.0 ┆ 7.0 ┆ null ┆ 0.5  │
        └─────┴─────┴──────┴──────┘
        """
    def mean_horizontal(self) -> Series:
        """
        Take the mean of all values horizontally across columns.

        Parameters
        ----------
        ignore_nulls
            Ignore null values (default).
            If set to `False`, any null value in the input will lead to a null output.

        Returns
        -------
        Series
            A Series named `"mean"`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [4.0, 5.0, 6.0],
        ...     }
        ... )
        >>> df.mean_horizontal()
        shape: (3,)
        Series: \'mean\' [f64]
        [
                2.5
                3.5
                4.5
        ]
        """
    def std(self, ddof: int = ...) -> Self:
        """
        Aggregate the columns of this DataFrame to their standard deviation value.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.std()
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ f64 ┆ f64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 1.0 ┆ 1.0 ┆ null │
        └─────┴─────┴──────┘
        >>> df.std(ddof=0)
        shape: (1, 3)
        ┌──────────┬──────────┬──────┐
        │ foo      ┆ bar      ┆ ham  │
        │ ---      ┆ ---      ┆ ---  │
        │ f64      ┆ f64      ┆ str  │
        ╞══════════╪══════════╪══════╡
        │ 0.816497 ┆ 0.816497 ┆ null │
        └──────────┴──────────┴──────┘
        """
    def var(self, ddof: int = ...) -> Self:
        """
        Aggregate the columns of this DataFrame to their variance value.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.var()
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ f64 ┆ f64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 1.0 ┆ 1.0 ┆ null │
        └─────┴─────┴──────┘
        >>> df.var(ddof=0)
        shape: (1, 3)
        ┌──────────┬──────────┬──────┐
        │ foo      ┆ bar      ┆ ham  │
        │ ---      ┆ ---      ┆ ---  │
        │ f64      ┆ f64      ┆ str  │
        ╞══════════╪══════════╪══════╡
        │ 0.666667 ┆ 0.666667 ┆ null │
        └──────────┴──────────┴──────┘
        """
    def median(self) -> Self:
        """
        Aggregate the columns of this DataFrame to their median value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.median()
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ f64 ┆ f64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 2.0 ┆ 7.0 ┆ null │
        └─────┴─────┴──────┘
        """
    def product(self) -> DataFrame:
        """
        Aggregate the columns of this DataFrame to their product values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": [0.5, 4, 10],
        ...         "c": [True, True, False],
        ...     }
        ... )

        >>> df.product()
        shape: (1, 3)
        ┌─────┬──────┬─────┐
        │ a   ┆ b    ┆ c   │
        │ --- ┆ ---  ┆ --- │
        │ i64 ┆ f64  ┆ i64 │
        ╞═════╪══════╪═════╡
        │ 6   ┆ 20.0 ┆ 0   │
        └─────┴──────┴─────┘
        """
    def quantile(self, quantile: float, interpolation: RollingInterpolationMethod = ...) -> Self:
        """
        Aggregate the columns of this DataFrame to their quantile value.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {\'nearest\', \'higher\', \'lower\', \'midpoint\', \'linear\'}
            Interpolation method.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.quantile(0.5, "nearest")
        shape: (1, 3)
        ┌─────┬─────┬──────┐
        │ foo ┆ bar ┆ ham  │
        │ --- ┆ --- ┆ ---  │
        │ f64 ┆ f64 ┆ str  │
        ╞═════╪═════╪══════╡
        │ 2.0 ┆ 7.0 ┆ null │
        └─────┴─────┴──────┘
        """
    def to_dummies(
        self, columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = ...
    ) -> Self:
        """
        Convert categorical variables into dummy/indicator variables.

        Parameters
        ----------
        columns
            Column name(s) or selector(s) that should be converted to dummy
            variables. If set to `None` (default), convert all columns.
        separator
            Separator/delimiter used when generating column names.
        drop_first
            Remove the first category from the variables being encoded.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2],
        ...         "bar": [3, 4],
        ...         "ham": ["a", "b"],
        ...     }
        ... )
        >>> df.to_dummies()
        shape: (2, 6)
        ┌───────┬───────┬───────┬───────┬───────┬───────┐
        │ foo_1 ┆ foo_2 ┆ bar_3 ┆ bar_4 ┆ ham_a ┆ ham_b │
        │ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   │
        │ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    │
        ╞═══════╪═══════╪═══════╪═══════╪═══════╪═══════╡
        │ 1     ┆ 0     ┆ 1     ┆ 0     ┆ 1     ┆ 0     │
        │ 0     ┆ 1     ┆ 0     ┆ 1     ┆ 0     ┆ 1     │
        └───────┴───────┴───────┴───────┴───────┴───────┘

        >>> df.to_dummies(drop_first=True)
        shape: (2, 3)
        ┌───────┬───────┬───────┐
        │ foo_2 ┆ bar_4 ┆ ham_b │
        │ ---   ┆ ---   ┆ ---   │
        │ u8    ┆ u8    ┆ u8    │
        ╞═══════╪═══════╪═══════╡
        │ 0     ┆ 0     ┆ 0     │
        │ 1     ┆ 1     ┆ 1     │
        └───────┴───────┴───────┘

        >>> import polars.selectors as cs
        >>> df.to_dummies(cs.integer(), separator=":")
        shape: (2, 5)
        ┌───────┬───────┬───────┬───────┬─────┐
        │ foo:1 ┆ foo:2 ┆ bar:3 ┆ bar:4 ┆ ham │
        │ ---   ┆ ---   ┆ ---   ┆ ---   ┆ --- │
        │ u8    ┆ u8    ┆ u8    ┆ u8    ┆ str │
        ╞═══════╪═══════╪═══════╪═══════╪═════╡
        │ 1     ┆ 0     ┆ 1     ┆ 0     ┆ a   │
        │ 0     ┆ 1     ┆ 0     ┆ 1     ┆ b   │
        └───────┴───────┴───────┴───────┴─────┘

        >>> df.to_dummies(cs.integer(), drop_first=True, separator=":")
        shape: (2, 3)
        ┌───────┬───────┬─────┐
        │ foo:2 ┆ bar:4 ┆ ham │
        │ ---   ┆ ---   ┆ --- │
        │ u8    ┆ u8    ┆ str │
        ╞═══════╪═══════╪═════╡
        │ 0     ┆ 0     ┆ a   │
        │ 1     ┆ 1     ┆ b   │
        └───────┴───────┴─────┘
        """
    def unique(
        self, subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = ...
    ) -> DataFrame:
        """
        Drop duplicate rows from this dataframe.

        Parameters
        ----------
        subset
            Column name(s) or selector(s), to consider when identifying
            duplicate rows. If set to `None` (default), use all columns.
        keep : {\'first\', \'last\', \'any\', \'none\'}
            Which of the duplicate rows to keep.

            * \'any\': Does not give any guarantee of which row is kept.
                     This allows more optimizations.
            * \'none\': Don\'t keep duplicate rows.
            * \'first\': Keep first unique row.
            * \'last\': Keep last unique row.
        maintain_order
            Keep the same order as the original DataFrame. This is more expensive to
            compute.
            Settings this to `True` blocks the possibility
            to run on the streaming engine.

        Returns
        -------
        DataFrame
            DataFrame with unique rows.

        Warnings
        --------
        This method will fail if there is a column of type `List` in the DataFrame or
        subset.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3, 1],
        ...         "bar": ["a", "a", "a", "a"],
        ...         "ham": ["b", "b", "b", "b"],
        ...     }
        ... )
        >>> df.unique(maintain_order=True)
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
        >>> df.unique(subset=["bar", "ham"], maintain_order=True)
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 1   ┆ a   ┆ b   │
        └─────┴─────┴─────┘
        >>> df.unique(keep="last", maintain_order=True)
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
    def n_unique(self, subset: str | Expr | Sequence[str | Expr] | None = ...) -> int:
        """
        Return the number of unique rows, or the number of unique row-subsets.

        Parameters
        ----------
        subset
            One or more columns/expressions that define what to count;
            omit to return the count of unique rows.

        Notes
        -----
        This method operates at the `DataFrame` level; to operate on subsets at the
        expression level you can make use of struct-packing instead, for example:

        >>> expr_unique_subset = pl.struct("a", "b").n_unique()

        If instead you want to count the number of unique values per-column, you can
        also use expression-level syntax to return a new frame containing that result:

        >>> df = pl.DataFrame([[1, 2, 3], [1, 2, 4]], schema=["a", "b", "c"])
        >>> df_nunique = df.select(pl.all().n_unique())

        In aggregate context there is also an equivalent method for returning the
        unique values per-group:

        >>> df_agg_nunique = df.group_by("a").n_unique()

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 1, 2, 3, 4, 5],
        ...         "b": [0.5, 0.5, 1.0, 2.0, 3.0, 3.0],
        ...         "c": [True, True, True, False, True, True],
        ...     }
        ... )
        >>> df.n_unique()
        5

        Simple columns subset.

        >>> df.n_unique(subset=["b", "c"])
        4

        Expression subset.

        >>> df.n_unique(
        ...     subset=[
        ...         (pl.col("a") // 2),
        ...         (pl.col("c") | (pl.col("b") >= 2)),
        ...     ],
        ... )
        3
        """
    def approx_n_unique(self) -> DataFrame:
        """
        Approximate count of unique values.

        .. deprecated:: 0.20.11
            Use `select(pl.all().approx_n_unique())` instead.

        This is done using the HyperLogLog++ algorithm for cardinality estimation.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4],
        ...         "b": [1, 2, 1, 1],
        ...     }
        ... )
        >>> df.approx_n_unique()  # doctest: +SKIP
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u32 ┆ u32 │
        ╞═════╪═════╡
        │ 4   ┆ 2   │
        └─────┴─────┘
        """
    def rechunk(self) -> Self:
        """
        Rechunk the data in this DataFrame to a contiguous allocation.

        This will make sure all subsequent operations have optimal and predictable
        performance.
        """
    def null_count(self) -> Self:
        """
        Create a new DataFrame that shows the null counts per column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, None, 3],
        ...         "bar": [6, 7, None],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.null_count()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ u32 ┆ u32 ┆ u32 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 1   ┆ 0   │
        └─────┴─────┴─────┘
        """
    def sample(self, n: int | Series | None = ...) -> Self:
        """
        Sample from this DataFrame.

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
            If set to True, the order of the sampled rows will be shuffled. If
            set to False (default), the order of the returned rows will be
            neither stable nor fully random.
        seed
            Seed for the random number generator. If set to None (default), a
            random seed is generated for each sample operation.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.sample(n=2, seed=0)  # doctest: +IGNORE_RESULT
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ foo ┆ bar ┆ ham │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ str │
        ╞═════╪═════╪═════╡
        │ 3   ┆ 8   ┆ c   │
        │ 2   ┆ 7   ┆ b   │
        └─────┴─────┴─────┘
        """
    def fold(self, operation: Callable[[Series, Series], Series]) -> Series:
        """
        Apply a horizontal reduction on a DataFrame.

        This can be used to effectively determine aggregations on a row level, and can
        be applied to any DataType that can be supercasted (casted to a similar parent
        type).

        An example of the supercast rules when applying an arithmetic operation on two
        DataTypes are for instance:

        - Int8 + String = String
        - Float32 + Int64 = Float32
        - Float32 + Float64 = Float64

        Examples
        --------
        A horizontal sum operation:

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [2, 1, 3],
        ...         "b": [1, 2, 3],
        ...         "c": [1.0, 2.0, 3.0],
        ...     }
        ... )
        >>> df.fold(lambda s1, s2: s1 + s2)
        shape: (3,)
        Series: \'a\' [f64]
        [
            4.0
            5.0
            9.0
        ]

        A horizontal minimum operation:

        >>> df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
        >>> df.fold(lambda s1, s2: s1.zip_with(s1 < s2, s2))
        shape: (3,)
        Series: \'a\' [f64]
        [
            1.0
            1.0
            3.0
        ]

        A horizontal string concatenation:

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["foo", "bar", 2],
        ...         "b": [1, 2, 3],
        ...         "c": [1.0, 2.0, 3.0],
        ...     }
        ... )
        >>> df.fold(lambda s1, s2: s1 + s2)
        shape: (3,)
        Series: \'a\' [str]
        [
            "foo11.0"
            "bar22.0"
            null
        ]

        A horizontal boolean or, similar to a row-wise .any():

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [False, False, True],
        ...         "b": [False, True, False],
        ...     }
        ... )
        >>> df.fold(lambda s1, s2: s1 | s2)
        shape: (3,)
        Series: \'a\' [bool]
        [
                false
                true
                true
        ]

        Parameters
        ----------
        operation
            function that takes two `Series` and returns a `Series`.
        """
    def row(self, index: int | None = ...) -> tuple[Any, ...] | dict[str, Any]:
        """
        Get the values of a single row, either by index or by predicate.

        Parameters
        ----------
        index
            Row index.
        by_predicate
            Select the row according to a given expression/predicate.
        named
            Return a dictionary instead of a tuple. The dictionary is a mapping of
            column name to row value. This is more expensive than returning a regular
            tuple, but allows for accessing values by column name.

        Returns
        -------
        tuple (default) or dictionary of row values

        Notes
        -----
        The `index` and `by_predicate` params are mutually exclusive. Additionally,
        to ensure clarity, the `by_predicate` parameter must be supplied by keyword.

        When using `by_predicate` it is an error condition if anything other than
        one row is returned; more than one row raises `TooManyRowsReturnedError`, and
        zero rows will raise `NoRowsReturnedError` (both inherit from `RowsError`).

        Warnings
        --------
        You should NEVER use this method to iterate over a DataFrame; if you require
        row-iteration you should strongly prefer use of `iter_rows()` instead.

        See Also
        --------
        iter_rows : Row iterator over frame data (does not materialise all rows).
        rows : Materialise all frame data as a list of rows (potentially expensive).
        item: Return dataframe element as a scalar.

        Examples
        --------
        Specify an index to return the row at the given index as a tuple.

        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, 2, 3],
        ...         "bar": [6, 7, 8],
        ...         "ham": ["a", "b", "c"],
        ...     }
        ... )
        >>> df.row(2)
        (3, 8, \'c\')

        Specify `named=True` to get a dictionary instead with a mapping of column
        names to row values.

        >>> df.row(2, named=True)
        {\'foo\': 3, \'bar\': 8, \'ham\': \'c\'}

        Use `by_predicate` to return the row that matches the given predicate.

        >>> df.row(by_predicate=(pl.col("ham") == "b"))
        (2, 7, \'b\')
        """
    def rows(self) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """
        Returns all data in the DataFrame as a list of rows of python-native values.

        By default, each row is returned as a tuple of values given in the same order
        as the frame columns. Setting `named=True` will return rows of dictionaries
        instead.

        Parameters
        ----------
        named
            Return dictionaries instead of tuples. The dictionaries are a mapping of
            column name to row value. This is more expensive than returning a regular
            tuple, but allows for accessing values by column name.

        Notes
        -----
        If you have `ns`-precision temporal values you should be aware that Python
        natively only supports up to `μs`-precision; `ns`-precision values will be
        truncated to microseconds on conversion to Python. If this matters to your
        use-case you should export to a different format (such as Arrow or NumPy).

        Warnings
        --------
        Row-iteration is not optimal as the underlying data is stored in columnar form;
        where possible, prefer export via one of the dedicated export/output methods.
        You should also consider using `iter_rows` instead, to avoid materialising all
        the data at once; there is little performance difference between the two, but
        peak memory can be reduced if processing rows in batches.

        Returns
        -------
        list of row value tuples (default), or list of dictionaries (if `named=True`).

        See Also
        --------
        iter_rows : Row iterator over frame data (does not materialise all rows).
        rows_by_key : Materialises frame data as a key-indexed dictionary.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "x": ["a", "b", "b", "a"],
        ...         "y": [1, 2, 3, 4],
        ...         "z": [0, 3, 6, 9],
        ...     }
        ... )
        >>> df.rows()
        [(\'a\', 1, 0), (\'b\', 2, 3), (\'b\', 3, 6), (\'a\', 4, 9)]
        >>> df.rows(named=True)
        [{\'x\': \'a\', \'y\': 1, \'z\': 0},
         {\'x\': \'b\', \'y\': 2, \'z\': 3},
         {\'x\': \'b\', \'y\': 3, \'z\': 6},
         {\'x\': \'a\', \'y\': 4, \'z\': 9}]
        """
    def rows_by_key(
        self, key: ColumnNameOrSelector | Sequence[ColumnNameOrSelector]
    ) -> dict[Any, Iterable[Any]]:
        """
        Returns all data as a dictionary of python-native values keyed by some column.

        This method is like `rows`, but instead of returning rows in a flat list, rows
        are grouped by the values in the `key` column(s) and returned as a dictionary.

        Note that this method should not be used in place of native operations, due to
        the high cost of materializing all frame data out into a dictionary; it should
        be used only when you need to move the values out into a Python data structure
        or other object that cannot operate directly with Polars/Arrow.

        Parameters
        ----------
        key
            The column(s) to use as the key for the returned dictionary. If multiple
            columns are specified, the key will be a tuple of those values, otherwise
            it will be a string.
        named
            Return dictionary rows instead of tuples, mapping column name to row value.
        include_key
            Include key values inline with the associated data (by default the key
            values are omitted as a memory/performance optimisation, as they can be
            reoconstructed from the key).
        unique
            Indicate that the key is unique; this will result in a 1:1 mapping from
            key to a single associated row. Note that if the key is *not* actually
            unique the last row with the given key will be returned.

        Notes
        -----
        If you have `ns`-precision temporal values you should be aware that Python
        natively only supports up to `μs`-precision; `ns`-precision values will be
        truncated to microseconds on conversion to Python. If this matters to your
        use-case you should export to a different format (such as Arrow or NumPy).

        See Also
        --------
        rows : Materialize all frame data as a list of rows (potentially expensive).
        iter_rows : Row iterator over frame data (does not materialize all rows).

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "w": ["a", "b", "b", "a"],
        ...         "x": ["q", "q", "q", "k"],
        ...         "y": [1.0, 2.5, 3.0, 4.5],
        ...         "z": [9, 8, 7, 6],
        ...     }
        ... )

        Group rows by the given key column(s):

        >>> df.rows_by_key(key=["w"])
        defaultdict(<class \'list\'>,
            {\'a\': [(\'q\', 1.0, 9), (\'k\', 4.5, 6)],
             \'b\': [(\'q\', 2.5, 8), (\'q\', 3.0, 7)]})

        Return the same row groupings as dictionaries:

        >>> df.rows_by_key(key=["w"], named=True)
        defaultdict(<class \'list\'>,
            {\'a\': [{\'x\': \'q\', \'y\': 1.0, \'z\': 9},
                   {\'x\': \'k\', \'y\': 4.5, \'z\': 6}],
             \'b\': [{\'x\': \'q\', \'y\': 2.5, \'z\': 8},
                   {\'x\': \'q\', \'y\': 3.0, \'z\': 7}]})

        Return row groupings, assuming keys are unique:

        >>> df.rows_by_key(key=["z"], unique=True)
        {9: (\'a\', \'q\', 1.0),
         8: (\'b\', \'q\', 2.5),
         7: (\'b\', \'q\', 3.0),
         6: (\'a\', \'k\', 4.5)}

        Return row groupings as dictionaries, assuming keys are unique:

        >>> df.rows_by_key(key=["z"], named=True, unique=True)
        {9: {\'w\': \'a\', \'x\': \'q\', \'y\': 1.0},
         8: {\'w\': \'b\', \'x\': \'q\', \'y\': 2.5},
         7: {\'w\': \'b\', \'x\': \'q\', \'y\': 3.0},
         6: {\'w\': \'a\', \'x\': \'k\', \'y\': 4.5}}

        Return dictionary rows grouped by a compound key, including key values:

        >>> df.rows_by_key(key=["w", "x"], named=True, include_key=True)
        defaultdict(<class \'list\'>,
            {(\'a\', \'q\'): [{\'w\': \'a\', \'x\': \'q\', \'y\': 1.0, \'z\': 9}],
             (\'b\', \'q\'): [{\'w\': \'b\', \'x\': \'q\', \'y\': 2.5, \'z\': 8},
                          {\'w\': \'b\', \'x\': \'q\', \'y\': 3.0, \'z\': 7}],
             (\'a\', \'k\'): [{\'w\': \'a\', \'x\': \'k\', \'y\': 4.5, \'z\': 6}]})
        """
    def iter_rows(self) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]:
        """
        Returns an iterator over the DataFrame of rows of python-native values.

        Parameters
        ----------
        named
            Return dictionaries instead of tuples. The dictionaries are a mapping of
            column name to row value. This is more expensive than returning a regular
            tuple, but allows for accessing values by column name.
        buffer_size
            Determines the number of rows that are buffered internally while iterating
            over the data; you should only modify this in very specific cases where the
            default value is determined not to be a good fit to your access pattern, as
            the speedup from using the buffer is significant (~2-4x). Setting this
            value to zero disables row buffering (not recommended).

        Notes
        -----
        If you have `ns`-precision temporal values you should be aware that Python
        natively only supports up to `μs`-precision; `ns`-precision values will be
        truncated to microseconds on conversion to Python. If this matters to your
        use-case you should export to a different format (such as Arrow or NumPy).

        Warnings
        --------
        Row iteration is not optimal as the underlying data is stored in columnar form;
        where possible, prefer export via one of the dedicated export/output methods
        that deals with columnar data.

        Returns
        -------
        iterator of tuples (default) or dictionaries (if named) of python row values

        See Also
        --------
        rows : Materialises all frame data as a list of rows (potentially expensive).
        rows_by_key : Materialises frame data as a key-indexed dictionary.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> [row[0] for row in df.iter_rows()]
        [1, 3, 5]
        >>> [row["b"] for row in df.iter_rows(named=True)]
        [2, 4, 6]
        """
    def iter_columns(self) -> Iterator[Series]:
        """
        Returns an iterator over the columns of this DataFrame.

        Yields
        ------
        Series

        Notes
        -----
        Consider whether you can use :func:`all` instead.
        If you can, it will be more efficient.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> [s.name for s in df.iter_columns()]
        [\'a\', \'b\']

        If you\'re using this to modify a dataframe\'s columns, e.g.

        >>> # Do NOT do this
        >>> pl.DataFrame(column * 2 for column in df.iter_columns())
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 4   │
        │ 6   ┆ 8   │
        │ 10  ┆ 12  │
        └─────┴─────┘

        then consider whether you can use :func:`all` instead:

        >>> df.select(pl.all() * 2)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 4   │
        │ 6   ┆ 8   │
        │ 10  ┆ 12  │
        └─────┴─────┘
        """
    def iter_slices(self, n_rows: int = ...) -> Iterator[DataFrame]:
        """
        Returns a non-copying iterator of slices over the underlying DataFrame.

        Parameters
        ----------
        n_rows
            Determines the number of rows contained in each DataFrame slice.

        Examples
        --------
        >>> from datetime import date
        >>> df = pl.DataFrame(
        ...     data={
        ...         "a": range(17_500),
        ...         "b": date(2023, 1, 1),
        ...         "c": "klmnoopqrstuvwxyz",
        ...     },
        ...     schema_overrides={"a": pl.Int32},
        ... )
        >>> for idx, frame in enumerate(df.iter_slices()):
        ...     print(f"{type(frame).__name__}:[{idx}]:{len(frame)}")
        DataFrame:[0]:10000
        DataFrame:[1]:7500

        Using `iter_slices` is an efficient way to chunk-iterate over DataFrames and
        any supported frame export/conversion types; for example, as RecordBatches:

        >>> for frame in df.iter_slices(n_rows=15_000):
        ...     record_batch = frame.to_arrow().to_batches()[0]
        ...     print(f"{record_batch.schema}\\n<< {len(record_batch)}")
        a: int32
        b: date32[day]
        c: large_string
        << 15000
        a: int32
        b: date32[day]
        c: large_string
        << 2500

        See Also
        --------
        iter_rows : Row iterator over frame data (does not materialise all rows).
        partition_by : Split into multiple DataFrames, partitioned by groups.
        """
    def shrink_to_fit(self) -> Self:
        """
        Shrink DataFrame memory usage.

        Shrinks to fit the exact capacity needed to hold the data.
        """
    def gather_every(self, n: int, offset: int = ...) -> DataFrame:
        """
        Take every nth row in the DataFrame and return as a new DataFrame.

        Parameters
        ----------
        n
            Gather every *n*-th row.
        offset
            Starting index.

        Examples
        --------
        >>> s = pl.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        >>> s.gather_every(2)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 5   │
        │ 3   ┆ 7   │
        └─────┴─────┘

        >>> s.gather_every(2, offset=1)
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
    def hash_rows(
        self,
        seed: int = ...,
        seed_1: int | None = ...,
        seed_2: int | None = ...,
        seed_3: int | None = ...,
    ) -> Series:
        """
        Hash and combine the rows in this DataFrame.

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
        This implementation of `hash_rows` does not guarantee stable results
        across different Polars versions. Its stability is only guaranteed within a
        single version.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, None, 3, 4],
        ...         "ham": ["a", "b", None, "d"],
        ...     }
        ... )
        >>> df.hash_rows(seed=42)  # doctest: +IGNORE_RESULT
        shape: (4,)
        Series: \'\' [u64]
        [
            10783150408545073287
            1438741209321515184
            10047419486152048166
            2047317070637311557
        ]
        """
    def interpolate(self) -> DataFrame:
        """
        Interpolate intermediate values. The interpolation method is linear.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1, None, 9, 10],
        ...         "bar": [6, 7, 9, None],
        ...         "baz": [1, None, None, 9],
        ...     }
        ... )
        >>> df.interpolate()
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
    def is_empty(self) -> bool:
        """
        Returns `True` if the DataFrame contains no rows.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> df.is_empty()
        False
        >>> df.filter(pl.col("foo") > 99).is_empty()
        True
        """
    def to_struct(self, name: str = ...) -> Series:
        """
        Convert a `DataFrame` to a `Series` of type `Struct`.

        Parameters
        ----------
        name
            Name for the struct Series

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 4, 5],
        ...         "b": ["one", "two", "three", "four", "five"],
        ...     }
        ... )
        >>> df.to_struct("nums")
        shape: (5,)
        Series: \'nums\' [struct[2]]
        [
            {1,"one"}
            {2,"two"}
            {3,"three"}
            {4,"four"}
            {5,"five"}
        ]
        """
    def unnest(
        self,
        columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> Self:
        """
        Decompose struct columns into separate columns for each of their fields.

        The new columns will be inserted into the dataframe at the location of the
        struct column.

        Parameters
        ----------
        columns
            Name of the struct column(s) that should be unnested.
        *more_columns
            Additional columns to unnest, specified as positional arguments.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "before": ["foo", "bar"],
        ...         "t_a": [1, 2],
        ...         "t_b": ["a", "b"],
        ...         "t_c": [True, None],
        ...         "t_d": [[1, 2], [3]],
        ...         "after": ["baz", "womp"],
        ...     }
        ... ).select("before", pl.struct(pl.col("^t_.$")).alias("t_struct"), "after")
        >>> df
        shape: (2, 3)
        ┌────────┬─────────────────────┬───────┐
        │ before ┆ t_struct            ┆ after │
        │ ---    ┆ ---                 ┆ ---   │
        │ str    ┆ struct[4]           ┆ str   │
        ╞════════╪═════════════════════╪═══════╡
        │ foo    ┆ {1,"a",true,[1, 2]} ┆ baz   │
        │ bar    ┆ {2,"b",null,[3]}    ┆ womp  │
        └────────┴─────────────────────┴───────┘
        >>> df.unnest("t_struct")
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
    def corr(self, **kwargs: Any) -> DataFrame:
        """
        Return pairwise Pearson product-moment correlation coefficients between columns.

        See numpy `corrcoef` for more information:
        https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html

        Notes
        -----
        This functionality requires numpy to be installed.

        Parameters
        ----------
        **kwargs
            Keyword arguments are passed to numpy `corrcoef`.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3], "bar": [3, 2, 1], "ham": [7, 8, 9]})
        >>> df.corr()
        shape: (3, 3)
        ┌──────┬──────┬──────┐
        │ foo  ┆ bar  ┆ ham  │
        │ ---  ┆ ---  ┆ ---  │
        │ f64  ┆ f64  ┆ f64  │
        ╞══════╪══════╪══════╡
        │ 1.0  ┆ -1.0 ┆ 1.0  │
        │ -1.0 ┆ 1.0  ┆ -1.0 │
        │ 1.0  ┆ -1.0 ┆ 1.0  │
        └──────┴──────┴──────┘
        """
    def merge_sorted(self, other: DataFrame, key: str) -> DataFrame:
        """
        Take two sorted DataFrames and merge them by the sorted key.

        The output of this operation will also be sorted.
        It is the callers responsibility that the frames are sorted
        by that key otherwise the output will not make sense.

        The schemas of both DataFrames must be equal.

        Parameters
        ----------
        other
            Other DataFrame that must be merged
        key
            Key that is sorted.

        Examples
        --------
        >>> df0 = pl.DataFrame(
        ...     {"name": ["steve", "elise", "bob"], "age": [42, 44, 18]}
        ... ).sort("age")
        >>> df0
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
        >>> df1 = pl.DataFrame(
        ...     {"name": ["anna", "megan", "steve", "thomas"], "age": [21, 33, 42, 20]}
        ... ).sort("age")
        >>> df1
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
        >>> df0.merge_sorted(df1, key="age")
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
    def set_sorted(self, column: str | Iterable[str], *more_columns: str) -> DataFrame:
        """
        Indicate that one or multiple columns are sorted.

        Parameters
        ----------
        column
            Columns that are sorted
        more_columns
            Additional columns that are sorted, specified as positional arguments.
        descending
            Whether the columns are sorted in descending order.
        """
    def update(
        self,
        other: DataFrame,
        on: str | Sequence[str] | None = ...,
        how: Literal["left", "inner", "full"] = ...,
    ) -> DataFrame:
        """
        Update the values in this `DataFrame` with the values in `other`.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        By default, null values in the right frame are ignored. Use
        `include_nulls=False` to overwrite values in this frame with
        null values in the other frame.

        Parameters
        ----------
        other
            DataFrame that will be used to update the values
        on
            Column names that will be joined on. If set to `None` (default),
            the implicit row index of each frame is used as a join key.
        how : {\'left\', \'inner\', \'full\'}
            * \'left\' will keep all rows from the left table; rows may be duplicated
              if multiple rows in the right frame match the left row\'s key.
            * \'inner\' keeps only those rows where the key exists in both frames.
            * \'full\' will update existing rows where the key matches while also
              adding any new rows contained in the given frame.
        left_on
           Join column(s) of the left DataFrame.
        right_on
           Join column(s) of the right DataFrame.
        include_nulls
            If True, null values from the right dataframe will be used to update the
            left dataframe.

        Notes
        -----
        This is syntactic sugar for a left/inner join, with an optional coalesce
        when `include_nulls = False`

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2, 3, 4],
        ...         "B": [400, 500, 600, 700],
        ...     }
        ... )
        >>> df
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
        >>> new_df = pl.DataFrame(
        ...     {
        ...         "B": [-66, None, -99],
        ...         "C": [5, 3, 1],
        ...     }
        ... )

        Update `df` values with the non-null values in `new_df`, by row index:

        >>> df.update(new_df)
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

        >>> df.update(new_df, how="inner")
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

        >>> df.update(new_df, left_on=["A"], right_on=["C"], how="full")
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

        Update `df` values including null values in `new_df`, using a full outer
        join strategy that defines explicit join columns in each frame:

        >>> df.update(new_df, left_on="A", right_on="C", how="full", include_nulls=True)
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
    def count(self) -> DataFrame:
        """
        Return the number of non-null elements for each column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"a": [1, 2, 3, 4], "b": [1, 2, 1, None], "c": [None, None, None, None]}
        ... )
        >>> df.count()
        shape: (1, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ u32 ┆ u32 ┆ u32 │
        ╞═════╪═════╪═════╡
        │ 4   ┆ 3   ┆ 0   │
        └─────┴─────┴─────┘
        """
    def groupby(self, by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr) -> GroupBy:
        """
        Start a group by operation.

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`DataFrame.group_by`.

        Parameters
        ----------
        by
            Column(s) to group by. Accepts expression input. Strings are parsed as
            column names.
        *more_by
            Additional columns to group by, specified as positional arguments.
        maintain_order
            Ensure that the order of the groups is consistent with the input data.
            This is slower than a default group by.
            Settings this to `True` blocks the possibility
            to run on the streaming engine.

            .. note::
                Within each group, the order of rows is always preserved, regardless
                of this argument.

        Returns
        -------
        GroupBy
            Object which can be used to perform aggregations.
        """
    def groupby_rolling(self, index_column: IntoExpr) -> RollingGroupBy:
        """
        Create rolling groups based on a time, Int32, or Int64 column.

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`DataFrame.rolling`.

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a rolling group by on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        period
            length of the window - must be non-negative
        offset
            offset of the window. Default is -period
        closed : {'right', 'left', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).
        by
            Also group by this column/these columns
        check_sorted
            Check whether the `index` column is sorted (or, if `by` is given,
            check whether it's sorted within each group).
            When the `by` argument is given, polars can not check sortedness
            by the metadata and has to do a full scan on the index column to
            verify data is sorted. This is expensive. If you are sure the
            data within the groups is sorted, you can set this to `False`.
            Doing so incorrectly will lead to incorrect output
        """
    def group_by_rolling(self, index_column: IntoExpr) -> RollingGroupBy:
        """
        Create rolling groups based on a time, Int32, or Int64 column.

        .. deprecated:: 0.19.9
            This method has been renamed to :func:`DataFrame.rolling`.

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a rolling group by on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        period
            length of the window - must be non-negative
        offset
            offset of the window. Default is -period
        closed : {'right', 'left', 'both', 'none'}
            Define which sides of the temporal interval are closed (inclusive).
        by
            Also group by this column/these columns
        check_sorted
            Check whether `index_column` is sorted (or, if `by` is given,
            check whether it's sorted within each group).
            When the `by` argument is given, polars can not check sortedness
            by the metadata and has to do a full scan on the index column to
            verify data is sorted. This is expensive. If you are sure the
            data within the groups is sorted, you can set this to `False`.
            Doing so incorrectly will lead to incorrect output
        """
    def groupby_dynamic(self, index_column: IntoExpr) -> DynamicGroupBy:
        """
        Group based on a time value (or index value of type Int32, Int64).

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`DataFrame.group_by_dynamic`.

        Parameters
        ----------
        index_column
            Column used to group based on the time window.
            Often of type Date/Datetime.
            This column must be sorted in ascending order (or, if `by` is specified,
            then it must be sorted in ascending order within each group).

            In case of a dynamic group by on indices, dtype needs to be one of
            {Int32, Int64}. Note that Int32 gets temporarily cast to Int64, so if
            performance matters use an Int64 column.
        every
            interval of the window
        period
            length of the window, if None it will equal \'every\'
        offset
            offset of the window, only takes effect if `start_by` is `\'window\'`.
            Defaults to negative `every`.
        truncate
            truncate the time value to the window lower bound
        include_boundaries
            Add the lower and upper bound of the window to the "_lower_bound" and
            "_upper_bound" columns. This will impact performance because it\'s harder to
            parallelize
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).
        by
            Also group by this column/these columns
        start_by : {\'window\', \'datapoint\', \'monday\', \'tuesday\', \'wednesday\', \'thursday\', \'friday\', \'saturday\', \'sunday\'}
            The strategy to determine the start of the first window by.

            * \'window\': Start by taking the earliest timestamp, truncating it with
              `every`, and then adding `offset`.
              Note that weekly windows start on Monday.
            * \'datapoint\': Start from the first encountered data point.
            * a day of the week (only takes effect if `every` contains `\'w\'`):

              * \'monday\': Start the window on the Monday before the first data point.
              * \'tuesday\': Start the window on the Tuesday before the first data point.
              * ...
              * \'sunday\': Start the window on the Sunday before the first data point.

              The resulting window is then shifted back until the earliest datapoint
              is in or in front of it.
        check_sorted
            Check whether `index_column` is sorted (or, if `by` is given,
            check whether it\'s sorted within each group).
            When the `by` argument is given, polars can not check sortedness
            by the metadata and has to do a full scan on the index column to
            verify data is sorted. This is expensive. If you are sure the
            data within the groups is sorted, you can set this to `False`.
            Doing so incorrectly will lead to incorrect output

        Returns
        -------
        DynamicGroupBy
            Object you can call `.agg` on to aggregate by groups, the result
            of which will be sorted by `index_column` (but note that if `by` columns are
            passed, it will only be sorted within each `by` group).
        """
    def apply(
        self, function: Callable[[tuple[Any, ...]], Any], return_dtype: PolarsDataType | None = ...
    ) -> DataFrame:
        """
        Apply a custom/user-defined function (UDF) over the rows of the DataFrame.

        .. deprecated:: 0.19.0
            This method has been renamed to :func:`DataFrame.map_rows`.

        Parameters
        ----------
        function
            Custom function or lambda.
        return_dtype
            Output type of the operation. If none given, Polars tries to infer the type.
        inference_size
            Only used in the case when the custom function returns rows.
            This uses the first `n` rows to determine the output schema
        """
    def shift_and_fill(self, fill_value: int | str | float) -> DataFrame:
        """
        Shift values by the given number of places and fill the resulting null values.

        .. deprecated:: 0.19.12
            Use :func:`shift` instead.

        Parameters
        ----------
        fill_value
            fill None values with this value.
        n
            Number of places to shift (may be negative).
        """
    def take_every(self, n: int, offset: int = ...) -> DataFrame:
        """
        Take every nth row in the DataFrame and return as a new DataFrame.

        .. deprecated:: 0.19.14
            This method has been renamed to :func:`gather_every`.

        Parameters
        ----------
        n
            Gather every *n*-th row.
        offset
            Starting index.
        """
    def find_idx_by_name(self, name: str) -> int:
        """
        Find the index of a column by name.

        .. deprecated:: 0.19.14
            This method has been renamed to :func:`get_column_index`.

        Parameters
        ----------
        name
            Name of the column to find.
        """
    def insert_at_idx(self, index: int, column: Series) -> Self:
        """
        Insert a Series at a certain column index. This operation is in place.

        .. deprecated:: 0.19.14
            This method has been renamed to :func:`insert_column`.

        Parameters
        ----------
        index
            Column to insert the new `Series` column.
        column
            `Series` to insert.
        """
    def replace_at_idx(self, index: int, new_column: Series) -> Self:
        """
        Replace a column at an index location.

        .. deprecated:: 0.19.14
            This method has been renamed to :func:`replace_column`.

        Parameters
        ----------
        index
            Column index.
        new_column
            Series that will replace the column.
        """
    def frame_equal(self, other: DataFrame) -> bool:
        """
        Check whether the DataFrame is equal to another DataFrame.

        .. deprecated:: 0.19.16
            This method has been renamed to :func:`equals`.

        Parameters
        ----------
        other
            DataFrame to compare with.
        null_equal
            Consider null values as equal.
        """
    @property
    def plot(self): ...
    @property
    def shape(self): ...
    @property
    def height(self): ...
    @property
    def width(self): ...
    @property
    def dtypes(self): ...
    @property
    def flags(self): ...
    @property
    def schema(self): ...

def _prepare_other_arg(other: Any, length: int | None = ...) -> Series: ...

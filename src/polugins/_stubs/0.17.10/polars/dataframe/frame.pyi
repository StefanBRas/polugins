from _typeshed import Incomplete
from datetime import timedelta
from io import BytesIO, IOBase
from pathlib import Path
from polars.dataframe._html import NotebookFormatter as NotebookFormatter
from polars.dataframe.groupby import DynamicGroupBy as DynamicGroupBy, GroupBy as GroupBy, RollingGroupBy as RollingGroupBy
from polars.datatypes import Boolean as Boolean, Categorical as Categorical, DataTypeClass as DataTypeClass, FLOAT_DTYPES as FLOAT_DTYPES, Float64 as Float64, INTEGER_DTYPES as INTEGER_DTYPES, Int16 as Int16, Int32 as Int32, Int64 as Int64, Int8 as Int8, NUMERIC_DTYPES as NUMERIC_DTYPES, N_INFER_DEFAULT as N_INFER_DEFAULT, Object as Object, SIGNED_INTEGER_DTYPES as SIGNED_INTEGER_DTYPES, UInt16 as UInt16, UInt32 as UInt32, UInt64 as UInt64, UInt8 as UInt8, Utf8 as Utf8, py_type_to_dtype as py_type_to_dtype, unpack_dtypes as unpack_dtypes
from polars.dependencies import _PYARROW_AVAILABLE as _PYARROW_AVAILABLE, _check_for_numpy as _check_for_numpy, _check_for_pandas as _check_for_pandas, _check_for_pyarrow as _check_for_pyarrow, numpy as np, pandas as pd, pyarrow as pa
from polars.exceptions import NoRowsReturnedError as NoRowsReturnedError, TooManyRowsReturnedError as TooManyRowsReturnedError
from polars.expr import Expr as Expr
from polars.functions.lazy import col as col, lit as lit
from polars.io._utils import _is_local_file as _is_local_file
from polars.io.excel._write_utils import _XLFormatCache as _XLFormatCache, _unpack_multi_column_dict as _unpack_multi_column_dict, _xl_apply_conditional_formats as _xl_apply_conditional_formats, _xl_inject_sparklines as _xl_inject_sparklines, _xl_setup_table_columns as _xl_setup_table_columns, _xl_setup_table_options as _xl_setup_table_options, _xl_setup_workbook as _xl_setup_workbook, _xl_unique_table_name as _xl_unique_table_name
from polars.lazyframe import LazyFrame as LazyFrame
from polars.polars import PyDataFrame as PyDataFrame
from polars.series import Series as Series
from polars.slice import PolarsSlice as PolarsSlice
from polars.type_aliases import AsofJoinStrategy as AsofJoinStrategy, AvroCompression as AvroCompression, ClosedInterval as ClosedInterval, ColumnTotalsDefinition as ColumnTotalsDefinition, ComparisonOperator as ComparisonOperator, ConditionalFormatDict as ConditionalFormatDict, CsvEncoding as CsvEncoding, DbWriteEngine as DbWriteEngine, DbWriteMode as DbWriteMode, FillNullStrategy as FillNullStrategy, FrameInitTypes as FrameInitTypes, IntoExpr as IntoExpr, IpcCompression as IpcCompression, JoinStrategy as JoinStrategy, NullStrategy as NullStrategy, OneOrMoreDataTypes as OneOrMoreDataTypes, Orientation as Orientation, ParallelStrategy as ParallelStrategy, ParquetCompression as ParquetCompression, PivotAgg as PivotAgg, PolarsDataType as PolarsDataType, RollingInterpolationMethod as RollingInterpolationMethod, RowTotalsDefinition as RowTotalsDefinition, SchemaDefinition as SchemaDefinition, SchemaDict as SchemaDict, SizeUnit as SizeUnit, StartBy as StartBy, UniqueKeepStrategy as UniqueKeepStrategy, UnstackDirection as UnstackDirection
from polars.utils import NoDefault as NoDefault, no_default as no_default
from polars.utils._construction import _post_apply_columns as _post_apply_columns, arrow_to_pydf as arrow_to_pydf, dict_to_pydf as dict_to_pydf, iterable_to_pydf as iterable_to_pydf, numpy_to_pydf as numpy_to_pydf, pandas_to_pydf as pandas_to_pydf, sequence_to_pydf as sequence_to_pydf, series_to_pydf as series_to_pydf
from polars.utils._parse_expr_input import expr_to_lit_or_expr as expr_to_lit_or_expr
from polars.utils._wrap import wrap_ldf as wrap_ldf, wrap_s as wrap_s
from polars.utils.convert import _timedelta_to_pl_duration as _timedelta_to_pl_duration
from polars.utils.decorators import deprecated_alias as deprecated_alias
from polars.utils.meta import get_index_type as get_index_type
from polars.utils.various import _prepare_row_count_args as _prepare_row_count_args, _process_null_values as _process_null_values, find_stacklevel as find_stacklevel, handle_projection_columns as handle_projection_columns, is_bool_sequence as is_bool_sequence, is_int_sequence as is_int_sequence, is_str_sequence as is_str_sequence, normalise_filepath as normalise_filepath, parse_version as parse_version, range_to_slice as range_to_slice, scale_bytes as scale_bytes
from pyarrow.interchange.dataframe import _PyArrowDataFrame as _PyArrowDataFrame
from typing import Any, BinaryIO, Callable, Concatenate, Iterable, Iterator, Literal, Mapping, NoReturn, Sequence, TypeAlias, TypeVar, overload
from typing_extensions import Self
from xlsxwriter import Workbook as Workbook

MultiRowSelector: TypeAlias
MultiColSelector: TypeAlias
T = TypeVar('T')
P: Incomplete

class DataFrame:
    _accessors: set[str]
    _df: Incomplete
    def __init__(self, data: FrameInitTypes | None = ..., schema: SchemaDefinition | None = ..., *, schema_overrides: SchemaDict | None = ..., orient: Orientation | None = ..., infer_schema_length: int | None = ..., nan_to_null: bool = ...) -> None: ...
    @classmethod
    def _from_pydf(cls, py_df: PyDataFrame) -> Self: ...
    @classmethod
    def _from_dicts(cls, data: Sequence[dict[str, Any]], schema: SchemaDefinition | None = ..., *, schema_overrides: SchemaDict | None = ..., infer_schema_length: int | None = ...) -> Self: ...
    @classmethod
    def _from_dict(cls, data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series], schema: SchemaDefinition | None = ..., *, schema_overrides: SchemaDict | None = ...) -> Self: ...
    @classmethod
    def _from_records(cls, data: Sequence[Sequence[Any]], schema: SchemaDefinition | None = ..., *, schema_overrides: SchemaDict | None = ..., orient: Orientation | None = ..., infer_schema_length: int | None = ...) -> Self: ...
    @classmethod
    def _from_numpy(cls, data: np.ndarray[Any, Any], schema: SchemaDefinition | None = ..., *, schema_overrides: SchemaDict | None = ..., orient: Orientation | None = ...) -> Self: ...
    @classmethod
    def _from_arrow(cls, data: pa.Table, schema: SchemaDefinition | None = ..., *, schema_overrides: SchemaDict | None = ..., rechunk: bool = ...) -> Self: ...
    @classmethod
    def _from_pandas(cls, data: pd.DataFrame, schema: SchemaDefinition | None = ..., *, schema_overrides: SchemaDict | None = ..., rechunk: bool = ..., nan_to_null: bool = ..., include_index: bool = ...) -> Self: ...
    @classmethod
    def _read_csv(cls, source: str | Path | BinaryIO | bytes, *, has_header: bool = ..., columns: Sequence[int] | Sequence[str] | None = ..., separator: str = ..., comment_char: str | None = ..., quote_char: str | None = ..., skip_rows: int = ..., dtypes: None | SchemaDict | Sequence[PolarsDataType] = ..., null_values: str | Sequence[str] | dict[str, str] | None = ..., missing_utf8_is_empty_string: bool = ..., ignore_errors: bool = ..., try_parse_dates: bool = ..., n_threads: int | None = ..., infer_schema_length: int | None = ..., batch_size: int = ..., n_rows: int | None = ..., encoding: CsvEncoding = ..., low_memory: bool = ..., rechunk: bool = ..., skip_rows_after_header: int = ..., row_count_name: str | None = ..., row_count_offset: int = ..., sample_size: int = ..., eol_char: str = ...) -> Self: ...
    @classmethod
    def _read_parquet(cls, source: str | Path | BinaryIO, *, columns: Sequence[int] | Sequence[str] | None = ..., n_rows: int | None = ..., parallel: ParallelStrategy = ..., row_count_name: str | None = ..., row_count_offset: int = ..., low_memory: bool = ..., use_statistics: bool = ..., rechunk: bool = ...) -> Self: ...
    @classmethod
    def _read_avro(cls, source: str | Path | BinaryIO, *, columns: Sequence[int] | Sequence[str] | None = ..., n_rows: int | None = ...) -> Self: ...
    @classmethod
    def _read_ipc(cls, source: str | Path | BinaryIO, *, columns: Sequence[int] | Sequence[str] | None = ..., n_rows: int | None = ..., row_count_name: str | None = ..., row_count_offset: int = ..., rechunk: bool = ..., memory_map: bool = ...) -> Self: ...
    @classmethod
    def _read_json(cls, source: str | Path | IOBase) -> Self: ...
    @classmethod
    def _read_ndjson(cls, source: str | Path | IOBase) -> Self: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def height(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def columns(self) -> list[str]: ...
    @columns.setter
    def columns(self, names: Sequence[str]) -> None: ...
    @property
    def dtypes(self) -> list[PolarsDataType]: ...
    @property
    def schema(self) -> SchemaDict: ...
    def __array__(self, dtype: Any = ...) -> np.ndarray[Any, Any]: ...
    def __dataframe__(self, nan_as_null: bool = ..., allow_copy: bool = ...) -> _PyArrowDataFrame: ...
    def _comp(self, other: Any, op: ComparisonOperator) -> DataFrame: ...
    def _compare_to_other_df(self, other: DataFrame, op: ComparisonOperator) -> DataFrame: ...
    def _compare_to_non_df(self, other: Any, op: ComparisonOperator) -> Self: ...
    def _div(self, other: Any, floordiv: bool) -> Self: ...
    def _cast_all_from_to(self, df: DataFrame, from_: frozenset[PolarsDataType], to: PolarsDataType) -> DataFrame: ...
    def __floordiv__(self, other: DataFrame | Series | int | float) -> Self: ...
    def __truediv__(self, other: DataFrame | Series | int | float) -> Self: ...
    def __bool__(self) -> NoReturn: ...
    def __eq__(self, other: Any) -> DataFrame: ...
    def __ne__(self, other: Any) -> DataFrame: ...
    def __gt__(self, other: Any) -> DataFrame: ...
    def __lt__(self, other: Any) -> DataFrame: ...
    def __ge__(self, other: Any) -> DataFrame: ...
    def __le__(self, other: Any) -> DataFrame: ...
    def __getstate__(self) -> list[Series]: ...
    def __setstate__(self, state) -> None: ...
    def __mul__(self, other: DataFrame | Series | int | float) -> Self: ...
    def __rmul__(self, other: DataFrame | Series | int | float) -> Self: ...
    def __add__(self, other: DataFrame | Series | int | float | bool | str) -> Self: ...
    def __radd__(self, other: DataFrame | Series | int | float | bool | str) -> Self: ...
    def __sub__(self, other: DataFrame | Series | int | float) -> Self: ...
    def __mod__(self, other: DataFrame | Series | int | float) -> Self: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __contains__(self, key: str) -> bool: ...
    def __iter__(self) -> Iterator[Any]: ...
    def _pos_idx(self, idx: int, dim: int) -> int: ...
    def _pos_idxs(self, idxs: np.ndarray[Any, Any] | Series, dim: int) -> Series: ...
    @overload
    def __getitem__(self, item: str) -> Series: ...
    @overload
    def __getitem__(self, item: int | np.ndarray[Any, Any] | MultiColSelector | tuple[int, MultiColSelector] | tuple[MultiRowSelector, MultiColSelector]) -> Self: ...
    @overload
    def __getitem__(self, item: tuple[int, int | str]) -> Any: ...
    @overload
    def __getitem__(self, item: tuple[MultiRowSelector, int | str]) -> Series: ...
    def __setitem__(self, key: str | Sequence[int] | Sequence[str] | tuple[Any, str | int], value: Any) -> None: ...
    def __len__(self) -> int: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: None = ...) -> Self: ...
    def _ipython_key_completions_(self) -> list[str]: ...
    def _repr_html_(self, **kwargs: Any) -> str: ...
    def item(self, row: int | None = ..., column: int | str | None = ...) -> Any: ...
    def to_arrow(self) -> pa.Table: ...
    @overload
    def to_dict(self, as_series: Literal[True] = ...) -> dict[str, Series]: ...
    @overload
    def to_dict(self, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(self, as_series: bool = ...) -> dict[str, Series] | dict[str, list[Any]]: ...
    def to_dicts(self) -> list[dict[str, Any]]: ...
    def to_numpy(self) -> np.ndarray[Any, Any]: ...
    def to_pandas(self, *args: Any, use_pyarrow_extension_array: bool = ..., **kwargs: Any) -> pd.DataFrame: ...
    def to_series(self, index: int = ...) -> Series: ...
    def to_init_repr(self, n: int = ...) -> str: ...
    @overload
    def write_json(self, file: None = ..., *, pretty: bool = ..., row_oriented: bool = ...) -> str: ...
    @overload
    def write_json(self, file: IOBase | str | Path, *, pretty: bool = ..., row_oriented: bool = ...) -> None: ...
    @overload
    def write_ndjson(self, file: None = ...) -> str: ...
    @overload
    def write_ndjson(self, file: IOBase | str | Path) -> None: ...
    @overload
    def write_csv(self, file: None = ..., *, has_header: bool = ..., separator: str = ..., quote: str = ..., batch_size: int = ..., datetime_format: str | None = ..., date_format: str | None = ..., time_format: str | None = ..., float_precision: int | None = ..., null_value: str | None = ...) -> str: ...
    @overload
    def write_csv(self, file: BytesIO | str | Path, *, has_header: bool = ..., separator: str = ..., quote: str = ..., batch_size: int = ..., datetime_format: str | None = ..., date_format: str | None = ..., time_format: str | None = ..., float_precision: int | None = ..., null_value: str | None = ...) -> None: ...
    def write_avro(self, file: BinaryIO | BytesIO | str | Path, compression: AvroCompression = ...) -> None: ...
    def write_excel(self, workbook: Workbook | BytesIO | Path | str | None = ..., worksheet: str | None = ..., *, position: tuple[int, int] | str = ..., table_style: str | dict[str, Any] | None = ..., table_name: str | None = ..., column_formats: dict[str | tuple[str, ...], str | dict[str, str]] | None = ..., dtype_formats: dict[OneOrMoreDataTypes, str] | None = ..., conditional_formats: ConditionalFormatDict | None = ..., column_totals: ColumnTotalsDefinition | None = ..., column_widths: dict[str | tuple[str, ...], int] | int | None = ..., row_totals: RowTotalsDefinition | None = ..., row_heights: dict[int | tuple[int, ...], int] | int | None = ..., sparklines: dict[str, Sequence[str] | dict[str, Any]] | None = ..., formulas: dict[str, str | dict[str, str]] | None = ..., float_precision: int = ..., has_header: bool = ..., autofilter: bool = ..., autofit: bool = ..., hidden_columns: Sequence[str] | None = ..., hide_gridlines: bool = ..., sheet_zoom: int | None = ...) -> Workbook: ...
    @overload
    def write_ipc(self, file: None, compression: IpcCompression = ...) -> BytesIO: ...
    @overload
    def write_ipc(self, file: BinaryIO | BytesIO | str | Path, compression: IpcCompression = ...) -> None: ...
    def write_parquet(self, file: str | Path | BytesIO, *, compression: ParquetCompression = ..., compression_level: int | None = ..., statistics: bool = ..., row_group_size: int | None = ..., use_pyarrow: bool = ..., pyarrow_options: dict[str, object] | None = ...) -> None: ...
    def write_database(self, table_name: str, connection_uri: str, *, if_exists: DbWriteMode = ..., engine: DbWriteEngine = ...) -> None: ...
    def estimated_size(self, unit: SizeUnit = ...) -> int | float: ...
    def transpose(self, *, include_header: bool = ..., header_name: str = ..., column_names: Iterator[str] | Sequence[str] | None = ...) -> Self: ...
    def reverse(self) -> Self: ...
    def rename(self, mapping: dict[str, str]) -> Self: ...
    def insert_at_idx(self, index: int, series: Series) -> Self: ...
    def filter(self, predicate: Expr | str | Series | list[bool] | np.ndarray[Any, Any] | bool) -> Self: ...
    @overload
    def glimpse(self, *, return_as_string: Literal[False]) -> None: ...
    @overload
    def glimpse(self, *, return_as_string: Literal[True]) -> str: ...
    def describe(self, percentiles: Sequence[float] | float | None = ...) -> Self: ...
    def find_idx_by_name(self, name: str) -> int: ...
    def replace_at_idx(self, index: int, series: Series) -> Self: ...
    def sort(self, by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr, descending: bool | Sequence[bool] = ..., nulls_last: bool = ...) -> Self: ...
    def top_k(self, k: int, *, by: IntoExpr | Iterable[IntoExpr], descending: bool | Sequence[bool] = ..., nulls_last: bool = ...) -> Self: ...
    def bottom_k(self, k: int, *, by: IntoExpr | Iterable[IntoExpr], descending: bool | Sequence[bool] = ..., nulls_last: bool = ...) -> Self: ...
    def frame_equal(self, other: DataFrame, *, null_equal: bool = ...) -> bool: ...
    def replace(self, column: str, new_column: Series) -> Self: ...
    def slice(self, offset: int, length: int | None = ...) -> Self: ...
    def head(self, n: int = ...) -> Self: ...
    def tail(self, n: int = ...) -> Self: ...
    def limit(self, n: int = ...) -> Self: ...
    def drop_nulls(self, subset: str | Sequence[str] | None = ...) -> Self: ...
    def pipe(self, function: Callable[Concatenate[DataFrame, P], T], *args: P.args, **kwargs: P.kwargs) -> T: ...
    def with_row_count(self, name: str = ..., offset: int = ...) -> Self: ...
    def groupby(self, by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr, maintain_order: bool = ...) -> GroupBy[Self]: ...
    def groupby_rolling(self, index_column: IntoExpr, *, period: str | timedelta, offset: str | timedelta | None = ..., closed: ClosedInterval = ..., by: IntoExpr | Iterable[IntoExpr] | None = ...) -> RollingGroupBy[Self]: ...
    def groupby_dynamic(self, index_column: IntoExpr, *, every: str | timedelta, period: str | timedelta | None = ..., offset: str | timedelta | None = ..., truncate: bool = ..., include_boundaries: bool = ..., closed: ClosedInterval = ..., by: IntoExpr | Iterable[IntoExpr] | None = ..., start_by: StartBy = ...) -> DynamicGroupBy[Self]: ...
    def upsample(self, time_column: str, *, every: str | timedelta, offset: str | timedelta | None = ..., by: str | Sequence[str] | None = ..., maintain_order: bool = ...) -> Self: ...
    def join_asof(self, other: DataFrame, *, left_on: str | None | Expr = ..., right_on: str | None | Expr = ..., on: str | None | Expr = ..., by_left: str | Sequence[str] | None = ..., by_right: str | Sequence[str] | None = ..., by: str | Sequence[str] | None = ..., strategy: AsofJoinStrategy = ..., suffix: str = ..., tolerance: str | int | float | None = ..., allow_parallel: bool = ..., force_parallel: bool = ...) -> Self: ...
    def join(self, other: DataFrame, on: str | Expr | Sequence[str | Expr] | None = ..., how: JoinStrategy = ..., *, left_on: str | Expr | Sequence[str | Expr] | None = ..., right_on: str | Expr | Sequence[str | Expr] | None = ..., suffix: str = ...) -> Self: ...
    def apply(self, function: Callable[[tuple[Any, ...]], Any], return_dtype: PolarsDataType | None = ..., *, inference_size: int = ...) -> Self: ...
    def hstack(self, columns: list[Series] | DataFrame, *, in_place: bool = ...) -> Self: ...
    def vstack(self, df: DataFrame, *, in_place: bool = ...) -> Self: ...
    def extend(self, other: Self) -> Self: ...
    def drop(self, columns: str | Sequence[str], *more_columns: str) -> Self: ...
    def drop_in_place(self, name: str) -> Series: ...
    def clear(self, n: int = ...) -> Self: ...
    def clone(self) -> Self: ...
    def get_columns(self) -> list[Series]: ...
    def get_column(self, name: str) -> Series: ...
    def fill_null(self, value: Any | None = ..., strategy: FillNullStrategy | None = ..., limit: int | None = ..., *, matches_supertype: bool = ...) -> Self: ...
    def fill_nan(self, value: Expr | int | float | None) -> Self: ...
    def explode(self, columns: str | Sequence[str] | Expr | Sequence[Expr], *more_columns: str | Expr) -> Self: ...
    def pivot(self, values: Sequence[str] | str, index: Sequence[str] | str, columns: Sequence[str] | str, aggregate_function: PivotAgg | Expr | None | NoDefault = ..., *, maintain_order: bool = ..., sort_columns: bool = ..., separator: str = ...) -> Self: ...
    def melt(self, id_vars: Sequence[str] | str | None = ..., value_vars: Sequence[str] | str | None = ..., variable_name: str | None = ..., value_name: str | None = ...) -> Self: ...
    def unstack(self, step: int, how: UnstackDirection = ..., columns: str | Sequence[str] | None = ..., fill_values: list[Any] | None = ...) -> Self: ...
    @overload
    def partition_by(self, by: str | Iterable[str], *more_by: str, maintain_order: bool = ..., as_dict: Literal[False] = ...) -> list[Self]: ...
    @overload
    def partition_by(self, by: str | Iterable[str], *more_by: str, maintain_order: bool = ..., as_dict: Literal[True]) -> dict[Any, Self]: ...
    def shift(self, periods: int) -> Self: ...
    def shift_and_fill(self, fill_value: int | str | float, *, periods: int = ...) -> Self: ...
    def is_duplicated(self) -> Series: ...
    def is_unique(self) -> Series: ...
    def lazy(self) -> LazyFrame: ...
    def select(self, exprs: IntoExpr | Iterable[IntoExpr] | None = ..., *more_exprs: IntoExpr, **named_exprs: IntoExpr) -> Self: ...
    def with_columns(self, exprs: IntoExpr | Iterable[IntoExpr] | None = ..., *more_exprs: IntoExpr, **named_exprs: IntoExpr) -> Self: ...
    @overload
    def n_chunks(self, strategy: Literal['first'] = ...) -> int: ...
    @overload
    def n_chunks(self, strategy: Literal['all']) -> list[int]: ...
    @overload
    def max(self, axis: Literal[0] = ...) -> Self: ...
    @overload
    def max(self, axis: Literal[1]) -> Series: ...
    @overload
    def max(self, axis: int = ...) -> Self | Series: ...
    @overload
    def min(self, axis: Literal[0] = ...) -> Self: ...
    @overload
    def min(self, axis: Literal[1]) -> Series: ...
    @overload
    def min(self, axis: int = ...) -> Self | Series: ...
    @overload
    def sum(self, *, axis: Literal[0] = ..., null_strategy: NullStrategy = ...) -> Self: ...
    @overload
    def sum(self, *, axis: Literal[1], null_strategy: NullStrategy = ...) -> Series: ...
    @overload
    def sum(self, *, axis: int = ..., null_strategy: NullStrategy = ...) -> Self | Series: ...
    @overload
    def mean(self, *, axis: Literal[0] = ..., null_strategy: NullStrategy = ...) -> Self: ...
    @overload
    def mean(self, *, axis: Literal[1], null_strategy: NullStrategy = ...) -> Series: ...
    @overload
    def mean(self, *, axis: int = ..., null_strategy: NullStrategy = ...) -> Self | Series: ...
    def std(self, ddof: int = ...) -> Self: ...
    def var(self, ddof: int = ...) -> Self: ...
    def median(self) -> Self: ...
    def product(self) -> Self: ...
    def quantile(self, quantile: float, interpolation: RollingInterpolationMethod = ...) -> Self: ...
    def to_dummies(self, columns: str | Sequence[str] | None = ..., *, separator: str = ...) -> Self: ...
    def unique(self, subset: str | Sequence[str] | None = ..., *, keep: UniqueKeepStrategy = ..., maintain_order: bool = ...) -> Self: ...
    def n_unique(self, subset: str | Expr | Sequence[str | Expr] | None = ...) -> int: ...
    def rechunk(self) -> Self: ...
    def null_count(self) -> Self: ...
    def sample(self, n: int | None = ..., *, fraction: float | None = ..., with_replacement: bool = ..., shuffle: bool = ..., seed: int | None = ...) -> Self: ...
    def fold(self, operation: Callable[[Series, Series], Series]) -> Series: ...
    @overload
    def row(self, index: int | None = ..., *, by_predicate: Expr | None = ..., named: Literal[False] = ...) -> tuple[Any, ...]: ...
    @overload
    def row(self, index: int | None = ..., *, by_predicate: Expr | None = ..., named: Literal[True]) -> dict[str, Any]: ...
    @overload
    def rows(self, *, named: Literal[False] = ...) -> list[tuple[Any, ...]]: ...
    @overload
    def rows(self, *, named: Literal[True]) -> list[dict[str, Any]]: ...
    @overload
    def iter_rows(self, *, named: Literal[False] = ..., buffer_size: int = ...) -> Iterator[tuple[Any, ...]]: ...
    @overload
    def iter_rows(self, *, named: Literal[True], buffer_size: int = ...) -> Iterator[dict[str, Any]]: ...
    def iter_slices(self, n_rows: int = ...) -> Iterator[DataFrame]: ...
    def shrink_to_fit(self, *, in_place: bool = ...) -> Self: ...
    def take_every(self, n: int) -> Self: ...
    def hash_rows(self, seed: int = ..., seed_1: int | None = ..., seed_2: int | None = ..., seed_3: int | None = ...) -> Series: ...
    def interpolate(self) -> Self: ...
    def is_empty(self) -> bool: ...
    def to_struct(self, name: str) -> Series: ...
    def unnest(self, columns: str | Sequence[str], *more_columns: str) -> Self: ...
    def corr(self, **kwargs): ...
    def merge_sorted(self, other: DataFrame, key: str) -> Self: ...
    def set_sorted(self, column: str | Iterable[str], *more_columns: str, descending: bool = ...) -> Self: ...
    def update(self, other: DataFrame, on: str | Sequence[str] | None = ..., how: Literal['left', 'inner'] = ...) -> Self: ...

def _prepare_other_arg(other: Any, length: int | None = ...) -> Series: ...

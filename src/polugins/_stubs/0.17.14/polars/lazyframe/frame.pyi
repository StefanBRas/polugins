import pyarrow as pa
from _typeshed import Incomplete
from datetime import timedelta
from io import IOBase
from pathlib import Path
from polars import DataFrame as DataFrame, Expr as Expr, Series as Series
from polars.datatypes import Boolean as Boolean, Categorical as Categorical, DTYPE_TEMPORAL_UNITS as DTYPE_TEMPORAL_UNITS, Date as Date, Datetime as Datetime, Duration as Duration, Float32 as Float32, Float64 as Float64, Int16 as Int16, Int32 as Int32, Int64 as Int64, Int8 as Int8, N_INFER_DEFAULT as N_INFER_DEFAULT, Time as Time, UInt16 as UInt16, UInt32 as UInt32, UInt64 as UInt64, UInt8 as UInt8, Utf8 as Utf8, py_type_to_dtype as py_type_to_dtype
from polars.dependencies import subprocess as subprocess
from polars.io._utils import _is_local_file as _is_local_file
from polars.io.ipc.anonymous_scan import _scan_ipc_fsspec as _scan_ipc_fsspec
from polars.io.parquet.anonymous_scan import _scan_parquet_fsspec as _scan_parquet_fsspec
from polars.lazyframe.groupby import LazyGroupBy as LazyGroupBy
from polars.polars import PyLazyFrame as PyLazyFrame
from polars.slice import LazyPolarsSlice as LazyPolarsSlice
from polars.type_aliases import AsofJoinStrategy as AsofJoinStrategy, ClosedInterval as ClosedInterval, CsvEncoding as CsvEncoding, FillNullStrategy as FillNullStrategy, FrameInitTypes as FrameInitTypes, IntoExpr as IntoExpr, JoinStrategy as JoinStrategy, Orientation as Orientation, ParallelStrategy as ParallelStrategy, PolarsDataType as PolarsDataType, RollingInterpolationMethod as RollingInterpolationMethod, SchemaDefinition as SchemaDefinition, SchemaDict as SchemaDict, StartBy as StartBy, UniqueKeepStrategy as UniqueKeepStrategy
from polars.utils._parse_expr_input import expr_to_lit_or_expr as expr_to_lit_or_expr, selection_to_pyexpr_list as selection_to_pyexpr_list
from polars.utils._wrap import wrap_df as wrap_df, wrap_expr as wrap_expr
from polars.utils.convert import _timedelta_to_pl_duration as _timedelta_to_pl_duration
from polars.utils.various import _in_notebook as _in_notebook, _prepare_row_count_args as _prepare_row_count_args, _process_null_values as _process_null_values, find_stacklevel as find_stacklevel, normalise_filepath as normalise_filepath
from typing import Any, Callable, Concatenate, Iterable, Literal, NoReturn, Sequence, TypeVar, overload
from typing_extensions import Self

T = TypeVar('T')
P: Incomplete

class LazyFrame:
    _ldf: PyLazyFrame
    _accessors: set[str]
    def __init__(self, data: FrameInitTypes | None = ..., schema: SchemaDefinition | None = ..., *, schema_overrides: SchemaDict | None = ..., orient: Orientation | None = ..., infer_schema_length: int | None = ..., nan_to_null: bool = ...) -> None: ...
    @classmethod
    def _from_pyldf(cls, ldf: PyLazyFrame) -> Self: ...
    def __getstate__(self) -> Any: ...
    def __setstate__(self, state) -> None: ...
    @classmethod
    def _scan_csv(cls, source: str, *, has_header: bool = ..., separator: str = ..., comment_char: str | None = ..., quote_char: str | None = ..., skip_rows: int = ..., dtypes: SchemaDict | None = ..., null_values: str | Sequence[str] | dict[str, str] | None = ..., missing_utf8_is_empty_string: bool = ..., ignore_errors: bool = ..., cache: bool = ..., with_column_names: Callable[[list[str]], list[str]] | None = ..., infer_schema_length: int | None = ..., n_rows: int | None = ..., encoding: CsvEncoding = ..., low_memory: bool = ..., rechunk: bool = ..., skip_rows_after_header: int = ..., row_count_name: str | None = ..., row_count_offset: int = ..., try_parse_dates: bool = ..., eol_char: str = ...) -> Self: ...
    @classmethod
    def _scan_parquet(cls, source: str, *, n_rows: int | None = ..., cache: bool = ..., parallel: ParallelStrategy = ..., rechunk: bool = ..., row_count_name: str | None = ..., row_count_offset: int = ..., storage_options: dict[str, object] | None = ..., low_memory: bool = ..., use_statistics: bool = ...) -> Self: ...
    @classmethod
    def _scan_ipc(cls, source: str | Path, *, n_rows: int | None = ..., cache: bool = ..., rechunk: bool = ..., row_count_name: str | None = ..., row_count_offset: int = ..., storage_options: dict[str, object] | None = ..., memory_map: bool = ...) -> Self: ...
    @classmethod
    def _scan_ndjson(cls, source: str, *, infer_schema_length: int | None = ..., batch_size: int | None = ..., n_rows: int | None = ..., low_memory: bool = ..., rechunk: bool = ..., row_count_name: str | None = ..., row_count_offset: int = ...) -> Self: ...
    @classmethod
    def _scan_python_function(cls, schema: pa.schema | dict[str, PolarsDataType], scan_fn: bytes, pyarrow: bool = ...) -> Self: ...
    @classmethod
    def from_json(cls, json: str) -> Self: ...
    @classmethod
    def read_json(cls, file: str | Path | IOBase) -> Self: ...
    @property
    def columns(self) -> list[str]: ...
    @property
    def dtypes(self) -> list[PolarsDataType]: ...
    @property
    def schema(self) -> SchemaDict: ...
    @property
    def width(self) -> int: ...
    def __bool__(self) -> NoReturn: ...
    def __contains__(self, key: str) -> bool: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: None = ...) -> Self: ...
    def __getitem__(self, item: int | range | slice) -> Self: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def _repr_html_(self) -> str: ...
    @overload
    def write_json(self, file: None = ...) -> str: ...
    @overload
    def write_json(self, file: IOBase | str | Path) -> None: ...
    def pipe(self, function: Callable[Concatenate[LazyFrame, P], T], *args: P.args, **kwargs: P.kwargs) -> T: ...
    def explain(self, *, optimized: bool = ..., type_coercion: bool = ..., predicate_pushdown: bool = ..., projection_pushdown: bool = ..., simplify_expression: bool = ..., slice_pushdown: bool = ..., common_subplan_elimination: bool = ..., streaming: bool = ...) -> str: ...
    def describe_plan(self, *, optimized: bool = ..., type_coercion: bool = ..., predicate_pushdown: bool = ..., projection_pushdown: bool = ..., simplify_expression: bool = ..., slice_pushdown: bool = ..., common_subplan_elimination: bool = ..., streaming: bool = ...) -> str: ...
    def describe_optimized_plan(self, *, type_coercion: bool = ..., predicate_pushdown: bool = ..., projection_pushdown: bool = ..., simplify_expression: bool = ..., slice_pushdown: bool = ..., common_subplan_elimination: bool = ..., streaming: bool = ...) -> str: ...
    def show_graph(self, *, optimized: bool = ..., show: bool = ..., output_path: str | None = ..., raw_output: bool = ..., figsize: tuple[float, float] = ..., type_coercion: bool = ..., predicate_pushdown: bool = ..., projection_pushdown: bool = ..., simplify_expression: bool = ..., slice_pushdown: bool = ..., common_subplan_elimination: bool = ..., streaming: bool = ...) -> str | None: ...
    def inspect(self, fmt: str = ...) -> Self: ...
    def sort(self, by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr, descending: bool | Sequence[bool] = ..., nulls_last: bool = ...) -> Self: ...
    def top_k(self, k: int, *, by: IntoExpr | Iterable[IntoExpr], descending: bool | Sequence[bool] = ..., nulls_last: bool = ...) -> Self: ...
    def bottom_k(self, k: int, *, by: IntoExpr | Iterable[IntoExpr], descending: bool | Sequence[bool] = ..., nulls_last: bool = ...) -> Self: ...
    def profile(self, *, type_coercion: bool = ..., predicate_pushdown: bool = ..., projection_pushdown: bool = ..., simplify_expression: bool = ..., no_optimization: bool = ..., slice_pushdown: bool = ..., common_subplan_elimination: bool = ..., show_plot: bool = ..., truncate_nodes: int = ..., figsize: tuple[int, int] = ..., streaming: bool = ...) -> tuple[DataFrame, DataFrame]: ...
    def collect(self, *, type_coercion: bool = ..., predicate_pushdown: bool = ..., projection_pushdown: bool = ..., simplify_expression: bool = ..., no_optimization: bool = ..., slice_pushdown: bool = ..., common_subplan_elimination: bool = ..., streaming: bool = ...) -> DataFrame: ...
    def sink_parquet(self, path: str | Path, *, compression: str = ..., compression_level: int | None = ..., statistics: bool = ..., row_group_size: int | None = ..., data_pagesize_limit: int | None = ..., maintain_order: bool = ..., type_coercion: bool = ..., predicate_pushdown: bool = ..., projection_pushdown: bool = ..., simplify_expression: bool = ..., no_optimization: bool = ..., slice_pushdown: bool = ...) -> DataFrame: ...
    def sink_ipc(self, path: str | Path, *, compression: str | None = ..., maintain_order: bool = ..., type_coercion: bool = ..., predicate_pushdown: bool = ..., projection_pushdown: bool = ..., simplify_expression: bool = ..., no_optimization: bool = ..., slice_pushdown: bool = ...) -> DataFrame: ...
    def fetch(self, n_rows: int = ..., *, type_coercion: bool = ..., predicate_pushdown: bool = ..., projection_pushdown: bool = ..., simplify_expression: bool = ..., no_optimization: bool = ..., slice_pushdown: bool = ..., common_subplan_elimination: bool = ..., streaming: bool = ...) -> DataFrame: ...
    def lazy(self) -> Self: ...
    def cache(self) -> Self: ...
    def clear(self, n: int = ...) -> Self: ...
    def clone(self) -> Self: ...
    def filter(self, predicate: Expr | str | Series | list[bool]) -> Self: ...
    def select(self, exprs: IntoExpr | Iterable[IntoExpr] | None = ..., *more_exprs: IntoExpr, **named_exprs: IntoExpr) -> Self: ...
    def groupby(self, by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr, maintain_order: bool = ...) -> LazyGroupBy[Self]: ...
    def groupby_rolling(self, index_column: IntoExpr, *, period: str | timedelta, offset: str | timedelta | None = ..., closed: ClosedInterval = ..., by: IntoExpr | Iterable[IntoExpr] | None = ...) -> LazyGroupBy[Self]: ...
    def groupby_dynamic(self, index_column: IntoExpr, *, every: str | timedelta, period: str | timedelta | None = ..., offset: str | timedelta | None = ..., truncate: bool = ..., include_boundaries: bool = ..., closed: ClosedInterval = ..., by: IntoExpr | Iterable[IntoExpr] | None = ..., start_by: StartBy = ...) -> LazyGroupBy[Self]: ...
    def join_asof(self, other: LazyFrame, *, left_on: str | None | Expr = ..., right_on: str | None | Expr = ..., on: str | None | Expr = ..., by_left: str | Sequence[str] | None = ..., by_right: str | Sequence[str] | None = ..., by: str | Sequence[str] | None = ..., strategy: AsofJoinStrategy = ..., suffix: str = ..., tolerance: str | int | float | None = ..., allow_parallel: bool = ..., force_parallel: bool = ...) -> Self: ...
    def join(self, other: LazyFrame, on: str | Expr | Sequence[str | Expr] | None = ..., how: JoinStrategy = ..., *, left_on: str | Expr | Sequence[str | Expr] | None = ..., right_on: str | Expr | Sequence[str | Expr] | None = ..., suffix: str = ..., allow_parallel: bool = ..., force_parallel: bool = ...) -> Self: ...
    def with_columns(self, exprs: IntoExpr | Iterable[IntoExpr] | None = ..., *more_exprs: IntoExpr, **named_exprs: IntoExpr) -> Self: ...
    def with_context(self, other): ...
    def drop(self, columns: str | Sequence[str], *more_columns: str) -> Self: ...
    def rename(self, mapping: dict[str, str]) -> Self: ...
    def reverse(self) -> Self: ...
    def shift(self, periods: int) -> Self: ...
    def shift_and_fill(self, fill_value: Expr | int | str | float, *, periods: int = ...) -> Self: ...
    def slice(self, offset: int, length: int | None = ...) -> Self: ...
    def limit(self, n: int = ...) -> Self: ...
    def head(self, n: int = ...) -> Self: ...
    def tail(self, n: int = ...) -> Self: ...
    def last(self) -> Self: ...
    def first(self) -> Self: ...
    def approx_unique(self) -> Self: ...
    def with_row_count(self, name: str = ..., offset: int = ...) -> Self: ...
    def take_every(self, n: int) -> Self: ...
    def fill_null(self, value: Any | None = ..., strategy: FillNullStrategy | None = ..., limit: int | None = ..., *, matches_supertype: bool = ...) -> Self: ...
    def fill_nan(self, value: int | float | Expr | None) -> Self: ...
    def std(self, ddof: int = ...) -> Self: ...
    def var(self, ddof: int = ...) -> Self: ...
    def max(self) -> Self: ...
    def min(self) -> Self: ...
    def sum(self) -> Self: ...
    def mean(self) -> Self: ...
    def median(self) -> Self: ...
    def null_count(self) -> Self: ...
    def quantile(self, quantile: float | Expr, interpolation: RollingInterpolationMethod = ...) -> Self: ...
    def explode(self, columns: str | Sequence[str] | Expr | Sequence[Expr], *more_columns: str | Expr) -> Self: ...
    def unique(self, subset: str | Sequence[str] | None = ..., *, keep: UniqueKeepStrategy = ..., maintain_order: bool = ...) -> Self: ...
    def drop_nulls(self, subset: str | Sequence[str] | None = ...) -> Self: ...
    def melt(self, id_vars: str | list[str] | None = ..., value_vars: str | list[str] | None = ..., variable_name: str | None = ..., value_name: str | None = ..., *, streamable: bool = ...) -> Self: ...
    def map(self, function: Callable[[DataFrame], DataFrame], *, predicate_pushdown: bool = ..., projection_pushdown: bool = ..., slice_pushdown: bool = ..., no_optimizations: bool = ..., schema: None | SchemaDict = ..., validate_output_schema: bool = ..., streamable: bool = ...) -> Self: ...
    def interpolate(self) -> Self: ...
    def unnest(self, columns: str | Sequence[str], *more_columns: str) -> Self: ...
    def merge_sorted(self, other: LazyFrame, key: str) -> Self: ...
    def set_sorted(self, column: str | Iterable[str], *more_columns: str, descending: bool = ...) -> Self: ...
    def update(self, other: LazyFrame, on: str | Sequence[str] | None = ..., how: Literal['left', 'inner'] = ...) -> Self: ...

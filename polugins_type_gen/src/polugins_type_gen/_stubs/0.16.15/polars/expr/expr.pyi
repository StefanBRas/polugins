#: version 0.16.15
import P
import np as np
from datetime import timedelta
from polars.datatypes.classes import Struct as Struct, UInt32 as UInt32
from polars.datatypes.convert import (
    is_polars_dtype as is_polars_dtype,
    py_type_to_dtype as py_type_to_dtype,
)
from polars.dependencies import _check_for_numpy as _check_for_numpy
from polars.expr.binary import ExprBinaryNameSpace as ExprBinaryNameSpace
from polars.expr.categorical import ExprCatNameSpace as ExprCatNameSpace
from polars.expr.datetime import ExprDateTimeNameSpace as ExprDateTimeNameSpace
from polars.expr.list import ExprListNameSpace as ExprListNameSpace
from polars.expr.meta import ExprMetaNameSpace as ExprMetaNameSpace
from polars.expr.string import ExprStringNameSpace as ExprStringNameSpace
from polars.expr.struct import ExprStructNameSpace as ExprStructNameSpace
from polars.utils._parse_expr_input import (
    expr_to_lit_or_expr as expr_to_lit_or_expr,
    selection_to_pyexpr_list as selection_to_pyexpr_list,
)
from polars.utils.convert import _timedelta_to_pl_duration as _timedelta_to_pl_duration
from polars.utils.decorators import (
    deprecate_nonkeyword_arguments as deprecate_nonkeyword_arguments,
    deprecated_alias as deprecated_alias,
)
from polars.utils.meta import threadpool_size as threadpool_size
from polars.utils.various import sphinx_accessor as sphinx_accessor
from typing import Any, Callable, ClassVar, Collection, Iterable, NoReturn

TYPE_CHECKING: bool
py_arg_where: builtin_function_or_method

class Expr:
    _pyexpr: ClassVar[None] = ...
    _accessors: ClassVar[set] = ...
    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Self: ...
    def _to_pyexpr(self, other: Any) -> PyExpr: ...
    def _to_expr(self, other: Any) -> Expr: ...
    def _repr_html_(self) -> str: ...
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
    def and_(self, *others: Any) -> Self:
        """Method equivalent of operator expression ``expr & other1 & other2 & ...``."""
    def or_(self, *others: Any) -> Self:
        """Method equivalent of operator expression ``expr | other1 | other2 | ...``."""
    def eq(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr == other``."""
    def ge(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr >= other``."""
    def gt(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr > other``."""
    def le(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr <= other``."""
    def lt(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr < other``."""
    def ne(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr != other``."""
    def add(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr + other``."""
    def floordiv(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr // other``."""
    def mod(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr % other``."""
    def mul(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr * other``."""
    def sub(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr - other``."""
    def truediv(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr / other``."""
    def xor(self, other: Any) -> Self:
        """Method equivalent of operator expression ``expr ^ other``."""
    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any
    ) -> Self:
        """Numpy universal functions."""
    def to_physical(self) -> Self:
        """
        Cast to physical representation of the logical dtype.

        - :func:`polars.datatypes.Date` -> :func:`polars.datatypes.Int32`
        - :func:`polars.datatypes.Datetime` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Time` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Duration` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Categorical` -> :func:`polars.datatypes.UInt32`
        - Other data types will be left unchanged.

        Examples
        --------
        Replicating the pandas
        `pd.factorize
        <https://pandas.pydata.org/docs/reference/api/pandas.factorize.html>`_
        function.

        >>> pl.DataFrame({"vals": ["a", "x", None, "a"]}).with_columns(
        ...     [
        ...         pl.col("vals").cast(pl.Categorical),
        ...         pl.col("vals")
        ...         .cast(pl.Categorical)
        ...         .to_physical()
        ...         .alias("vals_physical"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌──────┬───────────────┐
        │ vals ┆ vals_physical │
        │ ---  ┆ ---           │
        │ cat  ┆ u32           │
        ╞══════╪═══════════════╡
        │ a    ┆ 0             │
        │ x    ┆ 1             │
        │ null ┆ null          │
        │ a    ┆ 0             │
        └──────┴───────────────┘

        """
    def any(self) -> Self:
        """
        Check if any boolean value in a Boolean column is `True`.

        Returns
        -------
        Boolean literal

        Examples
        --------
        >>> df = pl.DataFrame({"TF": [True, False], "FF": [False, False]})
        >>> df.select(pl.all().any())
        shape: (1, 2)
        ┌──────┬───────┐
        │ TF   ┆ FF    │
        │ ---  ┆ ---   │
        │ bool ┆ bool  │
        ╞══════╪═══════╡
        │ true ┆ false │
        └──────┴───────┘

        """
    def all(self) -> Self:
        """
        Check if all boolean values in a Boolean column are `True`.

        This method is an expression - not to be confused with
        :func:`polars.all` which is a function to select all columns.

        Returns
        -------
        Boolean literal

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"TT": [True, True], "TF": [True, False], "FF": [False, False]}
        ... )
        >>> df.select(pl.col("*").all())
        shape: (1, 3)
        ┌──────┬───────┬───────┐
        │ TT   ┆ TF    ┆ FF    │
        │ ---  ┆ ---   ┆ ---   │
        │ bool ┆ bool  ┆ bool  │
        ╞══════╪═══════╪═══════╡
        │ true ┆ false ┆ false │
        └──────┴───────┴───────┘

        """
    def arg_true(self) -> Self:
        """
        Return indices where expression evaluates `True`.

        .. warning::
            Modifies number of rows returned, so will fail in combination with other
            expressions. Use as only expression in `select` / `with_columns`.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2, 1]})
        >>> df.select((pl.col("a") == 1).arg_true())
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 0   │
        │ 1   │
        │ 3   │
        └─────┘

        See Also
        --------
        Series.arg_true : Return indices where Series is True
        pl.arg_where

        """
    def sqrt(self) -> Self:
        """
        Compute the square root of the elements.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1.0, 2.0, 4.0]})
        >>> df.select(pl.col("values").sqrt())
        shape: (3, 1)
        ┌──────────┐
        │ values   │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.0      │
        │ 1.414214 │
        │ 2.0      │
        └──────────┘

        """
    def log10(self) -> Self:
        """
        Compute the base 10 logarithm of the input array, element-wise.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1.0, 2.0, 4.0]})
        >>> df.select(pl.col("values").log10())
        shape: (3, 1)
        ┌─────────┐
        │ values  │
        │ ---     │
        │ f64     │
        ╞═════════╡
        │ 0.0     │
        │ 0.30103 │
        │ 0.60206 │
        └─────────┘

        """
    def exp(self) -> Self:
        """
        Compute the exponential, element-wise.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1.0, 2.0, 4.0]})
        >>> df.select(pl.col("values").exp())
        shape: (3, 1)
        ┌──────────┐
        │ values   │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 2.718282 │
        │ 7.389056 │
        │ 54.59815 │
        └──────────┘

        """
    def alias(self, name: str) -> Self:
        """
        Rename the output of an expression.

        Parameters
        ----------
        name
            New name.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["a", "b", None],
        ...     }
        ... )
        >>> df
        shape: (3, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ str  │
        ╞═════╪══════╡
        │ 1   ┆ a    │
        │ 2   ┆ b    │
        │ 3   ┆ null │
        └─────┴──────┘
        >>> df.select(
        ...     [
        ...         pl.col("a").alias("bar"),
        ...         pl.col("b").alias("foo"),
        ...     ]
        ... )
        shape: (3, 2)
        ┌─────┬──────┐
        │ bar ┆ foo  │
        │ --- ┆ ---  │
        │ i64 ┆ str  │
        ╞═════╪══════╡
        │ 1   ┆ a    │
        │ 2   ┆ b    │
        │ 3   ┆ null │
        └─────┴──────┘

        """
    def exclude(
        self,
        columns: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
        *more_columns: str | PolarsDataType,
    ) -> Self:
        """
        Exclude columns from a multi-column expression.

        Only works after a wildcard or regex column selection.

        Parameters
        ----------
        columns
            The name or datatype of the column(s) to exclude. Accepts regular expression
            input. Regular expressions should start with ``^`` and end with ``$``.
        *more_columns
            Additional names or datatypes of columns to exclude, specified as positional
            arguments.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "aa": [1, 2, 3],
        ...         "ba": ["a", "b", None],
        ...         "cc": [None, 2.5, 1.5],
        ...     }
        ... )
        >>> df
        shape: (3, 3)
        ┌─────┬──────┬──────┐
        │ aa  ┆ ba   ┆ cc   │
        │ --- ┆ ---  ┆ ---  │
        │ i64 ┆ str  ┆ f64  │
        ╞═════╪══════╪══════╡
        │ 1   ┆ a    ┆ null │
        │ 2   ┆ b    ┆ 2.5  │
        │ 3   ┆ null ┆ 1.5  │
        └─────┴──────┴──────┘

        Exclude by column name(s):

        >>> df.select(pl.all().exclude("ba"))
        shape: (3, 2)
        ┌─────┬──────┐
        │ aa  ┆ cc   │
        │ --- ┆ ---  │
        │ i64 ┆ f64  │
        ╞═════╪══════╡
        │ 1   ┆ null │
        │ 2   ┆ 2.5  │
        │ 3   ┆ 1.5  │
        └─────┴──────┘

        Exclude by regex, e.g. removing all columns whose names end with the letter "a":

        >>> df.select(pl.all().exclude("^.*a$"))
        shape: (3, 1)
        ┌──────┐
        │ cc   │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ 2.5  │
        │ 1.5  │
        └──────┘

        Exclude by dtype(s), e.g. removing all columns of type Int64 or Float64:

        >>> df.select(pl.all().exclude([pl.Int64, pl.Float64]))
        shape: (3, 1)
        ┌──────┐
        │ ba   │
        │ ---  │
        │ str  │
        ╞══════╡
        │ a    │
        │ b    │
        │ null │
        └──────┘

        """
    def keep_name(self) -> Self:
        """
        Keep the original root name of the expression.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2],
        ...         "b": [3, 4],
        ...     }
        ... )

        Keep original column name to undo an alias operation.

        >>> df.with_columns([(pl.col("a") * 9).alias("c").keep_name()])
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 9   ┆ 3   │
        │ 18  ┆ 4   │
        └─────┴─────┘

        Prevent
        "DuplicateError: Column with name: \'literal\' has more than one occurrences"
        errors.

        >>> df.select([(pl.lit(10) / pl.all()).keep_name()])
        shape: (2, 2)
        ┌──────┬──────────┐
        │ a    ┆ b        │
        │ ---  ┆ ---      │
        │ f64  ┆ f64      │
        ╞══════╪══════════╡
        │ 10.0 ┆ 3.333333 │
        │ 5.0  ┆ 2.5      │
        └──────┴──────────┘

        """
    def pipe(
        self, function: Callable[Concatenate[Expr, P], T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """
        Offers a structured way to apply a sequence of user-defined functions (UDFs).

        Parameters
        ----------
        function
            Callable; will receive the expression as the first parameter,
            followed by any given args/kwargs.
        *args
            Arguments to pass to the UDF.
        **kwargs
            Keyword arguments to pass to the UDF.

        Examples
        --------
        >>> def extract_number(expr):
        ...     return expr.str.extract(r"\\d+", 0).cast(pl.Int64)
        ...
        >>> df = pl.DataFrame({"a": ["a: 1", "b: 2", "c: 3"]})
        >>> df.with_columns(pl.col("a").pipe(extract_number))
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        │ 3   │
        └─────┘

        """
    def prefix(self, prefix: str) -> Self:
        """
        Add a prefix to the root column name of the expression.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2, 3, 4, 5],
        ...         "fruits": ["banana", "banana", "apple", "apple", "banana"],
        ...         "B": [5, 4, 3, 2, 1],
        ...         "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        ...     }
        ... )
        >>> df
        shape: (5, 4)
        ┌─────┬────────┬─────┬────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   │
        │ --- ┆ ---    ┆ --- ┆ ---    │
        │ i64 ┆ str    ┆ i64 ┆ str    │
        ╞═════╪════════╪═════╪════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle │
        │ 2   ┆ banana ┆ 4   ┆ audi   │
        │ 3   ┆ apple  ┆ 3   ┆ beetle │
        │ 4   ┆ apple  ┆ 2   ┆ beetle │
        │ 5   ┆ banana ┆ 1   ┆ beetle │
        └─────┴────────┴─────┴────────┘
        >>> df.select(
        ...     [
        ...         pl.all(),
        ...         pl.all().reverse().prefix("reverse_"),
        ...     ]
        ... )
        shape: (5, 8)
        ┌─────┬────────┬─────┬────────┬───────────┬────────────────┬───────────┬──────────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   ┆ reverse_A ┆ reverse_fruits ┆ reverse_B ┆ reverse_cars │
        │ --- ┆ ---    ┆ --- ┆ ---    ┆ ---       ┆ ---            ┆ ---       ┆ ---          │
        │ i64 ┆ str    ┆ i64 ┆ str    ┆ i64       ┆ str            ┆ i64       ┆ str          │
        ╞═════╪════════╪═════╪════════╪═══════════╪════════════════╪═══════════╪══════════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle ┆ 5         ┆ banana         ┆ 1         ┆ beetle       │
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 4         ┆ apple          ┆ 2         ┆ beetle       │
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ 3         ┆ apple          ┆ 3         ┆ beetle       │
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2         ┆ banana         ┆ 4         ┆ audi         │
        │ 5   ┆ banana ┆ 1   ┆ beetle ┆ 1         ┆ banana         ┆ 5         ┆ beetle       │
        └─────┴────────┴─────┴────────┴───────────┴────────────────┴───────────┴──────────────┘

        """
    def suffix(self, suffix: str) -> Self:
        """
        Add a suffix to the root column name of the expression.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2, 3, 4, 5],
        ...         "fruits": ["banana", "banana", "apple", "apple", "banana"],
        ...         "B": [5, 4, 3, 2, 1],
        ...         "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        ...     }
        ... )
        >>> df
        shape: (5, 4)
        ┌─────┬────────┬─────┬────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   │
        │ --- ┆ ---    ┆ --- ┆ ---    │
        │ i64 ┆ str    ┆ i64 ┆ str    │
        ╞═════╪════════╪═════╪════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle │
        │ 2   ┆ banana ┆ 4   ┆ audi   │
        │ 3   ┆ apple  ┆ 3   ┆ beetle │
        │ 4   ┆ apple  ┆ 2   ┆ beetle │
        │ 5   ┆ banana ┆ 1   ┆ beetle │
        └─────┴────────┴─────┴────────┘
        >>> df.select(
        ...     [
        ...         pl.all(),
        ...         pl.all().reverse().suffix("_reverse"),
        ...     ]
        ... )
        shape: (5, 8)
        ┌─────┬────────┬─────┬────────┬───────────┬────────────────┬───────────┬──────────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   ┆ A_reverse ┆ fruits_reverse ┆ B_reverse ┆ cars_reverse │
        │ --- ┆ ---    ┆ --- ┆ ---    ┆ ---       ┆ ---            ┆ ---       ┆ ---          │
        │ i64 ┆ str    ┆ i64 ┆ str    ┆ i64       ┆ str            ┆ i64       ┆ str          │
        ╞═════╪════════╪═════╪════════╪═══════════╪════════════════╪═══════════╪══════════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle ┆ 5         ┆ banana         ┆ 1         ┆ beetle       │
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 4         ┆ apple          ┆ 2         ┆ beetle       │
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ 3         ┆ apple          ┆ 3         ┆ beetle       │
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2         ┆ banana         ┆ 4         ┆ audi         │
        │ 5   ┆ banana ┆ 1   ┆ beetle ┆ 1         ┆ banana         ┆ 5         ┆ beetle       │
        └─────┴────────┴─────┴────────┴───────────┴────────────────┴───────────┴──────────────┘

        """
    def map_alias(self, *args, **kwargs) -> Self:
        """
        Rename the output of an expression by mapping a function over the root name.

        Parameters
        ----------
        function
            Function that maps root name to new name.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2],
        ...         "B": [3, 4],
        ...     }
        ... )
        >>> df.select(
        ...     pl.all().reverse().map_alias(lambda colName: colName + "_reverse")
        ... )
        shape: (2, 2)
        ┌───────────┬───────────┐
        │ A_reverse ┆ B_reverse │
        │ ---       ┆ ---       │
        │ i64       ┆ i64       │
        ╞═══════════╪═══════════╡
        │ 2         ┆ 4         │
        │ 1         ┆ 3         │
        └───────────┴───────────┘

        """
    def is_not(self) -> Self:
        """
        Negate a boolean expression.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [True, False, False],
        ...         "b": ["a", "b", None],
        ...     }
        ... )
        >>> df
        shape: (3, 2)
        ┌───────┬──────┐
        │ a     ┆ b    │
        │ ---   ┆ ---  │
        │ bool  ┆ str  │
        ╞═══════╪══════╡
        │ true  ┆ a    │
        │ false ┆ b    │
        │ false ┆ null │
        └───────┴──────┘
        >>> df.select(pl.col("a").is_not())
        shape: (3, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        │ true  │
        │ true  │
        └───────┘

        """
    def is_null(self) -> Self:
        """
        Returns a boolean Series indicating which values are null.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_columns(pl.all().is_null().suffix("_isnull"))  # nan != null
        shape: (5, 4)
        ┌──────┬─────┬──────────┬──────────┐
        │ a    ┆ b   ┆ a_isnull ┆ b_isnull │
        │ ---  ┆ --- ┆ ---      ┆ ---      │
        │ i64  ┆ f64 ┆ bool     ┆ bool     │
        ╞══════╪═════╪══════════╪══════════╡
        │ 1    ┆ 1.0 ┆ false    ┆ false    │
        │ 2    ┆ 2.0 ┆ false    ┆ false    │
        │ null ┆ NaN ┆ true     ┆ false    │
        │ 1    ┆ 1.0 ┆ false    ┆ false    │
        │ 5    ┆ 5.0 ┆ false    ┆ false    │
        └──────┴─────┴──────────┴──────────┘

        """
    def is_not_null(self) -> Self:
        """
        Returns a boolean Series indicating which values are not null.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_columns(pl.all().is_not_null().suffix("_not_null"))  # nan != null
        shape: (5, 4)
        ┌──────┬─────┬────────────┬────────────┐
        │ a    ┆ b   ┆ a_not_null ┆ b_not_null │
        │ ---  ┆ --- ┆ ---        ┆ ---        │
        │ i64  ┆ f64 ┆ bool       ┆ bool       │
        ╞══════╪═════╪════════════╪════════════╡
        │ 1    ┆ 1.0 ┆ true       ┆ true       │
        │ 2    ┆ 2.0 ┆ true       ┆ true       │
        │ null ┆ NaN ┆ false      ┆ true       │
        │ 1    ┆ 1.0 ┆ true       ┆ true       │
        │ 5    ┆ 5.0 ┆ true       ┆ true       │
        └──────┴─────┴────────────┴────────────┘

        """
    def is_finite(self) -> Self:
        """
        Returns a boolean Series indicating which values are finite.

        Returns
        -------
        out
            Series of type Boolean

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1.0, 2],
        ...         "B": [3.0, float("inf")],
        ...     }
        ... )
        >>> df.select(pl.all().is_finite())
        shape: (2, 2)
        ┌──────┬───────┐
        │ A    ┆ B     │
        │ ---  ┆ ---   │
        │ bool ┆ bool  │
        ╞══════╪═══════╡
        │ true ┆ true  │
        │ true ┆ false │
        └──────┴───────┘

        """
    def is_infinite(self) -> Self:
        """
        Returns a boolean Series indicating which values are infinite.

        Returns
        -------
        out
            Series of type Boolean

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1.0, 2],
        ...         "B": [3.0, float("inf")],
        ...     }
        ... )
        >>> df.select(pl.all().is_infinite())
        shape: (2, 2)
        ┌───────┬───────┐
        │ A     ┆ B     │
        │ ---   ┆ ---   │
        │ bool  ┆ bool  │
        ╞═══════╪═══════╡
        │ false ┆ false │
        │ false ┆ true  │
        └───────┴───────┘

        """
    def is_nan(self) -> Self:
        """
        Returns a boolean Series indicating which values are NaN.

        Notes
        -----
        Floating point ```NaN`` (Not A Number) should not be confused
        with missing data represented as ``Null/None``.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_columns(pl.col(pl.Float64).is_nan().suffix("_isnan"))
        shape: (5, 3)
        ┌──────┬─────┬─────────┐
        │ a    ┆ b   ┆ b_isnan │
        │ ---  ┆ --- ┆ ---     │
        │ i64  ┆ f64 ┆ bool    │
        ╞══════╪═════╪═════════╡
        │ 1    ┆ 1.0 ┆ false   │
        │ 2    ┆ 2.0 ┆ false   │
        │ null ┆ NaN ┆ true    │
        │ 1    ┆ 1.0 ┆ false   │
        │ 5    ┆ 5.0 ┆ false   │
        └──────┴─────┴─────────┘

        """
    def is_not_nan(self) -> Self:
        """
        Returns a boolean Series indicating which values are not NaN.

        Notes
        -----
        Floating point ```NaN`` (Not A Number) should not be confused
        with missing data represented as ``Null/None``.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None, 1, 5],
        ...         "b": [1.0, 2.0, float("nan"), 1.0, 5.0],
        ...     }
        ... )
        >>> df.with_columns(pl.col(pl.Float64).is_not_nan().suffix("_is_not_nan"))
        shape: (5, 3)
        ┌──────┬─────┬──────────────┐
        │ a    ┆ b   ┆ b_is_not_nan │
        │ ---  ┆ --- ┆ ---          │
        │ i64  ┆ f64 ┆ bool         │
        ╞══════╪═════╪══════════════╡
        │ 1    ┆ 1.0 ┆ true         │
        │ 2    ┆ 2.0 ┆ true         │
        │ null ┆ NaN ┆ false        │
        │ 1    ┆ 1.0 ┆ true         │
        │ 5    ┆ 5.0 ┆ true         │
        └──────┴─────┴──────────────┘

        """
    def agg_groups(self) -> Self:
        """
        Get the group indexes of the group by operation.

        Should be used in aggregation context only.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group": [
        ...             "one",
        ...             "one",
        ...             "one",
        ...             "two",
        ...             "two",
        ...             "two",
        ...         ],
        ...         "value": [94, 95, 96, 97, 97, 99],
        ...     }
        ... )
        >>> df.groupby("group", maintain_order=True).agg(pl.col("value").agg_groups())
        shape: (2, 2)
        ┌───────┬───────────┐
        │ group ┆ value     │
        │ ---   ┆ ---       │
        │ str   ┆ list[u32] │
        ╞═══════╪═══════════╡
        │ one   ┆ [0, 1, 2] │
        │ two   ┆ [3, 4, 5] │
        └───────┴───────────┘

        """
    def count(self) -> Self:
        """
        Count the number of values in this expression.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [8, 9, 10], "b": [None, 4, 4]})
        >>> df.select(pl.all().count())  # counts nulls
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u32 ┆ u32 │
        ╞═════╪═════╡
        │ 3   ┆ 3   │
        └─────┴─────┘

        """
    def len(self) -> Self:
        """
        Count the number of values in this expression.

        Alias for :func:`count`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10],
        ...         "b": [None, 4, 4],
        ...     }
        ... )
        >>> df.select(pl.all().len())  # counts nulls
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u32 ┆ u32 │
        ╞═════╪═════╡
        │ 3   ┆ 3   │
        └─────┴─────┘

        """
    def slice(self, offset: int | Expr, length: int | Expr | None = ...) -> Self:
        """
        Get a slice of this expression.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to ``None``, all rows starting at the offset
            will be selected.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10, 11],
        ...         "b": [None, 4, 4, 4],
        ...     }
        ... )
        >>> df.select(pl.all().slice(1, 2))
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 9   ┆ 4   │
        │ 10  ┆ 4   │
        └─────┴─────┘

        """
    def append(self, other: Expr, upcast: bool = ...) -> Self:
        """
        Append expressions.

        This is done by adding the chunks of `other` to this `Series`.

        Parameters
        ----------
        other
            Expression to append.
        upcast
            Cast both `Series` to the same supertype.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10],
        ...         "b": [None, 4, 4],
        ...     }
        ... )
        >>> df.select(pl.all().head(1).append(pl.all().tail(1)))
        shape: (2, 2)
        ┌─────┬──────┐
        │ a   ┆ b    │
        │ --- ┆ ---  │
        │ i64 ┆ i64  │
        ╞═════╪══════╡
        │ 8   ┆ null │
        │ 10  ┆ 4    │
        └─────┴──────┘

        """
    def rechunk(self) -> Self:
        """
        Create a single chunk of memory for this Series.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})

        Create a Series with 3 nulls, append column a then rechunk

        >>> df.select(pl.repeat(None, 3).append(pl.col("a")).rechunk())
        shape: (6, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ i64     │
        ╞═════════╡
        │ null    │
        │ null    │
        │ null    │
        │ 1       │
        │ 1       │
        │ 2       │
        └─────────┘

        """
    def drop_nulls(self) -> Self:
        """
        Drop all null values.

        Warnings
        --------
        Note that null values are not floating point NaN values!
        To drop NaN values, use :func:`drop_nans`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10, 11],
        ...         "b": [None, 4.0, 4.0, float("nan")],
        ...     }
        ... )
        >>> df.select(pl.col("b").drop_nulls())
        shape: (3, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 4.0 │
        │ 4.0 │
        │ NaN │
        └─────┘

        """
    def drop_nans(self) -> Self:
        """
        Drop floating point NaN values.

        Warnings
        --------
        Note that NaN values are not null values!
        To drop null values, use :func:`drop_nulls`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10, 11],
        ...         "b": [None, 4.0, 4.0, float("nan")],
        ...     }
        ... )
        >>> df.select(pl.col("b").drop_nans())
        shape: (3, 1)
        ┌──────┐
        │ b    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ 4.0  │
        │ 4.0  │
        └──────┘

        """
    def cumsum(self, reverse: bool = ...) -> Self:
        """
        Get an array with the cumulative sum computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4]})
        >>> df.select(
        ...     [
        ...         pl.col("a").cumsum(),
        ...         pl.col("a").cumsum(reverse=True).alias("a_reverse"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────┬───────────┐
        │ a   ┆ a_reverse │
        │ --- ┆ ---       │
        │ i64 ┆ i64       │
        ╞═════╪═══════════╡
        │ 1   ┆ 10        │
        │ 3   ┆ 9         │
        │ 6   ┆ 7         │
        │ 10  ┆ 4         │
        └─────┴───────────┘

        Null values are excluded, but can also be filled by calling ``forward_fill``.

        >>> df = pl.DataFrame({"values": [None, 10, None, 8, 9, None, 16, None]})
        >>> df.with_columns(
        ...     [
        ...         pl.col("values").cumsum().alias("value_cumsum"),
        ...         pl.col("values")
        ...         .cumsum()
        ...         .forward_fill()
        ...         .alias("value_cumsum_all_filled"),
        ...     ]
        ... )
        shape: (8, 3)
        ┌────────┬──────────────┬─────────────────────────┐
        │ values ┆ value_cumsum ┆ value_cumsum_all_filled │
        │ ---    ┆ ---          ┆ ---                     │
        │ i64    ┆ i64          ┆ i64                     │
        ╞════════╪══════════════╪═════════════════════════╡
        │ null   ┆ null         ┆ null                    │
        │ 10     ┆ 10           ┆ 10                      │
        │ null   ┆ null         ┆ 10                      │
        │ 8      ┆ 18           ┆ 18                      │
        │ 9      ┆ 27           ┆ 27                      │
        │ null   ┆ null         ┆ 27                      │
        │ 16     ┆ 43           ┆ 43                      │
        │ null   ┆ null         ┆ 43                      │
        └────────┴──────────────┴─────────────────────────┘

        """
    def cumprod(self, reverse: bool = ...) -> Self:
        """
        Get an array with the cumulative product computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4]})
        >>> df.select(
        ...     [
        ...         pl.col("a").cumprod(),
        ...         pl.col("a").cumprod(reverse=True).alias("a_reverse"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────┬───────────┐
        │ a   ┆ a_reverse │
        │ --- ┆ ---       │
        │ i64 ┆ i64       │
        ╞═════╪═══════════╡
        │ 1   ┆ 24        │
        │ 2   ┆ 24        │
        │ 6   ┆ 12        │
        │ 24  ┆ 4         │
        └─────┴───────────┘

        """
    def cummin(self, reverse: bool = ...) -> Self:
        """
        Get an array with the cumulative min computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4]})
        >>> df.select(
        ...     [
        ...         pl.col("a").cummin(),
        ...         pl.col("a").cummin(reverse=True).alias("a_reverse"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────┬───────────┐
        │ a   ┆ a_reverse │
        │ --- ┆ ---       │
        │ i64 ┆ i64       │
        ╞═════╪═══════════╡
        │ 1   ┆ 1         │
        │ 1   ┆ 2         │
        │ 1   ┆ 3         │
        │ 1   ┆ 4         │
        └─────┴───────────┘

        """
    def cummax(self, reverse: bool = ...) -> Self:
        """
        Get an array with the cumulative max computed at every element.

        Parameters
        ----------
        reverse
            Reverse the operation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4]})
        >>> df.select(
        ...     [
        ...         pl.col("a").cummax(),
        ...         pl.col("a").cummax(reverse=True).alias("a_reverse"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────┬───────────┐
        │ a   ┆ a_reverse │
        │ --- ┆ ---       │
        │ i64 ┆ i64       │
        ╞═════╪═══════════╡
        │ 1   ┆ 4         │
        │ 2   ┆ 4         │
        │ 3   ┆ 4         │
        │ 4   ┆ 4         │
        └─────┴───────────┘

        Null values are excluded, but can also be filled by calling ``forward_fill``.

        >>> df = pl.DataFrame({"values": [None, 10, None, 8, 9, None, 16, None]})
        >>> df.with_columns(
        ...     [
        ...         pl.col("values").cummax().alias("value_cummax"),
        ...         pl.col("values")
        ...         .cummax()
        ...         .forward_fill()
        ...         .alias("value_cummax_all_filled"),
        ...     ]
        ... )
        shape: (8, 3)
        ┌────────┬──────────────┬─────────────────────────┐
        │ values ┆ value_cummax ┆ value_cummax_all_filled │
        │ ---    ┆ ---          ┆ ---                     │
        │ i64    ┆ i64          ┆ i64                     │
        ╞════════╪══════════════╪═════════════════════════╡
        │ null   ┆ null         ┆ null                    │
        │ 10     ┆ 10           ┆ 10                      │
        │ null   ┆ null         ┆ 10                      │
        │ 8      ┆ 10           ┆ 10                      │
        │ 9      ┆ 10           ┆ 10                      │
        │ null   ┆ null         ┆ 10                      │
        │ 16     ┆ 16           ┆ 16                      │
        │ null   ┆ null         ┆ 16                      │
        └────────┴──────────────┴─────────────────────────┘

        """
    def cumcount(self, reverse: bool = ...) -> Self:
        """
        Get an array with the cumulative count computed at every element.

        Counting from 0 to len

        Parameters
        ----------
        reverse
            Reverse the operation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 4]})
        >>> df.select(
        ...     [
        ...         pl.col("a").cumcount(),
        ...         pl.col("a").cumcount(reverse=True).alias("a_reverse"),
        ...     ]
        ... )
        shape: (4, 2)
        ┌─────┬───────────┐
        │ a   ┆ a_reverse │
        │ --- ┆ ---       │
        │ u32 ┆ u32       │
        ╞═════╪═══════════╡
        │ 0   ┆ 3         │
        │ 1   ┆ 2         │
        │ 2   ┆ 1         │
        │ 3   ┆ 0         │
        └─────┴───────────┘

        """
    def floor(self) -> Self:
        """
        Rounds down to the nearest integer value.

        Only works on floating point Series.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.3, 0.5, 1.0, 1.1]})
        >>> df.select(pl.col("a").floor())
        shape: (4, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.0 │
        │ 0.0 │
        │ 1.0 │
        │ 1.0 │
        └─────┘

        """
    def ceil(self) -> Self:
        """
        Rounds up to the nearest integer value.

        Only works on floating point Series.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.3, 0.5, 1.0, 1.1]})
        >>> df.select(pl.col("a").ceil())
        shape: (4, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        │ 1.0 │
        │ 1.0 │
        │ 2.0 │
        └─────┘

        """
    def round(self, decimals: int) -> Self:
        """
        Round underlying floating point data by `decimals` digits.

        Parameters
        ----------
        decimals
            Number of decimals to round by.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.33, 0.52, 1.02, 1.17]})
        >>> df.select(pl.col("a").round(1))
        shape: (4, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.3 │
        │ 0.5 │
        │ 1.0 │
        │ 1.2 │
        └─────┘

        """
    def dot(self, other: Expr | str) -> Self:
        """
        Compute the dot/inner product between two Expressions.

        Parameters
        ----------
        other
            Expression to compute dot product with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 3, 5],
        ...         "b": [2, 4, 6],
        ...     }
        ... )
        >>> df.select(pl.col("a").dot(pl.col("b")))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 44  │
        └─────┘

        """
    def mode(self) -> Self:
        """
        Compute the most occurring value(s).

        Can return multiple Values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 1, 2, 3],
        ...         "b": [1, 1, 2, 2],
        ...     }
        ... )
        >>> df.select(pl.all().mode())  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 1   │
        │ 1   ┆ 2   │
        └─────┴─────┘

        """
    def cast(self, dtype: PolarsDataType | type[Any], strict: bool = ...) -> Self:
        """
        Cast between data types.

        Parameters
        ----------
        dtype
            DataType to cast to.
        strict
            Throw an error if a cast could not be done.
            For instance, due to an overflow.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["4", "5", "6"],
        ...     }
        ... )
        >>> df.with_columns(
        ...     [
        ...         pl.col("a").cast(pl.Float64),
        ...         pl.col("b").cast(pl.Int32),
        ...     ]
        ... )
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ i32 │
        ╞═════╪═════╡
        │ 1.0 ┆ 4   │
        │ 2.0 ┆ 5   │
        │ 3.0 ┆ 6   │
        └─────┴─────┘

        """
    def sort(self) -> Self:
        """
        Sort this column.

        When used in a projection/selection context, the whole column is sorted.
        When used in a groupby context, the groups are sorted.

        Parameters
        ----------
        descending
            Sort in descending order.
        nulls_last
            Place null values last.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, None, 3, 2],
        ...     }
        ... )
        >>> df.select(pl.col("a").sort())
        shape: (4, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ null │
        │ 1    │
        │ 2    │
        │ 3    │
        └──────┘
        >>> df.select(pl.col("a").sort(descending=True))
        shape: (4, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ null │
        │ 3    │
        │ 2    │
        │ 1    │
        └──────┘
        >>> df.select(pl.col("a").sort(nulls_last=True))
        shape: (4, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 1    │
        │ 2    │
        │ 3    │
        │ null │
        └──────┘

        When sorting in a groupby context, the groups are sorted.

        >>> df = pl.DataFrame(
        ...     {
        ...         "group": ["one", "one", "one", "two", "two", "two"],
        ...         "value": [1, 98, 2, 3, 99, 4],
        ...     }
        ... )
        >>> df.groupby("group").agg(pl.col("value").sort())  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌───────┬────────────┐
        │ group ┆ value      │
        │ ---   ┆ ---        │
        │ str   ┆ list[i64]  │
        ╞═══════╪════════════╡
        │ two   ┆ [3, 4, 99] │
        │ one   ┆ [1, 2, 98] │
        └───────┴────────────┘

        """
    def top_k(self, *args, **kwargs) -> Self:
        """
        Return the `k` largest elements.

        If \'descending=True` the smallest elements will be given.

        This has time complexity:

        .. math:: O(n + k \\\\log{}n - \\frac{k}{2})

        Parameters
        ----------
        k
            Number of elements to return.
        descending
            Return the smallest elements.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "value": [1, 98, 2, 3, 99, 4],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("value").top_k().alias("top_k"),
        ...         pl.col("value").top_k(descending=True).alias("bottom_k"),
        ...     ]
        ... )
        shape: (5, 2)
        ┌───────┬──────────┐
        │ top_k ┆ bottom_k │
        │ ---   ┆ ---      │
        │ i64   ┆ i64      │
        ╞═══════╪══════════╡
        │ 99    ┆ 1        │
        │ 98    ┆ 2        │
        │ 4     ┆ 3        │
        │ 3     ┆ 4        │
        │ 2     ┆ 98       │
        └───────┴──────────┘

        """
    def arg_sort(self) -> Self:
        """
        Get the index values that would sort this column.

        Parameters
        ----------
        descending
            Sort in descending (descending) order.
        nulls_last
            Place null values last instead of first.

        Returns
        -------
        Expr
            Series of dtype UInt32.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [20, 10, 30],
        ...     }
        ... )
        >>> df.select(pl.col("a").arg_sort())
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 1   │
        │ 0   │
        │ 2   │
        └─────┘

        """
    def arg_max(self) -> Self:
        """
        Get the index of the maximal value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [20, 10, 30],
        ...     }
        ... )
        >>> df.select(pl.col("a").arg_max())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        └─────┘

        """
    def arg_min(self) -> Self:
        """
        Get the index of the minimal value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [20, 10, 30],
        ...     }
        ... )
        >>> df.select(pl.col("a").arg_min())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 1   │
        └─────┘

        """
    def search_sorted(
        self, element: Expr | int | float | Series, side: SearchSortedSide = ...
    ) -> Self:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "values": [1, 2, 3, 5],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("values").search_sorted(0).alias("zero"),
        ...         pl.col("values").search_sorted(3).alias("three"),
        ...         pl.col("values").search_sorted(6).alias("six"),
        ...     ]
        ... )
        shape: (1, 3)
        ┌──────┬───────┬─────┐
        │ zero ┆ three ┆ six │
        │ ---  ┆ ---   ┆ --- │
        │ u32  ┆ u32   ┆ u32 │
        ╞══════╪═══════╪═════╡
        │ 0    ┆ 2     ┆ 4   │
        └──────┴───────┴─────┘

        """
    def sort_by(self, *args, **kwargs) -> Self:
        """
        Sort this column by the ordering of other columns.

        When used in a projection/selection context, the whole column is sorted.
        When used in a groupby context, the groups are sorted.

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

        Examples
        --------
        Pass a single column name to sort by that column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "group": ["a", "a", "b", "b"],
        ...         "value1": [1, 3, 4, 2],
        ...         "value2": [8, 7, 6, 5],
        ...     }
        ... )
        >>> df.select(pl.col("group").sort_by("value1"))
        shape: (4, 1)
        ┌───────┐
        │ group │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ a     │
        │ b     │
        │ a     │
        │ b     │
        └───────┘

        Sorting by expressions is also supported.

        >>> df.select(pl.col("group").sort_by(pl.col("value1") + pl.col("value2")))
        shape: (4, 1)
        ┌───────┐
        │ group │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ b     │
        │ a     │
        │ a     │
        │ b     │
        └───────┘

        Sort by multiple columns by passing a list of columns.

        >>> df.select(pl.col("group").sort_by(["value1", "value2"], descending=True))
        shape: (4, 1)
        ┌───────┐
        │ group │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ b     │
        │ a     │
        │ b     │
        │ a     │
        └───────┘

        Or use positional arguments to sort by multiple columns in the same way.

        >>> df.select(pl.col("group").sort_by("value1", "value2"))
        shape: (4, 1)
        ┌───────┐
        │ group │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ a     │
        │ b     │
        │ a     │
        │ b     │
        └───────┘

        When sorting in a groupby context, the groups are sorted.

        >>> df.groupby("group").agg(
        ...     pl.col("value1").sort_by("value2")
        ... )  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌───────┬───────────┐
        │ group ┆ value1    │
        │ ---   ┆ ---       │
        │ str   ┆ list[i64] │
        ╞═══════╪═══════════╡
        │ a     ┆ [3, 1]    │
        │ b     ┆ [2, 4]    │
        └───────┴───────────┘

        Take a single row from each group where a column attains its minimal value
        within that group.

        >>> df.groupby("group").agg(
        ...     pl.all().sort_by("value2").first()
        ... )  # doctest: +IGNORE_RESULT
        shape: (2, 3)
        ┌───────┬────────┬────────┐
        │ group ┆ value1 ┆ value2 |
        │ ---   ┆ ---    ┆ ---    │
        │ str   ┆ i64    ┆ i64    |
        ╞═══════╪════════╪════════╡
        │ a     ┆ 3      ┆ 7      |
        │ b     ┆ 2      ┆ 5      |
        └───────┴────────┴────────┘

        """
    def take(self, indices: int | list[int] | Expr | Series | np.ndarray[Any, Any]) -> Self:
        """
        Take values by index.

        Parameters
        ----------
        indices
            An expression that leads to a UInt32 dtyped Series.

        Returns
        -------
        Values taken by index

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group": [
        ...             "one",
        ...             "one",
        ...             "one",
        ...             "two",
        ...             "two",
        ...             "two",
        ...         ],
        ...         "value": [1, 98, 2, 3, 99, 4],
        ...     }
        ... )
        >>> df.groupby("group", maintain_order=True).agg(pl.col("value").take(1))
        shape: (2, 2)
        ┌───────┬───────┐
        │ group ┆ value │
        │ ---   ┆ ---   │
        │ str   ┆ i64   │
        ╞═══════╪═══════╡
        │ one   ┆ 98    │
        │ two   ┆ 99    │
        └───────┴───────┘

        """
    def shift(self, periods: int = ...) -> Self:
        """
        Shift the values by a given period.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4]})
        >>> df.select(pl.col("foo").shift(1))
        shape: (4, 1)
        ┌──────┐
        │ foo  │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ null │
        │ 1    │
        │ 2    │
        │ 3    │
        └──────┘

        """
    def shift_and_fill(
        self, periods: int, fill_value: int | float | bool | str | Expr | list[Any]
    ) -> Self:
        """
        Shift the values by a given period and fill the resulting null values.

        Parameters
        ----------
        periods
            Number of places to shift (may be negative).
        fill_value
            Fill None values with the result of this expression.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4]})
        >>> df.select(pl.col("foo").shift_and_fill(1, "a"))
        shape: (4, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ str │
        ╞═════╡
        │ a   │
        │ 1   │
        │ 2   │
        │ 3   │
        └─────┘

        """
    def fill_null(
        self,
        value: Any | None = ...,
        strategy: FillNullStrategy | None = ...,
        limit: int | None = ...,
    ) -> Self:
        """
        Fill null values using the specified value or strategy.

        To interpolate over null values see interpolate.

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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": [4, None, 6],
        ...     }
        ... )
        >>> df.fill_null(strategy="zero")
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 0   │
        │ 0   ┆ 6   │
        └─────┴─────┘
        >>> df.fill_null(99)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 99  │
        │ 99  ┆ 6   │
        └─────┴─────┘
        >>> df.fill_null(strategy="forward")
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 4   │
        │ 2   ┆ 6   │
        └─────┴─────┘

        """
    def fill_nan(self, fill_value: int | float | Expr | None) -> Self:
        """
        Fill floating point NaN value with a fill value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1.0, None, float("nan")],
        ...         "b": [4.0, float("nan"), 6],
        ...     }
        ... )
        >>> df.fill_nan("zero")
        shape: (3, 2)
        ┌──────┬──────┐
        │ a    ┆ b    │
        │ ---  ┆ ---  │
        │ str  ┆ str  │
        ╞══════╪══════╡
        │ 1.0  ┆ 4.0  │
        │ null ┆ zero │
        │ zero ┆ 6.0  │
        └──────┴──────┘

        """
    def forward_fill(self, limit: int | None = ...) -> Self:
        """
        Fill missing values with the latest seen values.

        Parameters
        ----------
        limit
            The number of consecutive null values to forward fill.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": [4, None, 6],
        ...     }
        ... )
        >>> df.select(pl.all().forward_fill())
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 4   │
        │ 2   ┆ 6   │
        └─────┴─────┘

        """
    def backward_fill(self, limit: int | None = ...) -> Self:
        """
        Fill missing values with the next to be seen values.

        Parameters
        ----------
        limit
            The number of consecutive null values to backward fill.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": [4, None, 6],
        ...     }
        ... )
        >>> df.select(pl.all().backward_fill())
        shape: (3, 2)
        ┌──────┬─────┐
        │ a    ┆ b   │
        │ ---  ┆ --- │
        │ i64  ┆ i64 │
        ╞══════╪═════╡
        │ 1    ┆ 4   │
        │ 2    ┆ 6   │
        │ null ┆ 6   │
        └──────┴─────┘

        """
    def reverse(self) -> Self:
        """
        Reverse the selection.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1, 2, 3, 4, 5],
        ...         "fruits": ["banana", "banana", "apple", "apple", "banana"],
        ...         "B": [5, 4, 3, 2, 1],
        ...         "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.all(),
        ...         pl.all().reverse().suffix("_reverse"),
        ...     ]
        ... )
        shape: (5, 8)
        ┌─────┬────────┬─────┬────────┬───────────┬────────────────┬───────────┬──────────────┐
        │ A   ┆ fruits ┆ B   ┆ cars   ┆ A_reverse ┆ fruits_reverse ┆ B_reverse ┆ cars_reverse │
        │ --- ┆ ---    ┆ --- ┆ ---    ┆ ---       ┆ ---            ┆ ---       ┆ ---          │
        │ i64 ┆ str    ┆ i64 ┆ str    ┆ i64       ┆ str            ┆ i64       ┆ str          │
        ╞═════╪════════╪═════╪════════╪═══════════╪════════════════╪═══════════╪══════════════╡
        │ 1   ┆ banana ┆ 5   ┆ beetle ┆ 5         ┆ banana         ┆ 1         ┆ beetle       │
        │ 2   ┆ banana ┆ 4   ┆ audi   ┆ 4         ┆ apple          ┆ 2         ┆ beetle       │
        │ 3   ┆ apple  ┆ 3   ┆ beetle ┆ 3         ┆ apple          ┆ 3         ┆ beetle       │
        │ 4   ┆ apple  ┆ 2   ┆ beetle ┆ 2         ┆ banana         ┆ 4         ┆ audi         │
        │ 5   ┆ banana ┆ 1   ┆ beetle ┆ 1         ┆ banana         ┆ 5         ┆ beetle       │
        └─────┴────────┴─────┴────────┴───────────┴────────────────┴───────────┴──────────────┘

        """
    def std(self, ddof: int = ...) -> Self:
        """
        Get standard deviation.

        Parameters
        ----------
        ddof
            Degrees of freedom.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").std())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘

        """
    def var(self, ddof: int = ...) -> Self:
        """
        Get variance.

        Parameters
        ----------
        ddof
            Degrees of freedom.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").var())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘

        """
    def max(self) -> Self:
        """
        Get maximum value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, float("nan"), 1]})
        >>> df.select(pl.col("a").max())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘

        """
    def min(self) -> Self:
        """
        Get minimum value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, float("nan"), 1]})
        >>> df.select(pl.col("a").min())
        shape: (1, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ -1.0 │
        └──────┘

        """
    def nan_max(self) -> Self:
        """
        Get maximum value, but propagate/poison encountered NaN values.

        This differs from numpy\'s `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0, float("nan")]})
        >>> df.select(pl.col("a").nan_max())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ NaN │
        └─────┘

        """
    def nan_min(self) -> Self:
        """
        Get minimum value, but propagate/poison encountered NaN values.

        This differs from numpy\'s `nanmax` as numpy defaults to propagating NaN values,
        whereas polars defaults to ignoring them.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0, float("nan")]})
        >>> df.select(pl.col("a").nan_min())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ NaN │
        └─────┘

        """
    def sum(self) -> Self:
        """
        Get sum value.

        Notes
        -----
        Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
        Int64 before summing to prevent overflow issues.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").sum())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │  0  │
        └─────┘

        """
    def mean(self) -> Self:
        """
        Get mean value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").mean())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.0 │
        └─────┘

        """
    def median(self) -> Self:
        """
        Get median value using linear interpolation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-1, 0, 1]})
        >>> df.select(pl.col("a").median())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.0 │
        └─────┘

        """
    def product(self) -> Self:
        """
        Compute the product of an expression.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").product())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 6   │
        └─────┘

        """
    def n_unique(self) -> Self:
        """
        Count unique values.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").n_unique())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        └─────┘

        """
    def null_count(self) -> Self:
        """
        Count null values.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [None, 1, None],
        ...         "b": [1, 2, 3],
        ...     }
        ... )
        >>> df.select(pl.all().null_count())
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u32 ┆ u32 │
        ╞═════╪═════╡
        │ 2   ┆ 0   │
        └─────┴─────┘

        """
    def arg_unique(self) -> Self:
        """
        Get index of first unique value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [8, 9, 10],
        ...         "b": [None, 4, 4],
        ...     }
        ... )
        >>> df.select(pl.col("a").arg_unique())
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 0   │
        │ 1   │
        │ 2   │
        └─────┘
        >>> df.select(pl.col("b").arg_unique())
        shape: (2, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 0   │
        │ 1   │
        └─────┘

        """
    def unique(self, maintain_order: bool = ...) -> Self:
        """
        Get unique values of this expression.

        Parameters
        ----------
        maintain_order
            Maintain order of data. This requires more work.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").unique())  # doctest: +IGNORE_RESULT
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        │ 1   │
        └─────┘
        >>> df.select(pl.col("a").unique(maintain_order=True))
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        └─────┘

        """
    def first(self) -> Self:
        """
        Get the first value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").first())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        └─────┘

        """
    def last(self) -> Self:
        """
        Get the last value.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").last())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        └─────┘

        """
    def over(self, expr: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Self:
        """
        Compute expressions over the given groups.

        This expression is similar to performing a groupby aggregation and joining the
        result back into the original dataframe.

        The outcome is similar to how `window functions
        <https://www.postgresql.org/docs/current/tutorial-window.html>`_
        work in PostgreSQL.

        Parameters
        ----------
        expr
            Column(s) to group by. Accepts expression input. Strings are parsed as
            column names.
        *more_exprs
            Additional columns to group by, specified as positional arguments.

        Examples
        --------
        Pass the name of a column to compute the expression over that column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["a", "a", "b", "b", "b"],
        ...         "b": [1, 2, 3, 5, 3],
        ...         "c": [5, 4, 3, 2, 1],
        ...     }
        ... )
        >>> df.with_columns(pl.col("c").max().over("a").suffix("_max"))
        shape: (5, 4)
        ┌─────┬─────┬─────┬───────┐
        │ a   ┆ b   ┆ c   ┆ c_max │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ str ┆ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═════╪═══════╡
        │ a   ┆ 1   ┆ 5   ┆ 5     │
        │ a   ┆ 2   ┆ 4   ┆ 5     │
        │ b   ┆ 3   ┆ 3   ┆ 3     │
        │ b   ┆ 5   ┆ 2   ┆ 3     │
        │ b   ┆ 3   ┆ 1   ┆ 3     │
        └─────┴─────┴─────┴───────┘

        Expression input is supported.

        >>> df.with_columns(pl.col("c").max().over(pl.col("b") // 2).suffix("_max"))
        shape: (5, 4)
        ┌─────┬─────┬─────┬───────┐
        │ a   ┆ b   ┆ c   ┆ c_max │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ str ┆ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═════╪═══════╡
        │ a   ┆ 1   ┆ 5   ┆ 5     │
        │ a   ┆ 2   ┆ 4   ┆ 4     │
        │ b   ┆ 3   ┆ 3   ┆ 4     │
        │ b   ┆ 5   ┆ 2   ┆ 2     │
        │ b   ┆ 3   ┆ 1   ┆ 4     │
        └─────┴─────┴─────┴───────┘

        Group by multiple columns by passing a list of column names or expressions.

        >>> df.with_columns(pl.col("c").min().over(["a", "b"]).suffix("_min"))
        shape: (5, 4)
        ┌─────┬─────┬─────┬───────┐
        │ a   ┆ b   ┆ c   ┆ c_min │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ str ┆ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═════╪═══════╡
        │ a   ┆ 1   ┆ 5   ┆ 5     │
        │ a   ┆ 2   ┆ 4   ┆ 4     │
        │ b   ┆ 3   ┆ 3   ┆ 1     │
        │ b   ┆ 5   ┆ 2   ┆ 2     │
        │ b   ┆ 3   ┆ 1   ┆ 1     │
        └─────┴─────┴─────┴───────┘

        Or use positional arguments to group by multiple columns in the same way.

        >>> df.with_columns(pl.col("c").min().over("a", pl.col("b") % 2).suffix("_min"))
        shape: (5, 4)
        ┌─────┬─────┬─────┬───────┐
        │ a   ┆ b   ┆ c   ┆ c_min │
        │ --- ┆ --- ┆ --- ┆ ---   │
        │ str ┆ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═════╪═══════╡
        │ a   ┆ 1   ┆ 5   ┆ 5     │
        │ a   ┆ 2   ┆ 4   ┆ 4     │
        │ b   ┆ 3   ┆ 3   ┆ 1     │
        │ b   ┆ 5   ┆ 2   ┆ 1     │
        │ b   ┆ 3   ┆ 1   ┆ 1     │
        └─────┴─────┴─────┴───────┘

        """
    def is_unique(self) -> Self:
        """
        Get mask of unique values.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").is_unique())
        shape: (3, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        │ false │
        │ true  │
        └───────┘

        """
    def is_first(self) -> Self:
        """
        Get a mask of the first unique value.

        Returns
        -------
        Boolean Series

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "num": [1, 2, 3, 1, 5],
        ...     }
        ... )
        >>> df.with_columns(pl.col("num").is_first().alias("is_first"))
        shape: (5, 2)
        ┌─────┬──────────┐
        │ num ┆ is_first │
        │ --- ┆ ---      │
        │ i64 ┆ bool     │
        ╞═════╪══════════╡
        │ 1   ┆ true     │
        │ 2   ┆ true     │
        │ 3   ┆ true     │
        │ 1   ┆ false    │
        │ 5   ┆ true     │
        └─────┴──────────┘

        """
    def is_duplicated(self) -> Self:
        """
        Get mask of duplicated values.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").is_duplicated())
        shape: (3, 1)
        ┌───────┐
        │ a     │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        │ true  │
        │ false │
        └───────┘

        """
    def quantile(
        self, quantile: float | Expr, interpolation: RollingInterpolationMethod = ...
    ) -> Self:
        """
        Get quantile value.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {\'nearest\', \'higher\', \'lower\', \'midpoint\', \'linear\'}
            Interpolation method.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0, 1, 2, 3, 4, 5]})
        >>> df.select(pl.col("a").quantile(0.3))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘
        >>> df.select(pl.col("a").quantile(0.3, interpolation="higher"))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 2.0 │
        └─────┘
        >>> df.select(pl.col("a").quantile(0.3, interpolation="lower"))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘
        >>> df.select(pl.col("a").quantile(0.3, interpolation="midpoint"))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.5 │
        └─────┘
        >>> df.select(pl.col("a").quantile(0.3, interpolation="linear"))
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.5 │
        └─────┘

        """
    def filter(self, predicate: Expr) -> Self:
        """
        Filter a single column.

        Mostly useful in an aggregation context. If you want to filter on a DataFrame
        level, use `LazyFrame.filter`.

        Parameters
        ----------
        predicate
            Boolean expression.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group_col": ["g1", "g1", "g2"],
        ...         "b": [1, 2, 3],
        ...     }
        ... )
        >>> df.groupby("group_col").agg(
        ...     [
        ...         pl.col("b").filter(pl.col("b") < 2).sum().alias("lt"),
        ...         pl.col("b").filter(pl.col("b") >= 2).sum().alias("gte"),
        ...     ]
        ... ).sort("group_col")
        shape: (2, 3)
        ┌───────────┬──────┬─────┐
        │ group_col ┆ lt   ┆ gte │
        │ ---       ┆ ---  ┆ --- │
        │ str       ┆ i64  ┆ i64 │
        ╞═══════════╪══════╪═════╡
        │ g1        ┆ 1    ┆ 2   │
        │ g2        ┆ null ┆ 3   │
        └───────────┴──────┴─────┘

        """
    def where(self, predicate: Expr) -> Self:
        """
        Filter a single column.

        Alias for :func:`filter`.

        Parameters
        ----------
        predicate
            Boolean expression.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group_col": ["g1", "g1", "g2"],
        ...         "b": [1, 2, 3],
        ...     }
        ... )
        >>> df.groupby("group_col").agg(
        ...     [
        ...         pl.col("b").where(pl.col("b") < 2).sum().alias("lt"),
        ...         pl.col("b").where(pl.col("b") >= 2).sum().alias("gte"),
        ...     ]
        ... ).sort("group_col")
        shape: (2, 3)
        ┌───────────┬──────┬─────┐
        │ group_col ┆ lt   ┆ gte │
        │ ---       ┆ ---  ┆ --- │
        │ str       ┆ i64  ┆ i64 │
        ╞═══════════╪══════╪═════╡
        │ g1        ┆ 1    ┆ 2   │
        │ g2        ┆ null ┆ 3   │
        └───────────┴──────┴─────┘

        """
    def map(
        self, function: Callable[[Series], Series | Any], return_dtype: PolarsDataType | None = ...
    ) -> Self:
        """
        Apply a custom python function to a Series or sequence of Series.

        The output of this custom function must be a Series.
        If you want to apply a custom function elementwise over single values, see
        :func:`apply`. A use case for ``map`` is when you want to transform an
        expression with a third-party library.

        Read more in `the book
        <https://pola-rs.github.io/polars-book/user-guide/dsl/custom_functions.html>`_.

        Parameters
        ----------
        function
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
        agg_list
            Aggregate list

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "sine": [0.0, 1.0, 0.0, -1.0],
        ...         "cosine": [1.0, 0.0, -1.0, 0.0],
        ...     }
        ... )
        >>> df.select(pl.all().map(lambda x: x.to_numpy().argmax()))
        shape: (1, 2)
        ┌──────┬────────┐
        │ sine ┆ cosine │
        │ ---  ┆ ---    │
        │ i64  ┆ i64    │
        ╞══════╪════════╡
        │ 1    ┆ 0      │
        └──────┴────────┘

        """
    def apply(
        self,
        function: Callable[[Series], Series] | Callable[[Any], Any],
        return_dtype: PolarsDataType | None = ...,
    ) -> Self:
        """
        Apply a custom/user-defined function (UDF) in a GroupBy or Projection context.

        Depending on the context it has the following behavior:

        * Selection
            Expects `f` to be of type Callable[[Any], Any].
            Applies a python function over each individual value in the column.
        * GroupBy
            Expects `f` to be of type Callable[[Series], Series].
            Applies a python function over each group.

        Notes
        -----
        * Using ``apply`` is strongly discouraged as you will be effectively running
          python "for" loops. This will be very slow. Wherever possible you should
          strongly prefer the native expression API to achieve the best performance.

        * If your function is expensive and you don\'t want it to be called more than
          once for a given input, consider applying an ``@lru_cache`` decorator to it.
          With suitable data you may achieve order-of-magnitude speedups (or more).

        Parameters
        ----------
        function
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
            If not set, polars will assume that
            the dtype remains unchanged.
        skip_nulls
            Don\'t apply the function over values
            that contain nulls. This is faster.
        pass_name
            Pass the Series name to the custom function
            This is more expensive.
        strategy : {\'thread_local\', \'threading\'}
            This functionality is in `alpha` stage. This may be removed
            /changed without it being considdered a breaking change.

            - \'thread_local\': run the python function on a single thread.
            - \'threading\': run the python function on separate threads. Use with
                        care as this can slow performance. This might only speed up
                        your code if the amount of work per element is significant
                        and the python function releases the GIL (e.g. via calling
                        a c function)

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 1],
        ...         "b": ["a", "b", "c", "c"],
        ...     }
        ... )

        In a selection context, the function is applied by row.

        >>> df.with_columns(
        ...     pl.col("a").apply(lambda x: x * 2).alias("a_times_2"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬───────────┐
        │ a   ┆ b   ┆ a_times_2 │
        │ --- ┆ --- ┆ ---       │
        │ i64 ┆ str ┆ i64       │
        ╞═════╪═════╪═══════════╡
        │ 1   ┆ a   ┆ 2         │
        │ 2   ┆ b   ┆ 4         │
        │ 3   ┆ c   ┆ 6         │
        │ 1   ┆ c   ┆ 2         │
        └─────┴─────┴───────────┘

        It is better to implement this with an expression:

        >>> df.with_columns(
        ...     (pl.col("a") * 2).alias("a_times_2"),
        ... )  # doctest: +IGNORE_RESULT

        In a GroupBy context the function is applied by group:

        >>> df.lazy().groupby("b", maintain_order=True).agg(
        ...     pl.col("a").apply(lambda x: x.sum())
        ... ).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ b   ┆ a   │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ a   ┆ 1   │
        │ b   ┆ 2   │
        │ c   ┆ 4   │
        └─────┴─────┘

        It is better to implement this with an expression:

        >>> df.groupby("b", maintain_order=True).agg(
        ...     pl.col("a").sum(),
        ... )  # doctest: +IGNORE_RESULT

        """
    def flatten(self) -> Self:
        """
        Flatten a list or string column.

        Alias for :func:`polars.expr.list.ExprListNameSpace.explode`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group": ["a", "b", "b"],
        ...         "values": [[1, 2], [2, 3], [4]],
        ...     }
        ... )
        >>> df.groupby("group").agg(pl.col("values").flatten())  # doctest: +SKIP
        shape: (2, 2)
        ┌───────┬───────────┐
        │ group ┆ values    │
        │ ---   ┆ ---       │
        │ str   ┆ list[i64] │
        ╞═══════╪═══════════╡
        │ a     ┆ [1, 2]    │
        │ b     ┆ [2, 3, 4] │
        └───────┴───────────┘

        """
    def explode(self) -> Self:
        """
        Explode a list or utf8 Series.

        This means that every item is expanded to a new row.

        .. deprecated:: 0.15.16
            `Expr.explode` will be removed in favour of `Expr.arr.explode` and
            `Expr.str.explode`.

        Returns
        -------
        Exploded Series of same dtype

        See Also
        --------
        ExprListNameSpace.explode : Explode a list column.
        ExprStringNameSpace.explode : Explode a string column.

        """
    def take_every(self, n: int) -> Self:
        """
        Take every nth value in the Series and return as a new Series.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        >>> df.select(pl.col("foo").take_every(3))
        shape: (3, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 4   │
        │ 7   │
        └─────┘

        """
    def head(self, n: int = ...) -> Self:
        """
        Get the first `n` rows.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6, 7]})
        >>> df.head(3)
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

        """
    def tail(self, n: int = ...) -> Self:
        """
        Get the last `n` rows.

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6, 7]})
        >>> df.tail(3)
        shape: (3, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 5   │
        │ 6   │
        │ 7   │
        └─────┘

        """
    def limit(self, n: int = ...) -> Self:
        """
        Get the first `n` rows.

        Alias for :func:`Expr.head`.

        Parameters
        ----------
        n
            Number of rows to return.

        """
    def pow(self, exponent: int | float | Series | Expr) -> Self:
        """
        Raise expression to the power of exponent.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4]})
        >>> df.select(pl.col("foo").pow(3))
        shape: (4, 1)
        ┌──────┐
        │ foo  │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ 1.0  │
        │ 8.0  │
        │ 27.0 │
        │ 64.0 │
        └──────┘

        """
    def is_in(self, other: Expr | Collection[Any] | Series) -> Self:
        """
        Check if elements of this expression are present in the other Series.

        Parameters
        ----------
        other
            Series or sequence of primitive type.

        Returns
        -------
        Expr that evaluates to a Boolean Series.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"sets": [[1, 2, 3], [1, 2], [9, 10]], "optional_members": [1, 2, 3]}
        ... )
        >>> df.select([pl.col("optional_members").is_in("sets").alias("contains")])
        shape: (3, 1)
        ┌──────────┐
        │ contains │
        │ ---      │
        │ bool     │
        ╞══════════╡
        │ true     │
        │ true     │
        │ false    │
        └──────────┘

        """
    def repeat_by(self, by: Expr | str) -> Self:
        """
        Repeat the elements in this Series as specified in the given expression.

        The repeated elements are expanded into a `List`.

        Parameters
        ----------
        by
            Numeric column that determines how often the values will be repeated.
            The column will be coerced to UInt32. Give this dtype to make the coercion a
            no-op.

        Returns
        -------
        Series of type List

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": ["x", "y", "z"],
        ...         "n": [1, 2, 3],
        ...     }
        ... )
        >>> df.select(pl.col("a").repeat_by("n"))
        shape: (3, 1)
        ┌─────────────────┐
        │ a               │
        │ ---             │
        │ list[str]       │
        ╞═════════════════╡
        │ ["x"]           │
        │ ["y", "y"]      │
        │ ["z", "z", "z"] │
        └─────────────────┘

        """
    def is_between(
        self,
        start: Expr | datetime | date | time | int | float | str,
        end: Expr | datetime | date | time | int | float | str,
        closed: ClosedInterval = ...,
    ) -> Self:
        """
        Check if this expression is between the given start and end values.

        Parameters
        ----------
        start
            Lower bound value (can be an expression or literal).
        end
            Upper bound value (can be an expression or literal).
        closed : {\'both\', \'left\', \'right\', \'none\'}
            Define which sides of the interval are closed (inclusive).

        Returns
        -------
        Expr that evaluates to a Boolean Series.

        Examples
        --------
        >>> df = pl.DataFrame({"num": [1, 2, 3, 4, 5]})
        >>> df.with_columns(pl.col("num").is_between(2, 4).alias("is_between"))
        shape: (5, 2)
        ┌─────┬────────────┐
        │ num ┆ is_between │
        │ --- ┆ ---        │
        │ i64 ┆ bool       │
        ╞═════╪════════════╡
        │ 1   ┆ false      │
        │ 2   ┆ true       │
        │ 3   ┆ true       │
        │ 4   ┆ true       │
        │ 5   ┆ false      │
        └─────┴────────────┘

        Use the ``closed`` argument to include or exclude the values at the bounds:

        >>> df.with_columns(
        ...     pl.col("num").is_between(2, 4, closed="left").alias("is_between")
        ... )
        shape: (5, 2)
        ┌─────┬────────────┐
        │ num ┆ is_between │
        │ --- ┆ ---        │
        │ i64 ┆ bool       │
        ╞═════╪════════════╡
        │ 1   ┆ false      │
        │ 2   ┆ true       │
        │ 3   ┆ true       │
        │ 4   ┆ false      │
        │ 5   ┆ false      │
        └─────┴────────────┘

        You can also use strings as well as numeric/temporal values (note: ensure that
        string literals are wrapped with ``lit`` so as not to conflate them with
        column names):

        >>> df = pl.DataFrame({"a": ["a", "b", "c", "d", "e"]})
        >>> df.with_columns(
        ...     pl.col("a")
        ...     .is_between(pl.lit("a"), pl.lit("c"), closed="both")
        ...     .alias("is_between")
        ... )
        shape: (5, 2)
        ┌─────┬────────────┐
        │ a   ┆ is_between │
        │ --- ┆ ---        │
        │ str ┆ bool       │
        ╞═════╪════════════╡
        │ a   ┆ true       │
        │ b   ┆ true       │
        │ c   ┆ true       │
        │ d   ┆ false      │
        │ e   ┆ false      │
        └─────┴────────────┘

        """
    def hash(
        self,
        seed: int = ...,
        seed_1: int | None = ...,
        seed_2: int | None = ...,
        seed_3: int | None = ...,
    ) -> Self:
        """
        Hash the elements in the selection.

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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, None],
        ...         "b": ["x", None, "z"],
        ...     }
        ... )
        >>> df.with_columns(pl.all().hash(10, 20, 30, 40))  # doctest: +IGNORE_RESULT
        shape: (3, 2)
        ┌──────────────────────┬──────────────────────┐
        │ a                    ┆ b                    │
        │ ---                  ┆ ---                  │
        │ u64                  ┆ u64                  │
        ╞══════════════════════╪══════════════════════╡
        │ 9774092659964970114  ┆ 13614470193936745724 │
        │ 1101441246220388612  ┆ 11638928888656214026 │
        │ 11638928888656214026 ┆ 13382926553367784577 │
        └──────────────────────┴──────────────────────┘

        """
    def reinterpret(self, signed: bool = ...) -> Self:
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
        >>> s = pl.Series("a", [1, 1, 2], dtype=pl.UInt64)
        >>> df = pl.DataFrame([s])
        >>> df.select(
        ...     [
        ...         pl.col("a").reinterpret(signed=True).alias("reinterpreted"),
        ...         pl.col("a").alias("original"),
        ...     ]
        ... )
        shape: (3, 2)
        ┌───────────────┬──────────┐
        │ reinterpreted ┆ original │
        │ ---           ┆ ---      │
        │ i64           ┆ u64      │
        ╞═══════════════╪══════════╡
        │ 1             ┆ 1        │
        │ 1             ┆ 1        │
        │ 2             ┆ 2        │
        └───────────────┴──────────┘

        """
    def inspect(self, fmt: str = ...) -> Self:
        """
        Print the value that this expression evaluates to and pass on the value.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 1, 2]})
        >>> df.select(pl.col("foo").cumsum().inspect("value is: {}").alias("bar"))
        value is: shape: (3,)
        Series: \'foo\' [i64]
        [
            1
            2
            4
        ]
        shape: (3, 1)
        ┌─────┐
        │ bar │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        │ 4   │
        └─────┘

        """
    def interpolate(self, method: InterpolationMethod = ...) -> Self:
        """
        Fill nulls with linear interpolation over missing values.

        Can also be used to regrid data to a new grid - see examples below.

        Parameters
        ----------
        method : {\'linear\', \'linear\'}
            Interpolation method

        Examples
        --------
        >>> # Fill nulls with linear interpolation
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, None, 3],
        ...         "b": [1.0, float("nan"), 3.0],
        ...     }
        ... )
        >>> df.select(pl.all().interpolate())
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 1   ┆ 1.0 │
        │ 2   ┆ NaN │
        │ 3   ┆ 3.0 │
        └─────┴─────┘
        >>> df_original_grid = pl.DataFrame(
        ...     {
        ...         "grid_points": [1, 3, 10],
        ...         "values": [2.0, 6.0, 20.0],
        ...     }
        ... )  # Interpolate from this to the new grid
        >>> df_new_grid = pl.DataFrame({"grid_points": range(1, 11)})
        >>> df_new_grid.join(
        ...     df_original_grid, on="grid_points", how="left"
        ... ).with_columns(pl.col("values").interpolate())
        shape: (10, 2)
        ┌─────────────┬────────┐
        │ grid_points ┆ values │
        │ ---         ┆ ---    │
        │ i64         ┆ f64    │
        ╞═════════════╪════════╡
        │ 1           ┆ 2.0    │
        │ 2           ┆ 4.0    │
        │ 3           ┆ 6.0    │
        │ 4           ┆ 8.0    │
        │ …           ┆ …      │
        │ 7           ┆ 14.0   │
        │ 8           ┆ 16.0   │
        │ 9           ┆ 18.0   │
        │ 10          ┆ 20.0   │
        └─────────────┴────────┘

        """
    def rolling_min(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = ...,
        min_periods: int | None = ...,
        center: bool = ...,
        by: str | None = ...,
        closed: ClosedInterval = ...,
    ) -> Self:
        """
        Apply a rolling min (moving min) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by a timedelta or the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_min(window_size=2),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ 1.0  │
        │ 2.0  │
        │ 3.0  │
        │ 4.0  │
        │ 5.0  │
        └──────┘

        """
    def rolling_max(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = ...,
        min_periods: int | None = ...,
        center: bool = ...,
        by: str | None = ...,
        closed: ClosedInterval = ...,
    ) -> Self:
        """
        Apply a rolling max (moving max) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by a timedelta or the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal, for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_max(window_size=2),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ 2.0  │
        │ 3.0  │
        │ 4.0  │
        │ 5.0  │
        │ 6.0  │
        └──────┘

        """
    def rolling_mean(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = ...,
        min_periods: int | None = ...,
        center: bool = ...,
        by: str | None = ...,
        closed: ClosedInterval = ...,
    ) -> Self:
        """
        Apply a rolling mean (moving mean) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by a timedelta or the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 8.0, 6.0, 2.0, 16.0, 10.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_mean(window_size=2),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ 4.5  │
        │ 7.0  │
        │ 4.0  │
        │ 9.0  │
        │ 13.0 │
        └──────┘

        """
    def rolling_sum(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = ...,
        min_periods: int | None = ...,
        center: bool = ...,
        by: str | None = ...,
        closed: ClosedInterval = ...,
    ) -> Self:
        """
        Apply a rolling sum (moving sum) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by a timedelta or the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length of the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            of dtype `{Date, Datetime}`
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_sum(window_size=2),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ 3.0  │
        │ 5.0  │
        │ 7.0  │
        │ 9.0  │
        │ 11.0 │
        └──────┘

        """
    def rolling_std(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = ...,
        min_periods: int | None = ...,
        center: bool = ...,
        by: str | None = ...,
        closed: ClosedInterval = ...,
    ) -> Self:
        """
        Compute a rolling standard deviation.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by a timedelta or the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_std(window_size=3),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────────┐
        │ A        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ null     │
        │ null     │
        │ 1.0      │
        │ 1.0      │
        │ 1.527525 │
        │ 2.0      │
        └──────────┘

        """
    def rolling_var(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = ...,
        min_periods: int | None = ...,
        center: bool = ...,
        by: str | None = ...,
        closed: ClosedInterval = ...,
    ) -> Self:
        """
        Compute a rolling variance.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by a timedelta or the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_var(window_size=3),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────────┐
        │ A        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ null     │
        │ null     │
        │ 1.0      │
        │ 1.0      │
        │ 2.333333 │
        │ 4.0      │
        └──────────┘

        """
    def rolling_median(
        self,
        window_size: int | timedelta | str,
        weights: list[float] | None = ...,
        min_periods: int | None = ...,
        center: bool = ...,
        by: str | None = ...,
        closed: ClosedInterval = ...,
    ) -> Self:
        """
        Compute a rolling median.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by a timedelta or the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_median(window_size=3),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ null │
        │ 2.0  │
        │ 3.0  │
        │ 4.0  │
        │ 6.0  │
        └──────┘

        """
    def rolling_quantile(
        self,
        quantile: float,
        interpolation: RollingInterpolationMethod = ...,
        window_size: int | timedelta | str = ...,
        weights: list[float] | None = ...,
        min_periods: int | None = ...,
        center: bool = ...,
        by: str | None = ...,
        closed: ClosedInterval = ...,
    ) -> Self:
        """
        Compute a rolling quantile.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {\'nearest\', \'higher\', \'lower\', \'midpoint\', \'linear\'}
            Interpolation method.
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic temporal
            size indicated by a timedelta or the following string language:

            - 1ns   (1 nanosecond)
            - 1us   (1 microsecond)
            - 1ms   (1 millisecond)
            - 1s    (1 second)
            - 1m    (1 minute)
            - 1h    (1 hour)
            - 1d    (1 day)
            - 1w    (1 week)
            - 1mo   (1 calendar month)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that will be multiplied
            elementwise with the values in the window.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype `{Date, Datetime}`
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive).

        Warnings
        --------
        This functionality is experimental and may change without it being considered a
        breaking change.

        Notes
        -----
        If you want to compute multiple aggregation statistics over the same dynamic
        window, consider using `groupby_rolling` this method can cache the window size
        computation.

        Examples
        --------
        >>> df = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]})
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_quantile(quantile=0.33, window_size=3),
        ...     ]
        ... )
        shape: (6, 1)
        ┌──────┐
        │ A    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ null │
        │ null │
        │ 1.0  │
        │ 2.0  │
        │ 3.0  │
        │ 4.0  │
        └──────┘

        """
    def rolling_apply(
        self,
        function: Callable[[Series], Any],
        window_size: int,
        weights: list[float] | None = ...,
        min_periods: int | None = ...,
        center: bool = ...,
    ) -> Self:
        """
        Apply a custom rolling window function.

        Prefer the specific rolling window functions over this one, as they are faster.

        Prefer:

            * rolling_min
            * rolling_max
            * rolling_mean
            * rolling_sum

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
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [1.0, 2.0, 9.0, 2.0, 13.0],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("A").rolling_apply(lambda s: s.std(), window_size=3),
        ...     ]
        ... )
         shape: (5, 1)
        ┌──────────┐
        │ A        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ null     │
        │ null     │
        │ 4.358899 │
        │ 4.041452 │
        │ 5.567764 │
        └──────────┘

        """
    def rolling_skew(self, window_size: int, bias: bool = ...) -> Self:
        """
        Compute a rolling skew.

        Parameters
        ----------
        window_size
            Integer size of the rolling window.
        bias
            If False, the calculations are corrected for statistical bias.

        """
    def abs(self) -> Self:
        """
        Compute absolute values.

        Same as `abs(expr)`.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "A": [-1.0, 0.0, 1.0, 2.0],
        ...     }
        ... )
        >>> df.select(pl.col("A").abs())
        shape: (4, 1)
        ┌─────┐
        │ A   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        │ 0.0 │
        │ 1.0 │
        │ 2.0 │
        └─────┘

        """
    def argsort(self, *args, **kwargs) -> Self:
        """
        Get the index values that would sort this column.

        Alias for :func:`Expr.arg_sort`.

        .. deprecated:: 0.16.5
            `Expr.argsort` will be removed in favour of `Expr.arg_sort`.

        Parameters
        ----------
        descending
            Sort in descending order.
        nulls_last
            Place null values last instead of first.

        Returns
        -------
        Expr
            Series of dtype UInt32.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [20, 10, 30],
        ...     }
        ... )
        >>> df.select(pl.col("a").argsort())
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 1   │
        │ 0   │
        │ 2   │
        └─────┘

        """
    def rank(self, *args, **kwargs) -> Self:
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

        Examples
        --------
        The \'average\' method:

        >>> df = pl.DataFrame({"a": [3, 6, 1, 1, 6]})
        >>> df.select(pl.col("a").rank())
        shape: (5, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f32 │
        ╞═════╡
        │ 3.0 │
        │ 4.5 │
        │ 1.5 │
        │ 1.5 │
        │ 4.5 │
        └─────┘

        The \'ordinal\' method:

        >>> df = pl.DataFrame({"a": [3, 6, 1, 1, 6]})
        >>> df.select(pl.col("a").rank("ordinal"))
        shape: (5, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 3   │
        │ 4   │
        │ 1   │
        │ 2   │
        │ 5   │
        └─────┘

        """
    def diff(self, n: int = ..., null_behavior: NullBehavior = ...) -> Self:
        """
        Calculate the n-th discrete difference.

        Parameters
        ----------
        n
            Number of slots to shift.
        null_behavior : {\'ignore\', \'drop\'}
            How to handle null values.

        Examples
        --------
        >>> df = pl.DataFrame({"int": [20, 10, 30, 25, 35]})
        >>> df.with_columns(change=pl.col("int").diff())
        shape: (5, 2)
        ┌─────┬────────┐
        │ int ┆ change │
        │ --- ┆ ---    │
        │ i64 ┆ i64    │
        ╞═════╪════════╡
        │ 20  ┆ null   │
        │ 10  ┆ -10    │
        │ 30  ┆ 20     │
        │ 25  ┆ -5     │
        │ 35  ┆ 10     │
        └─────┴────────┘

        >>> df.with_columns(change=pl.col("int").diff(n=2))
        shape: (5, 2)
        ┌─────┬────────┐
        │ int ┆ change │
        │ --- ┆ ---    │
        │ i64 ┆ i64    │
        ╞═════╪════════╡
        │ 20  ┆ null   │
        │ 10  ┆ null   │
        │ 30  ┆ 10     │
        │ 25  ┆ 15     │
        │ 35  ┆ 5      │
        └─────┴────────┘

        >>> df.select(pl.col("int").diff(n=2, null_behavior="drop").alias("diff"))
        shape: (3, 1)
        ┌──────┐
        │ diff │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ 10   │
        │ 15   │
        │ 5    │
        └──────┘

        """
    def pct_change(self, n: int = ...) -> Self:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [10, 11, 12, None, 12],
        ...     }
        ... )
        >>> df.with_columns(pl.col("a").pct_change().alias("pct_change"))
        shape: (5, 2)
        ┌──────┬────────────┐
        │ a    ┆ pct_change │
        │ ---  ┆ ---        │
        │ i64  ┆ f64        │
        ╞══════╪════════════╡
        │ 10   ┆ null       │
        │ 11   ┆ 0.1        │
        │ 12   ┆ 0.090909   │
        │ null ┆ 0.0        │
        │ 12   ┆ 0.0        │
        └──────┴────────────┘

        """
    def skew(self, bias: bool = ...) -> Self:
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

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 2, 1]})
        >>> df.select(pl.col("a").skew())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.343622 │
        └──────────┘

        """
    def kurtosis(self, fisher: bool = ..., bias: bool = ...) -> Self:
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
        >>> df = pl.DataFrame({"a": [1, 2, 3, 2, 1]})
        >>> df.select(pl.col("a").kurtosis())
        shape: (1, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ f64       │
        ╞═══════════╡
        │ -1.153061 │
        └───────────┘

        """
    def clip(self, min_val: int | float, max_val: int | float) -> Self:
        """
        Clip (limit) the values in an array to a `min` and `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        min_val
            Minimum value.
        max_val
            Maximum value.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [-50, 5, None, 50]})
        >>> df.with_columns(pl.col("foo").clip(1, 10).alias("foo_clipped"))
        shape: (4, 2)
        ┌──────┬─────────────┐
        │ foo  ┆ foo_clipped │
        │ ---  ┆ ---         │
        │ i64  ┆ i64         │
        ╞══════╪═════════════╡
        │ -50  ┆ 1           │
        │ 5    ┆ 5           │
        │ null ┆ null        │
        │ 50   ┆ 10          │
        └──────┴─────────────┘

        """
    def clip_min(self, min_val: int | float) -> Self:
        """
        Clip (limit) the values in an array to a `min` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        min_val
            Minimum value.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [-50, 5, None, 50]})
        >>> df.with_columns(pl.col("foo").clip_min(0).alias("foo_clipped"))
        shape: (4, 2)
        ┌──────┬─────────────┐
        │ foo  ┆ foo_clipped │
        │ ---  ┆ ---         │
        │ i64  ┆ i64         │
        ╞══════╪═════════════╡
        │ -50  ┆ 0           │
        │ 5    ┆ 5           │
        │ null ┆ null        │
        │ 50   ┆ 50          │
        └──────┴─────────────┘

        """
    def clip_max(self, max_val: int | float) -> Self:
        """
        Clip (limit) the values in an array to a `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        max_val
            Maximum value.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [-50, 5, None, 50]})
        >>> df.with_columns(pl.col("foo").clip_max(0).alias("foo_clipped"))
        shape: (4, 2)
        ┌──────┬─────────────┐
        │ foo  ┆ foo_clipped │
        │ ---  ┆ ---         │
        │ i64  ┆ i64         │
        ╞══════╪═════════════╡
        │ -50  ┆ -50         │
        │ 5    ┆ 0           │
        │ null ┆ null        │
        │ 50   ┆ 0           │
        └──────┴─────────────┘

        """
    def lower_bound(self) -> Self:
        """
        Calculate the lower bound.

        Returns a unit Series with the lowest value possible for the dtype of this
        expression.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 2, 1]})
        >>> df.select(pl.col("a").lower_bound())
        shape: (1, 1)
        ┌──────────────────────┐
        │ a                    │
        │ ---                  │
        │ i64                  │
        ╞══════════════════════╡
        │ -9223372036854775808 │
        └──────────────────────┘

        """
    def upper_bound(self) -> Self:
        """
        Calculate the upper bound.

        Returns a unit Series with the highest value possible for the dtype of this
        expression.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3, 2, 1]})
        >>> df.select(pl.col("a").upper_bound())
        shape: (1, 1)
        ┌─────────────────────┐
        │ a                   │
        │ ---                 │
        │ i64                 │
        ╞═════════════════════╡
        │ 9223372036854775807 │
        └─────────────────────┘

        """
    def sign(self) -> Self:
        """
        Compute the element-wise indication of the sign.

        The returned values can be -1, 0, or 1:

        * -1 if x  < 0.
        *  0 if x == 0.
        *  1 if x  > 0.

        (null values are preserved as-is).

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-9.0, -0.0, 0.0, 4.0, None]})
        >>> df.select(pl.col("a").sign())
        shape: (5, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ i64  │
        ╞══════╡
        │ -1   │
        │ 0    │
        │ 0    │
        │ 1    │
        │ null │
        └──────┘

        """
    def sin(self) -> Self:
        """
        Compute the element-wise value for the sine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.0]})
        >>> df.select(pl.col("a").sin())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.0 │
        └─────┘

        """
    def cos(self) -> Self:
        """
        Compute the element-wise value for the cosine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.0]})
        >>> df.select(pl.col("a").cos())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 1.0 │
        └─────┘

        """
    def tan(self) -> Self:
        """
        Compute the element-wise value for the tangent.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").tan().round(2))
        shape: (1, 1)
        ┌──────┐
        │ a    │
        │ ---  │
        │ f64  │
        ╞══════╡
        │ 1.56 │
        └──────┘

        """
    def arcsin(self) -> Self:
        """
        Compute the element-wise value for the inverse sine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arcsin())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.570796 │
        └──────────┘

        """
    def arccos(self) -> Self:
        """
        Compute the element-wise value for the inverse cosine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [0.0]})
        >>> df.select(pl.col("a").arccos())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.570796 │
        └──────────┘

        """
    def arctan(self) -> Self:
        """
        Compute the element-wise value for the inverse tangent.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arctan())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.785398 │
        └──────────┘

        """
    def sinh(self) -> Self:
        """
        Compute the element-wise value for the hyperbolic sine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").sinh())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.175201 │
        └──────────┘

        """
    def cosh(self) -> Self:
        """
        Compute the element-wise value for the hyperbolic cosine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").cosh())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.543081 │
        └──────────┘

        """
    def tanh(self) -> Self:
        """
        Compute the element-wise value for the hyperbolic tangent.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").tanh())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.761594 │
        └──────────┘

        """
    def arcsinh(self) -> Self:
        """
        Compute the element-wise value for the inverse hyperbolic sine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arcsinh())
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.881374 │
        └──────────┘

        """
    def arccosh(self) -> Self:
        """
        Compute the element-wise value for the inverse hyperbolic cosine.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arccosh())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 0.0 │
        └─────┘

        """
    def arctanh(self) -> Self:
        """
        Compute the element-wise value for the inverse hyperbolic tangent.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1.0]})
        >>> df.select(pl.col("a").arctanh())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ inf │
        └─────┘

        """
    def reshape(self, dims: tuple[int, ...]) -> Self:
        """
        Reshape this Expr to a flat Series or a Series of Lists.

        Parameters
        ----------
        dims
            Tuple of the dimension sizes. If a -1 is used in any of the dimensions, that
            dimension is inferred.

        Returns
        -------
        Expr
            If a single dimension is given, results in a flat Series of shape (len,).
            If a multiple dimensions are given, results in a Series of Lists with shape
            (rows, cols).

        See Also
        --------
        ExprListNameSpace.explode : Explode a list column.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        >>> df.select(pl.col("foo").reshape((3, 3)))
        shape: (3, 1)
        ┌───────────┐
        │ foo       │
        │ ---       │
        │ list[i64] │
        ╞═══════════╡
        │ [1, 2, 3] │
        │ [4, 5, 6] │
        │ [7, 8, 9] │
        └───────────┘

        """
    def shuffle(self, seed: int | None = ...) -> Self:
        """
        Shuffle the contents of this expression.

        Parameters
        ----------
        seed
            Seed for the random number generator. If set to None (default), a random
            seed is generated using the ``random`` module.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").shuffle(seed=1))
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        │ 1   │
        │ 3   │
        └─────┘

        """
    def sample(
        self,
        n: int | None = ...,
        frac: float | None = ...,
        with_replacement: bool = ...,
        shuffle: bool = ...,
        seed: int | None = ...,
    ) -> Self:
        """
        Sample from this expression.

        Parameters
        ----------
        n
            Number of items to return. Cannot be used with `frac`. Defaults to 1 if
            `frac` is None.
        frac
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
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").sample(frac=1.0, with_replacement=True, seed=1))
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 3   │
        │ 1   │
        │ 1   │
        └─────┘

        """
    def ewm_mean(
        self,
        com: float | None = ...,
        span: float | None = ...,
        half_life: float | None = ...,
        alpha: float | None = ...,
        adjust: bool = ...,
        min_periods: int = ...,
        ignore_nulls: bool = ...,
    ) -> Self:
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


        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").ewm_mean(com=1))
        shape: (3, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.0      │
        │ 1.666667 │
        │ 2.428571 │
        └──────────┘

        """
    def ewm_std(
        self,
        com: float | None = ...,
        span: float | None = ...,
        half_life: float | None = ...,
        alpha: float | None = ...,
        adjust: bool = ...,
        bias: bool = ...,
        min_periods: int = ...,
        ignore_nulls: bool = ...,
    ) -> Self:
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
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").ewm_std(com=1))
        shape: (3, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.0      │
        │ 0.707107 │
        │ 0.963624 │
        └──────────┘

        """
    def ewm_var(
        self,
        com: float | None = ...,
        span: float | None = ...,
        half_life: float | None = ...,
        alpha: float | None = ...,
        adjust: bool = ...,
        bias: bool = ...,
        min_periods: int = ...,
        ignore_nulls: bool = ...,
    ) -> Self:
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
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").ewm_var(com=1))
        shape: (3, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.0      │
        │ 0.5      │
        │ 0.928571 │
        └──────────┘

        """
    def extend_constant(self, value: PythonLiteral | None, n: int) -> Self:
        """
        Extremely fast method for extending the Series with \'n\' copies of a value.

        Parameters
        ----------
        value
            A constant literal value (not an expression) with which to extend the
            expression result Series; can pass None to extend with nulls.
        n
            The number of additional values that will be added.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1, 2, 3]})
        >>> df.select((pl.col("values") - 1).extend_constant(99, n=2))
        shape: (5, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ 0      │
        │ 1      │
        │ 2      │
        │ 99     │
        │ 99     │
        └────────┘

        """
    def value_counts(self, multithreaded: bool = ..., sort: bool = ...) -> Self:
        """
        Count all unique values and create a struct mapping value to count.

        Parameters
        ----------
        multithreaded:
            Better to turn this off in the aggregation context, as it can lead to
            contention.
        sort:
            Ensure the output is sorted from most values to least.

        Returns
        -------
        Dtype Struct

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "id": ["a", "b", "b", "c", "c", "c"],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("id").value_counts(sort=True),
        ...     ]
        ... )
        shape: (3, 1)
        ┌───────────┐
        │ id        │
        │ ---       │
        │ struct[2] │
        ╞═══════════╡
        │ {"c",3}   │
        │ {"b",2}   │
        │ {"a",1}   │
        └───────────┘

        """
    def unique_counts(self) -> Self:
        """
        Return a count of the unique values in the order of appearance.

        This method differs from `value_counts` in that it does not return the
        values, only the counts and might be faster

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "id": ["a", "b", "b", "c", "c", "c"],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("id").unique_counts(),
        ...     ]
        ... )
        shape: (3, 1)
        ┌─────┐
        │ id  │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 1   │
        │ 2   │
        │ 3   │
        └─────┘

        """
    def log(self, base: float = ...) -> Self:
        """
        Compute the logarithm to a given base.

        Parameters
        ----------
        base
            Given base, defaults to `e`

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").log(base=2))
        shape: (3, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.0      │
        │ 1.0      │
        │ 1.584963 │
        └──────────┘

        """
    def entropy(self, base: float = ..., normalize: bool = ...) -> Self:
        """
        Computes the entropy.

        Uses the formula ``-sum(pk * log(pk)`` where ``pk`` are discrete probabilities.

        Parameters
        ----------
        base
            Given base, defaults to `e`
        normalize
            Normalize pk if it doesn\'t sum to 1.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").entropy(base=2))
        shape: (1, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.459148 │
        └──────────┘
        >>> df.select(pl.col("a").entropy(base=2, normalize=False))
        shape: (1, 1)
        ┌───────────┐
        │ a         │
        │ ---       │
        │ f64       │
        ╞═══════════╡
        │ -6.754888 │
        └───────────┘

        """
    def cumulative_eval(self, expr: Expr, min_periods: int = ..., parallel: bool = ...) -> Self:
        """
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
        >>> df = pl.DataFrame({"values": [1, 2, 3, 4, 5]})
        >>> df.select(
        ...     [
        ...         pl.col("values").cumulative_eval(
        ...             pl.element().first() - pl.element().last() ** 2
        ...         )
        ...     ]
        ... )
        shape: (5, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ f64    │
        ╞════════╡
        │ 0.0    │
        │ -3.0   │
        │ -8.0   │
        │ -15.0  │
        │ -24.0  │
        └────────┘

        """
    def set_sorted(self, *args, **kwargs) -> Self:
        """
        Flags the expression as \'sorted\'.

        Enables downstream code to user fast paths for sorted arrays.

        Parameters
        ----------
        descending
            Whether the `Series` order is descending.

        Warnings
        --------
        This can lead to incorrect results if this `Series` is not sorted!!
        Use with care!

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1, 2, 3]})
        >>> df.select(pl.col("values").set_sorted().max())
        shape: (1, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ 3      │
        └────────┘

        """
    def list(self) -> Self:
        """
        Aggregate to list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": [4, 5, 6],
        ...     }
        ... )
        >>> df.select(pl.all().list())
        shape: (1, 2)
        ┌───────────┬───────────┐
        │ a         ┆ b         │
        │ ---       ┆ ---       │
        │ list[i64] ┆ list[i64] │
        ╞═══════════╪═══════════╡
        │ [1, 2, 3] ┆ [4, 5, 6] │
        └───────────┴───────────┘

        """
    def shrink_dtype(self) -> Self:
        """
        Shrink numeric columns to the minimal required datatype.

        Shrink to the dtype needed to fit the extrema of this [`Series`].
        This can be used to reduce memory pressure.

        Examples
        --------
        >>> pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": [1, 2, 2 << 32],
        ...         "c": [-1, 2, 1 << 30],
        ...         "d": [-112, 2, 112],
        ...         "e": [-112, 2, 129],
        ...         "f": ["a", "b", "c"],
        ...         "g": [0.1, 1.32, 0.12],
        ...         "h": [True, None, False],
        ...     }
        ... ).select(pl.all().shrink_dtype())
        shape: (3, 8)
        ┌─────┬────────────┬────────────┬──────┬──────┬─────┬──────┬───────┐
        │ a   ┆ b          ┆ c          ┆ d    ┆ e    ┆ f   ┆ g    ┆ h     │
        │ --- ┆ ---        ┆ ---        ┆ ---  ┆ ---  ┆ --- ┆ ---  ┆ ---   │
        │ i8  ┆ i64        ┆ i32        ┆ i8   ┆ i16  ┆ str ┆ f32  ┆ bool  │
        ╞═════╪════════════╪════════════╪══════╪══════╪═════╪══════╪═══════╡
        │ 1   ┆ 1          ┆ -1         ┆ -112 ┆ -112 ┆ a   ┆ 0.1  ┆ true  │
        │ 2   ┆ 2          ┆ 2          ┆ 2    ┆ 2    ┆ b   ┆ 1.32 ┆ null  │
        │ 3   ┆ 8589934592 ┆ 1073741824 ┆ 112  ┆ 129  ┆ c   ┆ 0.12 ┆ false │
        └─────┴────────────┴────────────┴──────┴──────┴─────┴──────┴───────┘

        """
    def map_dict(self, remapping: dict[Any, Any]) -> Self:
        """
        Replace values in column according to remapping dictionary.

        Needs a global string cache for lazily evaluated queries on columns of
        type ``pl.Categorical``.

        Parameters
        ----------
        remapping
            Dictionary containing the before/after values to map.
        default
            Value to use when the remapping dict does not contain the lookup value.
            Use ``pl.first()``, to keep the original value.

        Examples
        --------
        >>> country_code_dict = {
        ...     "CA": "Canada",
        ...     "DE": "Germany",
        ...     "FR": "France",
        ...     None: "Not specified",
        ... }
        >>> df = pl.DataFrame(
        ...     {
        ...         "country_code": ["FR", None, "ES", "DE"],
        ...     }
        ... ).with_row_count()
        >>> df
        shape: (4, 2)
        ┌────────┬──────────────┐
        │ row_nr ┆ country_code │
        │ ---    ┆ ---          │
        │ u32    ┆ str          │
        ╞════════╪══════════════╡
        │ 0      ┆ FR           │
        │ 1      ┆ null         │
        │ 2      ┆ ES           │
        │ 3      ┆ DE           │
        └────────┴──────────────┘

        >>> df.with_columns(
        ...     pl.col("country_code").map_dict(country_code_dict).alias("remapped")
        ... )
        shape: (4, 3)
        ┌────────┬──────────────┬───────────────┐
        │ row_nr ┆ country_code ┆ remapped      │
        │ ---    ┆ ---          ┆ ---           │
        │ u32    ┆ str          ┆ str           │
        ╞════════╪══════════════╪═══════════════╡
        │ 0      ┆ FR           ┆ France        │
        │ 1      ┆ null         ┆ Not specified │
        │ 2      ┆ ES           ┆ null          │
        │ 3      ┆ DE           ┆ Germany       │
        └────────┴──────────────┴───────────────┘

        Set a default value for values that cannot be mapped...

        >>> df.with_columns(
        ...     pl.col("country_code")
        ...     .map_dict(country_code_dict, default="unknown")
        ...     .alias("remapped")
        ... )
        shape: (4, 3)
        ┌────────┬──────────────┬───────────────┐
        │ row_nr ┆ country_code ┆ remapped      │
        │ ---    ┆ ---          ┆ ---           │
        │ u32    ┆ str          ┆ str           │
        ╞════════╪══════════════╪═══════════════╡
        │ 0      ┆ FR           ┆ France        │
        │ 1      ┆ null         ┆ Not specified │
        │ 2      ┆ ES           ┆ unknown       │
        │ 3      ┆ DE           ┆ Germany       │
        └────────┴──────────────┴───────────────┘

        ...or keep the original value, by making use of ``pl.first()``:

        >>> df.with_columns(
        ...     pl.col("country_code")
        ...     .map_dict(country_code_dict, default=pl.first())
        ...     .alias("remapped")
        ... )
        shape: (4, 3)
        ┌────────┬──────────────┬───────────────┐
        │ row_nr ┆ country_code ┆ remapped      │
        │ ---    ┆ ---          ┆ ---           │
        │ u32    ┆ str          ┆ str           │
        ╞════════╪══════════════╪═══════════════╡
        │ 0      ┆ FR           ┆ France        │
        │ 1      ┆ null         ┆ Not specified │
        │ 2      ┆ ES           ┆ ES            │
        │ 3      ┆ DE           ┆ Germany       │
        └────────┴──────────────┴───────────────┘

        ...or keep the original value, by explicitly referring to the column:

        >>> df.with_columns(
        ...     pl.col("country_code")
        ...     .map_dict(country_code_dict, default=pl.col("country_code"))
        ...     .alias("remapped")
        ... )
        shape: (4, 3)
        ┌────────┬──────────────┬───────────────┐
        │ row_nr ┆ country_code ┆ remapped      │
        │ ---    ┆ ---          ┆ ---           │
        │ u32    ┆ str          ┆ str           │
        ╞════════╪══════════════╪═══════════════╡
        │ 0      ┆ FR           ┆ France        │
        │ 1      ┆ null         ┆ Not specified │
        │ 2      ┆ ES           ┆ ES            │
        │ 3      ┆ DE           ┆ Germany       │
        └────────┴──────────────┴───────────────┘

        If you need to access different columns to set a default value, a struct needs
        to be constructed; in the first field is the column that you want to remap and
        the rest of the fields are the other columns used in the default expression.

        >>> df.with_columns(
        ...     pl.struct(pl.col(["country_code", "row_nr"])).map_dict(
        ...         remapping=country_code_dict,
        ...         default=pl.col("row_nr").cast(pl.Utf8),
        ...     )
        ... )
        shape: (4, 2)
        ┌────────┬───────────────┐
        │ row_nr ┆ country_code  │
        │ ---    ┆ ---           │
        │ u32    ┆ str           │
        ╞════════╪═══════════════╡
        │ 0      ┆ France        │
        │ 1      ┆ Not specified │
        │ 2      ┆ 2             │
        │ 3      ┆ Germany       │
        └────────┴───────────────┘

        """
    @property
    def arr(self): ...
    @property
    def bin(self): ...
    @property
    def cat(self): ...
    @property
    def dt(self): ...
    @property
    def meta(self): ...
    @property
    def str(self): ...
    @property
    def struct(self): ...

def _prepare_alpha(
    com: float | int | None = ...,
    span: float | int | None = ...,
    half_life: float | int | None = ...,
    alpha: float | int | None = ...,
) -> float:
    """Normalise EWM decay specification in terms of smoothing factor 'alpha'."""

def _prepare_rolling_window_args(
    window_size: int | timedelta | str, min_periods: int | None = ...
) -> tuple[str, int]: ...

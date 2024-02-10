#: version 0.18.8
import P
import np as np
import pl
from builtins import PyExpr
from datetime import timedelta
from polars.datatypes.classes import Categorical as Categorical, FLOAT_DTYPES as FLOAT_DTYPES, Struct as Struct, UInt32 as UInt32, Utf8 as Utf8
from polars.datatypes.convert import is_polars_dtype as is_polars_dtype, py_type_to_dtype as py_type_to_dtype
from polars.dependencies import _check_for_numpy as _check_for_numpy
from polars.expr.array import ExprArrayNameSpace as ExprArrayNameSpace
from polars.expr.binary import ExprBinaryNameSpace as ExprBinaryNameSpace
from polars.expr.categorical import ExprCatNameSpace as ExprCatNameSpace
from polars.expr.datetime import ExprDateTimeNameSpace as ExprDateTimeNameSpace
from polars.expr.list import ExprListNameSpace as ExprListNameSpace
from polars.expr.meta import ExprMetaNameSpace as ExprMetaNameSpace
from polars.expr.string import ExprStringNameSpace as ExprStringNameSpace
from polars.expr.struct import ExprStructNameSpace as ExprStructNameSpace
from polars.utils._parse_expr_input import parse_as_expression as parse_as_expression, parse_as_list_of_expressions as parse_as_list_of_expressions
from polars.utils.convert import _timedelta_to_pl_duration as _timedelta_to_pl_duration
from polars.utils.decorators import deprecated_alias as deprecated_alias, warn_closed_future_change as warn_closed_future_change
from polars.utils.meta import threadpool_size as threadpool_size
from polars.utils.various import sphinx_accessor as sphinx_accessor
from typing import Any, Callable, ClassVar as _ClassVar, Collection, Iterable, NoReturn

TYPE_CHECKING: bool
INTEGER_DTYPES: frozenset
py_arg_where: builtin_function_or_method
pyreduce: builtin_function_or_method

class Expr:
    _pyexpr: _ClassVar[None] = ...
    _accessors: _ClassVar[set] = ...
    @classmethod
    def _from_pyexpr(cls, pyexpr: PyExpr) -> Self: ...
    def _to_pyexpr(self, other: Any) -> PyExpr: ...
    def _to_expr(self, other: Any) -> Expr: ...
    def _repr_html_(self) -> str: ...
    def __bool__(self) -> NoReturn: ...
    def __abs__(self) -> Self: ...
    def __add__(self, other: Any) -> Self: ...
    def __radd__(self, other: Any) -> Self: ...
    def __and__(self, other: Expr | int) -> Self: ...
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
    def __or__(self, other: Expr | int) -> Self: ...
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
    def __array_ufunc__(self, ufunc: Callable[..., Any], method: str, *inputs: Any, **kwargs: Any) -> Self:
        """Numpy universal functions."""
    @classmethod
    def from_json(cls, value: str) -> Self:
        """
        Read an expression from a JSON encoded string to construct an Expression.

        Parameters
        ----------
        value
            JSON encoded string value

        """
    def to_physical(self) -> Self:
        '''
        Cast to physical representation of the logical dtype.

        - :func:`polars.datatypes.Date` -> :func:`polars.datatypes.Int32`
        - :func:`polars.datatypes.Datetime` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Time` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Duration` -> :func:`polars.datatypes.Int64`
        - :func:`polars.datatypes.Categorical` -> :func:`polars.datatypes.UInt32`
        - ``List(inner)`` -> ``List(physical of inner)``

        Other data types will be left unchanged.

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

        '''
    def any(self, drop_nulls: bool = ...) -> Self:
        '''
        Check if any boolean value in a Boolean column is `True`.

        Parameters
        ----------
        drop_nulls
            If False, return None if there are nulls but no Trues.

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
        >>> df = pl.DataFrame(dict(x=[None, False], y=[None, True]))
        >>> df.select(pl.col("x").any(True), pl.col("y").any(True))
        shape: (1, 2)
        ┌───────┬──────┐
        │ x     ┆ y    │
        │ ---   ┆ ---  │
        │ bool  ┆ bool │
        ╞═══════╪══════╡
        │ false ┆ true │
        └───────┴──────┘
        >>> df.select(pl.col("x").any(False), pl.col("y").any(False))
        shape: (1, 2)
        ┌──────┬──────┐
        │ x    ┆ y    │
        │ ---  ┆ ---  │
        │ bool ┆ bool │
        ╞══════╪══════╡
        │ null ┆ true │
        └──────┴──────┘

        '''
    def all(self, drop_nulls: bool = ...) -> Self:
        '''
        Check if all boolean values in a Boolean column are `True`.

        This method is an expression - not to be confused with
        :func:`polars.all` which is a function to select all columns.

        Parameters
        ----------
        drop_nulls
            If False, return None if there are any nulls.


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
        >>> df = pl.DataFrame(dict(x=[None, False], y=[None, True]))
        >>> df.select(pl.col("x").all(True), pl.col("y").all(True))
        shape: (1, 2)
        ┌───────┬───────┐
        │ x     ┆ y     │
        │ ---   ┆ ---   │
        │ bool  ┆ bool  │
        ╞═══════╪═══════╡
        │ false ┆ false │
        └───────┴───────┘
        >>> df.select(pl.col("x").all(False), pl.col("y").all(False))
        shape: (1, 2)
        ┌──────┬──────┐
        │ x    ┆ y    │
        │ ---  ┆ ---  │
        │ bool ┆ bool │
        ╞══════╪══════╡
        │ null ┆ null │
        └──────┴──────┘

        '''
    def arg_true(self) -> Self:
        '''
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
        polars.arg_where

        '''
    def sqrt(self) -> Self:
        '''
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

        '''
    def cbrt(self) -> Self:
        '''
        Compute the cube root of the elements.

        Examples
        --------
        >>> df = pl.DataFrame({"values": [1.0, 2.0, 4.0]})
        >>> df.select(pl.col("values").cbrt())
        shape: (3, 1)
        ┌──────────┐
        │ values   │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 1.0      │
        │ 1.259921 │
        │ 1.587401 │
        └──────────┘

        '''
    def log10(self) -> Self:
        '''
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

        '''
    def exp(self) -> Self:
        '''
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

        '''
    def alias(self, name: str) -> Self:
        '''
        Rename the expression.

        Parameters
        ----------
        name
            The new name.

        See Also
        --------
        map_alias
        prefix
        suffix

        Examples
        --------
        Rename an expression to avoid overwriting an existing column.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["x", "y", "z"],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("a") + 10,
        ...     pl.col("b").str.to_uppercase().alias("c"),
        ... )
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ str ┆ str │
        ╞═════╪═════╪═════╡
        │ 11  ┆ x   ┆ X   │
        │ 12  ┆ y   ┆ Y   │
        │ 13  ┆ z   ┆ Z   │
        └─────┴─────┴─────┘

        Overwrite the default name of literal columns to prevent errors due to duplicate
        column names.

        >>> df.with_columns(
        ...     pl.lit(True).alias("c"),
        ...     pl.lit(4.0).alias("d"),
        ... )
        shape: (3, 4)
        ┌─────┬─────┬──────┬─────┐
        │ a   ┆ b   ┆ c    ┆ d   │
        │ --- ┆ --- ┆ ---  ┆ --- │
        │ i64 ┆ str ┆ bool ┆ f64 │
        ╞═════╪═════╪══════╪═════╡
        │ 1   ┆ x   ┆ true ┆ 4.0 │
        │ 2   ┆ y   ┆ true ┆ 4.0 │
        │ 3   ┆ z   ┆ true ┆ 4.0 │
        └─────┴─────┴──────┴─────┘

        '''
    def map_alias(self, function: Callable[[str], str]) -> Self:
        '''
        Rename the output of an expression by mapping a function over the root name.

        Parameters
        ----------
        function
            Function that maps a root name to a new name.

        See Also
        --------
        alias
        prefix
        suffix

        Examples
        --------
        Remove a common suffix and convert to lower case.

        >>> df = pl.DataFrame(
        ...     {
        ...         "A_reverse": [3, 2, 1],
        ...         "B_reverse": ["z", "y", "x"],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.all().reverse().map_alias(lambda c: c.rstrip("_reverse").lower())
        ... )
        shape: (3, 4)
        ┌───────────┬───────────┬─────┬─────┐
        │ A_reverse ┆ B_reverse ┆ a   ┆ b   │
        │ ---       ┆ ---       ┆ --- ┆ --- │
        │ i64       ┆ str       ┆ i64 ┆ str │
        ╞═══════════╪═══════════╪═════╪═════╡
        │ 3         ┆ z         ┆ 1   ┆ x   │
        │ 2         ┆ y         ┆ 2   ┆ y   │
        │ 1         ┆ x         ┆ 3   ┆ z   │
        └───────────┴───────────┴─────┴─────┘

        '''
    def prefix(self, prefix: str) -> Self:
        '''
        Add a prefix to the root column name of the expression.

        Parameters
        ----------
        prefix
            Prefix to add to the root column name.

        Notes
        -----
        This will undo any previous renaming operations on the expression.

        Due to implementation constraints, this method can only be called as the last
        expression in a chain.

        See Also
        --------
        suffix

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["x", "y", "z"],
        ...     }
        ... )
        >>> df.with_columns(pl.all().reverse().prefix("reverse_"))
        shape: (3, 4)
        ┌─────┬─────┬───────────┬───────────┐
        │ a   ┆ b   ┆ reverse_a ┆ reverse_b │
        │ --- ┆ --- ┆ ---       ┆ ---       │
        │ i64 ┆ str ┆ i64       ┆ str       │
        ╞═════╪═════╪═══════════╪═══════════╡
        │ 1   ┆ x   ┆ 3         ┆ z         │
        │ 2   ┆ y   ┆ 2         ┆ y         │
        │ 3   ┆ z   ┆ 1         ┆ x         │
        └─────┴─────┴───────────┴───────────┘

        '''
    def suffix(self, suffix: str) -> Self:
        '''
        Add a suffix to the root column name of the expression.

        Parameters
        ----------
        suffix
            Suffix to add to the root column name.

        Notes
        -----
        This will undo any previous renaming operations on the expression.

        Due to implementation constraints, this method can only be called as the last
        expression in a chain.

        See Also
        --------
        prefix

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": ["x", "y", "z"],
        ...     }
        ... )
        >>> df.with_columns(pl.all().reverse().suffix("_reverse"))
        shape: (3, 4)
        ┌─────┬─────┬───────────┬───────────┐
        │ a   ┆ b   ┆ a_reverse ┆ b_reverse │
        │ --- ┆ --- ┆ ---       ┆ ---       │
        │ i64 ┆ str ┆ i64       ┆ str       │
        ╞═════╪═════╪═══════════╪═══════════╡
        │ 1   ┆ x   ┆ 3         ┆ z         │
        │ 2   ┆ y   ┆ 2         ┆ y         │
        │ 3   ┆ z   ┆ 1         ┆ x         │
        └─────┴─────┴───────────┴───────────┘

        '''
    def keep_name(self) -> Self:
        '''
        Keep the original root name of the expression.

        Notes
        -----
        Due to implementation constraints, this method can only be called as the last
        expression in a chain.

        See Also
        --------
        alias

        Examples
        --------
        Undo an alias operation.

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2],
        ...         "b": [3, 4],
        ...     }
        ... )
        >>> df.with_columns((pl.col("a") * 9).alias("c").keep_name())
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 9   ┆ 3   │
        │ 18  ┆ 4   │
        └─────┴─────┘

        Prevent errors due to duplicate column names.

        >>> df.select((pl.lit(10) / pl.all()).keep_name())
        shape: (2, 2)
        ┌──────┬──────────┐
        │ a    ┆ b        │
        │ ---  ┆ ---      │
        │ f64  ┆ f64      │
        ╞══════╪══════════╡
        │ 10.0 ┆ 3.333333 │
        │ 5.0  ┆ 2.5      │
        └──────┴──────────┘

        '''
    def exclude(self, columns: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType], *more_columns: str | PolarsDataType) -> Self:
        '''
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

        '''
    def pipe(self, function: Callable[Concatenate[Expr, P], T], *args: P.args, **kwargs: P.kwargs) -> T:
        '''
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
        >>> def extract_number(expr: pl.Expr) -> pl.Expr:
        ...     """Extract the digits from a string."""
        ...     return expr.str.extract(r"\\d+", 0).cast(pl.Int64)
        >>>
        >>> def scale_negative_even(expr: pl.Expr, *, n: int = 1) -> pl.Expr:
        ...     """Set even numbers negative, and scale by a user-supplied value."""
        ...     expr = pl.when(expr % 2 == 0).then(-expr).otherwise(expr)
        ...     return expr * n
        >>>
        >>> df = pl.DataFrame({"val": ["a: 1", "b: 2", "c: 3", "d: 4"]})
        >>> df.with_columns(
        ...     udfs=(
        ...         pl.col("val").pipe(extract_number).pipe(scale_negative_even, n=5)
        ...     ),
        ... )
        shape: (4, 2)
        ┌──────┬──────┐
        │ val  ┆ udfs │
        │ ---  ┆ ---  │
        │ str  ┆ i64  │
        ╞══════╪══════╡
        │ a: 1 ┆ 5    │
        │ b: 2 ┆ -10  │
        │ c: 3 ┆ 15   │
        │ d: 4 ┆ -20  │
        └──────┴──────┘

        '''
    def is_not(self) -> Self:
        '''
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

        '''
    def is_null(self) -> Self:
        '''
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

        '''
    def is_not_null(self) -> Self:
        '''
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

        '''
    def is_finite(self) -> Self:
        '''
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

        '''
    def is_infinite(self) -> Self:
        '''
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

        '''
    def is_nan(self) -> Self:
        '''
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

        '''
    def is_not_nan(self) -> Self:
        '''
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

        '''
    def agg_groups(self) -> Self:
        '''
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

        '''
    def count(self) -> Self:
        '''
        Count the number of values in this expression.

        .. warning::
            `null` is deemed a value in this context.

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

        '''
    def len(self) -> Self:
        '''
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

        '''
    def slice(self, offset: int | Expr, length: int | Expr | None = ...) -> Self:
        '''
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

        '''
    def append(self, other: IntoExpr) -> Self:
        '''
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

        '''
    def rechunk(self) -> Self:
        '''
        Create a single chunk of memory for this Series.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})

        Create a Series with 3 nulls, append column a then rechunk

        >>> df.select(pl.repeat(None, 3).append(pl.col("a")).rechunk())
        shape: (6, 1)
        ┌────────┐
        │ repeat │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ null   │
        │ null   │
        │ null   │
        │ 1      │
        │ 1      │
        │ 2      │
        └────────┘

        '''
    def drop_nulls(self) -> Self:
        '''
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

        '''
    def drop_nans(self) -> Self:
        '''
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

        '''
    def cumsum(self) -> Self:
        '''
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

        '''
    def cumprod(self) -> Self:
        '''
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

        '''
    def cummin(self) -> Self:
        '''
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

        '''
    def cummax(self) -> Self:
        '''
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

        '''
    def cumcount(self) -> Self:
        '''
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

        '''
    def floor(self) -> Self:
        '''
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

        '''
    def ceil(self) -> Self:
        '''
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

        '''
    def round(self, decimals: int = ...) -> Self:
        '''
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

        '''
    def dot(self, other: Expr | str) -> Self:
        '''
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

        '''
    def mode(self) -> Self:
        '''
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

        '''
    def cast(self, dtype: PolarsDataType | type[Any]) -> Self:
        '''
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

        '''
    def sort(self) -> Self:
        '''
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

        '''
    def top_k(self, k: int = ...) -> Self:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "value": [1, 98, 2, 3, 99, 4],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("value").top_k().alias("top_k"),
        ...         pl.col("value").bottom_k().alias("bottom_k"),
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

        '''
    def bottom_k(self, k: int = ...) -> Self:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "value": [1, 98, 2, 3, 99, 4],
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("value").top_k().alias("top_k"),
        ...         pl.col("value").bottom_k().alias("bottom_k"),
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

        '''
    def arg_sort(self) -> Self:
        '''
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

        '''
    def arg_max(self) -> Self:
        '''
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

        '''
    def arg_min(self) -> Self:
        '''
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

        '''
    def search_sorted(self, element: Expr | int | float | Series, side: SearchSortedSide = ...) -> Self:
        '''
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

        '''
    def sort_by(self, by: IntoExpr | Iterable[IntoExpr], *more_by: IntoExpr) -> Self:
        '''
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

        '''
    def take(self, indices: int | list[int] | Expr | Series | np.ndarray[Any, Any]) -> Self:
        '''
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

        '''
    def shift(self, periods: int = ...) -> Self:
        '''
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

        '''
    def shift_and_fill(self, fill_value: IntoExpr) -> Self:
        '''
        Shift the values by a given period and fill the resulting null values.

        Parameters
        ----------
        fill_value
            Fill None values with the result of this expression.
        periods
            Number of places to shift (may be negative).

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4]})
        >>> df.select(pl.col("foo").shift_and_fill("a", periods=1))
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

        '''
    def fill_null(self, value: Any | None = ..., strategy: FillNullStrategy | None = ..., limit: int | None = ...) -> Self:
        '''
        Fill null values using the specified value or strategy.

        To interpolate over null values see interpolate.
        See the examples below to fill nulls with an expression.

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
        >>> df.with_columns(pl.col("b").fill_null(strategy="zero"))
        shape: (3, 2)
        ┌──────┬─────┐
        │ a    ┆ b   │
        │ ---  ┆ --- │
        │ i64  ┆ i64 │
        ╞══════╪═════╡
        │ 1    ┆ 4   │
        │ 2    ┆ 0   │
        │ null ┆ 6   │
        └──────┴─────┘
        >>> df.with_columns(pl.col("b").fill_null(99))
        shape: (3, 2)
        ┌──────┬─────┐
        │ a    ┆ b   │
        │ ---  ┆ --- │
        │ i64  ┆ i64 │
        ╞══════╪═════╡
        │ 1    ┆ 4   │
        │ 2    ┆ 99  │
        │ null ┆ 6   │
        └──────┴─────┘
        >>> df.with_columns(pl.col("b").fill_null(strategy="forward"))
        shape: (3, 2)
        ┌──────┬─────┐
        │ a    ┆ b   │
        │ ---  ┆ --- │
        │ i64  ┆ i64 │
        ╞══════╪═════╡
        │ 1    ┆ 4   │
        │ 2    ┆ 4   │
        │ null ┆ 6   │
        └──────┴─────┘
        >>> df.with_columns(pl.col("b").fill_null(pl.col("b").median()))
        shape: (3, 2)
        ┌──────┬─────┐
        │ a    ┆ b   │
        │ ---  ┆ --- │
        │ i64  ┆ f64 │
        ╞══════╪═════╡
        │ 1    ┆ 4.0 │
        │ 2    ┆ 5.0 │
        │ null ┆ 6.0 │
        └──────┴─────┘
        >>> df.with_columns(pl.all().fill_null(pl.all().median()))
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 1.0 ┆ 4.0 │
        │ 2.0 ┆ 5.0 │
        │ 1.5 ┆ 6.0 │
        └─────┴─────┘

        '''
    def fill_nan(self, value: int | float | Expr | None) -> Self:
        '''
        Fill floating point NaN value with a fill value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1.0, None, float("nan")],
        ...         "b": [4.0, float("nan"), 6],
        ...     }
        ... )
        >>> df.with_columns(pl.col("b").fill_nan(0))
        shape: (3, 2)
        ┌──────┬─────┐
        │ a    ┆ b   │
        │ ---  ┆ --- │
        │ f64  ┆ f64 │
        ╞══════╪═════╡
        │ 1.0  ┆ 4.0 │
        │ null ┆ 0.0 │
        │ NaN  ┆ 6.0 │
        └──────┴─────┘

        '''
    def forward_fill(self, limit: int | None = ...) -> Self:
        '''
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

        '''
    def backward_fill(self, limit: int | None = ...) -> Self:
        '''
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
        ...         "c": [None, None, 2],
        ...     }
        ... )
        >>> df.select(pl.all().backward_fill())
        shape: (3, 3)
        ┌──────┬─────┬─────┐
        │ a    ┆ b   ┆ c   │
        │ ---  ┆ --- ┆ --- │
        │ i64  ┆ i64 ┆ i64 │
        ╞══════╪═════╪═════╡
        │ 1    ┆ 4   ┆ 2   │
        │ 2    ┆ 6   ┆ 2   │
        │ null ┆ 6   ┆ 2   │
        └──────┴─────┴─────┘
        >>> df.select(pl.all().backward_fill(limit=1))
        shape: (3, 3)
        ┌──────┬─────┬──────┐
        │ a    ┆ b   ┆ c    │
        │ ---  ┆ --- ┆ ---  │
        │ i64  ┆ i64 ┆ i64  │
        ╞══════╪═════╪══════╡
        │ 1    ┆ 4   ┆ null │
        │ 2    ┆ 6   ┆ 2    │
        │ null ┆ 6   ┆ 2    │
        └──────┴─────┴──────┘

        '''
    def reverse(self) -> Self:
        '''
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

        '''
    def std(self, ddof: int = ...) -> Self:
        '''
        Get standard deviation.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

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

        '''
    def var(self, ddof: int = ...) -> Self:
        '''
        Get variance.

        Parameters
        ----------
        ddof
            “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
            By default ddof is 1.

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

        '''
    def max(self) -> Self:
        '''
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

        '''
    def min(self) -> Self:
        '''
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

        '''
    def nan_max(self) -> Self:
        '''
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

        '''
    def nan_min(self) -> Self:
        '''
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

        '''
    def sum(self) -> Self:
        '''
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

        '''
    def mean(self) -> Self:
        '''
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

        '''
    def median(self) -> Self:
        '''
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

        '''
    def product(self) -> Self:
        '''
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

        '''
    def n_unique(self) -> Self:
        '''
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

        '''
    def approx_unique(self) -> Self:
        '''
        Approx count unique values.

        This is done using the HyperLogLog++ algorithm for cardinality estimation.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 1, 2]})
        >>> df.select(pl.col("a").approx_unique())
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        └─────┘

        '''
    def null_count(self) -> Self:
        '''
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

        '''
    def arg_unique(self) -> Self:
        '''
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

        '''
    def unique(self) -> Self:
        '''
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

        '''
    def first(self) -> Self:
        '''
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

        '''
    def last(self) -> Self:
        '''
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

        '''
    def over(self, expr: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Self:
        '''
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
        mapping_strategy: {\'group_to_rows\', \'join\', \'explode\'}
            - group_to_rows
                If the aggregation results in multiple values, assign them back to their
                position in the DataFrame. This can only be done if the group yields
                the same elements before aggregation as after.
            - join
                Join the groups as \'List<group_dtype>\' to the row positions.
                warning: this can be memory intensive.
            - explode
                Don\'t do any mapping, but simply flatten the group.
                This only makes sense if the input data is sorted.

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

        '''
    def is_unique(self) -> Self:
        '''
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

        '''
    def is_first(self) -> Self:
        '''
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

        '''
    def is_duplicated(self) -> Self:
        '''
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

        '''
    def quantile(self, quantile: float | Expr, interpolation: RollingInterpolationMethod = ...) -> Self:
        '''
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

        '''
    def cut(self, breaks: list[float], labels: list[str] | None = ..., left_closed: bool = ..., include_breaks: bool = ...) -> Self:
        '''
        Bin continuous values into discrete categories.

        Parameters
        ----------
        breaks
            A list of unique cut points.
        labels
            Labels to assign to bins. If given, the length must be len(breaks) + 1.
        left_closed
            Whether intervals should be [) instead of the default of (]
        include_breaks
            Include the the right endpoint of the bin each observation falls in.
            If True, the resulting column will be a Struct.

        Examples
        --------
        >>> g = pl.repeat("a", 5, eager=True).append(pl.repeat("b", 5, eager=True))
        >>> df = pl.DataFrame(dict(g=g, x=range(10)))
        >>> df.with_columns(q=pl.col("x").cut([2, 5]))
        shape: (10, 3)
        ┌─────┬─────┬───────────┐
        │ g   ┆ x   ┆ q         │
        │ --- ┆ --- ┆ ---       │
        │ str ┆ i64 ┆ cat       │
        ╞═════╪═════╪═══════════╡
        │ a   ┆ 0   ┆ (-inf, 2] │
        │ a   ┆ 1   ┆ (-inf, 2] │
        │ a   ┆ 2   ┆ (-inf, 2] │
        │ a   ┆ 3   ┆ (2, 5]    │
        │ …   ┆ …   ┆ …         │
        │ b   ┆ 6   ┆ (5, inf]  │
        │ b   ┆ 7   ┆ (5, inf]  │
        │ b   ┆ 8   ┆ (5, inf]  │
        │ b   ┆ 9   ┆ (5, inf]  │
        └─────┴─────┴───────────┘
        >>> df.with_columns(q=pl.col("x").cut([2, 5], left_closed=True))
        shape: (10, 3)
        ┌─────┬─────┬───────────┐
        │ g   ┆ x   ┆ q         │
        │ --- ┆ --- ┆ ---       │
        │ str ┆ i64 ┆ cat       │
        ╞═════╪═════╪═══════════╡
        │ a   ┆ 0   ┆ [-inf, 2) │
        │ a   ┆ 1   ┆ [-inf, 2) │
        │ a   ┆ 2   ┆ [2, 5)    │
        │ a   ┆ 3   ┆ [2, 5)    │
        │ …   ┆ …   ┆ …         │
        │ b   ┆ 6   ┆ [5, inf)  │
        │ b   ┆ 7   ┆ [5, inf)  │
        │ b   ┆ 8   ┆ [5, inf)  │
        │ b   ┆ 9   ┆ [5, inf)  │
        └─────┴─────┴───────────┘
        '''
    def qcut(self, *args, **kwargs) -> Self:
        '''
        Bin continuous values into discrete categories based on their quantiles.

        Parameters
        ----------
        q
            Either a list of quantile probabilities between 0 and 1 or a positive
            integer determining the number of evenly spaced probabilities to use.
        labels
            Labels to assign to bins. If given, the length must be len(probs) + 1.
            If computing over groups this must be set for now.
        left_closed
            Whether intervals should be [) instead of the default of (]
        allow_duplicates
            If True, the resulting quantile breaks don\'t have to be unique. This can
            happen even with unique probs depending on the data. Duplicates will be
            dropped, resulting in fewer bins.
        include_breaks
            Include the the right endpoint of the bin each observation falls in.
            If True, the resulting column will be a Struct.


        Examples
        --------
        >>> g = pl.repeat("a", 5, eager=True).append(pl.repeat("b", 5, eager=True))
        >>> df = pl.DataFrame(dict(g=g, x=range(10)))
        >>> df.with_columns(q=pl.col("x").qcut([0.5]))
        shape: (10, 3)
        ┌─────┬─────┬─────────────┐
        │ g   ┆ x   ┆ q           │
        │ --- ┆ --- ┆ ---         │
        │ str ┆ i64 ┆ cat         │
        ╞═════╪═════╪═════════════╡
        │ a   ┆ 0   ┆ (-inf, 4.5] │
        │ a   ┆ 1   ┆ (-inf, 4.5] │
        │ a   ┆ 2   ┆ (-inf, 4.5] │
        │ a   ┆ 3   ┆ (-inf, 4.5] │
        │ …   ┆ …   ┆ …           │
        │ b   ┆ 6   ┆ (4.5, inf]  │
        │ b   ┆ 7   ┆ (4.5, inf]  │
        │ b   ┆ 8   ┆ (4.5, inf]  │
        │ b   ┆ 9   ┆ (4.5, inf]  │
        └─────┴─────┴─────────────┘
        >>> df.with_columns(q=pl.col("x").qcut(2))
        shape: (10, 3)
        ┌─────┬─────┬─────────────┐
        │ g   ┆ x   ┆ q           │
        │ --- ┆ --- ┆ ---         │
        │ str ┆ i64 ┆ cat         │
        ╞═════╪═════╪═════════════╡
        │ a   ┆ 0   ┆ (-inf, 4.5] │
        │ a   ┆ 1   ┆ (-inf, 4.5] │
        │ a   ┆ 2   ┆ (-inf, 4.5] │
        │ a   ┆ 3   ┆ (-inf, 4.5] │
        │ …   ┆ …   ┆ …           │
        │ b   ┆ 6   ┆ (4.5, inf]  │
        │ b   ┆ 7   ┆ (4.5, inf]  │
        │ b   ┆ 8   ┆ (4.5, inf]  │
        │ b   ┆ 9   ┆ (4.5, inf]  │
        └─────┴─────┴─────────────┘
        >>> df.with_columns(q=pl.col("x").qcut([0.5], ["lo", "hi"]).over("g"))
        shape: (10, 3)
        ┌─────┬─────┬─────┐
        │ g   ┆ x   ┆ q   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ cat │
        ╞═════╪═════╪═════╡
        │ a   ┆ 0   ┆ lo  │
        │ a   ┆ 1   ┆ lo  │
        │ a   ┆ 2   ┆ lo  │
        │ a   ┆ 3   ┆ hi  │
        │ …   ┆ …   ┆ …   │
        │ b   ┆ 6   ┆ lo  │
        │ b   ┆ 7   ┆ lo  │
        │ b   ┆ 8   ┆ hi  │
        │ b   ┆ 9   ┆ hi  │
        └─────┴─────┴─────┘
        >>> df.with_columns(q=pl.col("x").qcut([0.5], ["lo", "hi"], True).over("g"))
        shape: (10, 3)
        ┌─────┬─────┬─────┐
        │ g   ┆ x   ┆ q   │
        │ --- ┆ --- ┆ --- │
        │ str ┆ i64 ┆ cat │
        ╞═════╪═════╪═════╡
        │ a   ┆ 0   ┆ lo  │
        │ a   ┆ 1   ┆ lo  │
        │ a   ┆ 2   ┆ hi  │
        │ a   ┆ 3   ┆ hi  │
        │ …   ┆ …   ┆ …   │
        │ b   ┆ 6   ┆ lo  │
        │ b   ┆ 7   ┆ hi  │
        │ b   ┆ 8   ┆ hi  │
        │ b   ┆ 9   ┆ hi  │
        └─────┴─────┴─────┘
        >>> df.with_columns(q=pl.col("x").qcut([0.25, 0.5], include_breaks=True))
        shape: (10, 3)
        ┌─────┬─────┬───────────────────────┐
        │ g   ┆ x   ┆ q                     │
        │ --- ┆ --- ┆ ---                   │
        │ str ┆ i64 ┆ struct[2]             │
        ╞═════╪═════╪═══════════════════════╡
        │ a   ┆ 0   ┆ {2.25,"(-inf, 2.25]"} │
        │ a   ┆ 1   ┆ {2.25,"(-inf, 2.25]"} │
        │ a   ┆ 2   ┆ {2.25,"(-inf, 2.25]"} │
        │ a   ┆ 3   ┆ {4.5,"(2.25, 4.5]"}   │
        │ …   ┆ …   ┆ …                     │
        │ b   ┆ 6   ┆ {inf,"(4.5, inf]"}    │
        │ b   ┆ 7   ┆ {inf,"(4.5, inf]"}    │
        │ b   ┆ 8   ┆ {inf,"(4.5, inf]"}    │
        │ b   ┆ 9   ┆ {inf,"(4.5, inf]"}    │
        └─────┴─────┴───────────────────────┘
        '''
    def rle(self) -> Self:
        '''
        Get the lengths of runs of identical values.

        Returns
        -------
            A Struct Series containing "lengths" and "values" Fields

        Examples
        --------
        >>> df = pl.DataFrame(pl.Series("s", [1, 1, 2, 1, None, 1, 3, 3]))
        >>> df.select(pl.col("s").rle()).unnest("s")
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
    def rle_id(self) -> Self:
        '''
        Map values to run IDs.

        Similar to RLE, but it maps each value to an ID corresponding to the run into
        which it falls. This is especially useful when you want to define groups by
        runs of identical values rather than the values themselves.


        Examples
        --------
        >>> df = pl.DataFrame(dict(a=[1, 2, 1, 1, 1], b=["x", "x", None, "y", "y"]))
        >>> # It works on structs of multiple values too!
        >>> df.with_columns(a_r=pl.col("a").rle_id(), ab_r=pl.struct("a", "b").rle_id())
        shape: (5, 4)
        ┌─────┬──────┬─────┬──────┐
        │ a   ┆ b    ┆ a_r ┆ ab_r │
        │ --- ┆ ---  ┆ --- ┆ ---  │
        │ i64 ┆ str  ┆ u32 ┆ u32  │
        ╞═════╪══════╪═════╪══════╡
        │ 1   ┆ x    ┆ 0   ┆ 0    │
        │ 2   ┆ x    ┆ 1   ┆ 1    │
        │ 1   ┆ null ┆ 2   ┆ 2    │
        │ 1   ┆ y    ┆ 2   ┆ 3    │
        │ 1   ┆ y    ┆ 2   ┆ 3    │
        └─────┴──────┴─────┴──────┘
        '''
    def filter(self, predicate: Expr) -> Self:
        '''
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
        ┌───────────┬─────┬─────┐
        │ group_col ┆ lt  ┆ gte │
        │ ---       ┆ --- ┆ --- │
        │ str       ┆ i64 ┆ i64 │
        ╞═══════════╪═════╪═════╡
        │ g1        ┆ 1   ┆ 2   │
        │ g2        ┆ 0   ┆ 3   │
        └───────────┴─────┴─────┘

        '''
    def where(self, predicate: Expr) -> Self:
        '''
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
        ┌───────────┬─────┬─────┐
        │ group_col ┆ lt  ┆ gte │
        │ ---       ┆ --- ┆ --- │
        │ str       ┆ i64 ┆ i64 │
        ╞═══════════╪═════╪═════╡
        │ g1        ┆ 1   ┆ 2   │
        │ g2        ┆ 0   ┆ 3   │
        └───────────┴─────┴─────┘

        '''
    def map(self, function: Callable[[Series], Series | Any], return_dtype: PolarsDataType | None = ...) -> Self:
        '''
        Apply a custom python function to a Series or sequence of Series.

        The output of this custom function must be a Series.
        If you want to apply a custom function elementwise over single values, see
        :func:`apply`. A use case for ``map`` is when you want to transform an
        expression with a third-party library.

        Read more in `the book
        <https://pola-rs.github.io/polars-book/user-guide/expressions/user-defined-functions>`_.

        Parameters
        ----------
        function
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
        agg_list
            Aggregate list

        Warnings
        --------
        If ``return_dtype`` is not provided, this may lead to unexpected results.
        We allow this, but it is considered a bug in the user\'s query.

        See Also
        --------
        map_dict

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

        '''
    def apply(self, function: Callable[[Series], Series] | Callable[[Any], Any], return_dtype: PolarsDataType | None = ...) -> Self:
        '''
        Apply a custom/user-defined function (UDF) in a GroupBy or Projection context.

        .. warning::
            This method is much slower than the native expressions API.
            Only use it if you cannot implement your logic otherwise.

        Depending on the context it has the following behavior:

        * Selection
            Expects `f` to be of type Callable[[Any], Any].
            Applies a python function over each individual value in the column.
        * GroupBy
            Expects `f` to be of type Callable[[Series], Series].
            Applies a python function over each group.

        Parameters
        ----------
        function
            Lambda/ function to apply.
        return_dtype
            Dtype of the output Series.
            If not set, the dtype will be
            ``polars.Unknown``.
        skip_nulls
            Don\'t apply the function over values
            that contain nulls. This is faster.
        pass_name
            Pass the Series name to the custom function
            This is more expensive.
        strategy : {\'thread_local\', \'threading\'}
            This functionality is in `alpha` stage. This may be removed
            /changed without it being considered a breaking change.

            - \'thread_local\': run the python function on a single thread.
            - \'threading\': run the python function on separate threads. Use with
                        care as this can slow performance. This might only speed up
                        your code if the amount of work per element is significant
                        and the python function releases the GIL (e.g. via calling
                        a c function)

        Notes
        -----
        * Using ``apply`` is strongly discouraged as you will be effectively running
          python "for" loops. This will be very slow. Wherever possible you should
          strongly prefer the native expression API to achieve the best performance.

        * If your function is expensive and you don\'t want it to be called more than
          once for a given input, consider applying an ``@lru_cache`` decorator to it.
          With suitable data you may achieve order-of-magnitude speedups (or more).

        Warnings
        --------
        If ``return_dtype`` is not provided, this may lead to unexpected results.
        We allow this, but it is considered a bug in the user\'s query.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3, 1],
        ...         "b": ["a", "b", "c", "c"],
        ...     }
        ... )

        In a selection context, the function is applied by row.

        >>> df.with_columns(  # doctest: +SKIP
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

        '''
    def flatten(self) -> Self:
        '''
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

        '''
    def explode(self) -> Self:
        '''
        Explode a list Series.

        This means that every item is expanded to a new row.

        Returns
        -------
        Exploded Series of same dtype

        See Also
        --------
        Expr.list.explode : Explode a list column.
        Expr.str.explode : Explode a string column.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "group": ["a", "b"],
        ...         "values": [
        ...             [1, 2],
        ...             [3, 4],
        ...         ],
        ...     }
        ... )
        >>> df.select(pl.col("values").explode())
        shape: (4, 1)
        ┌────────┐
        │ values │
        │ ---    │
        │ i64    │
        ╞════════╡
        │ 1      │
        │ 2      │
        │ 3      │
        │ 4      │
        └────────┘

        '''
    def implode(self) -> Self:
        '''
        Aggregate values into a list.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [1, 2, 3],
        ...         "b": [4, 5, 6],
        ...     }
        ... )
        >>> df.select(pl.all().implode())
        shape: (1, 2)
        ┌───────────┬───────────┐
        │ a         ┆ b         │
        │ ---       ┆ ---       │
        │ list[i64] ┆ list[i64] │
        ╞═══════════╪═══════════╡
        │ [1, 2, 3] ┆ [4, 5, 6] │
        └───────────┴───────────┘

        '''
    def take_every(self, n: int) -> Self:
        '''
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

        '''
    def head(self, n: int | Expr = ...) -> Self:
        '''
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

        '''
    def tail(self, n: int | Expr = ...) -> Self:
        '''
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

        '''
    def limit(self, n: int | Expr = ...) -> Self:
        '''
        Get the first `n` rows (alias for :func:`Expr.head`).

        Parameters
        ----------
        n
            Number of rows to return.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, 2, 3, 4, 5, 6, 7]})
        >>> df.limit(3)
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

        '''
    def and_(self, *others: Any) -> Self:
        '''
        Method equivalent of bitwise "and" operator ``expr & other & ...``.

        Parameters
        ----------
        *others
            One or more integer or boolean expressions to evaluate/combine.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [5, 6, 7, 4, 8],
        ...         "y": [1.5, 2.5, 1.0, 4.0, -5.75],
        ...         "z": [-9, 2, -1, 4, 8],
        ...     }
        ... )
        >>> df.select(
        ...     (pl.col("x") >= pl.col("z"))
        ...     .and_(
        ...         pl.col("y") >= pl.col("z"),
        ...         pl.col("y") == pl.col("y"),
        ...         pl.col("z") <= pl.col("x"),
        ...         pl.col("y") != pl.col("x"),
        ...     )
        ...     .alias("all")
        ... )
        shape: (5, 1)
        ┌───────┐
        │ all   │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ true  │
        │ true  │
        │ true  │
        │ false │
        │ false │
        └───────┘

        '''
    def or_(self, *others: Any) -> Self:
        '''
        Method equivalent of bitwise "or" operator ``expr | other | ...``.

        Parameters
        ----------
        *others
            One or more integer or boolean expressions to evaluate/combine.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [5, 6, 7, 4, 8],
        ...         "y": [1.5, 2.5, 1.0, 4.0, -5.75],
        ...         "z": [-9, 2, -1, 4, 8],
        ...     }
        ... )
        >>> df.select(
        ...     (pl.col("x") == pl.col("y"))
        ...     .or_(
        ...         pl.col("x") == pl.col("y"),
        ...         pl.col("y") == pl.col("z"),
        ...         pl.col("y").cast(int) == pl.col("z"),
        ...     )
        ...     .alias("any")
        ... )
        shape: (5, 1)
        ┌───────┐
        │ any   │
        │ ---   │
        │ bool  │
        ╞═══════╡
        │ false │
        │ true  │
        │ false │
        │ true  │
        │ false │
        └───────┘

        '''
    def eq(self, other: Any) -> Self:
        '''
        Method equivalent of equality operator ``expr == other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [1.0, 2.0, float("nan"), 4.0],
        ...         "y": [2.0, 2.0, float("nan"), 4.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").eq(pl.col("y")).alias("x == y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬────────┐
        │ x   ┆ y   ┆ x == y │
        │ --- ┆ --- ┆ ---    │
        │ f64 ┆ f64 ┆ bool   │
        ╞═════╪═════╪════════╡
        │ 1.0 ┆ 2.0 ┆ false  │
        │ 2.0 ┆ 2.0 ┆ true   │
        │ NaN ┆ NaN ┆ false  │
        │ 4.0 ┆ 4.0 ┆ true   │
        └─────┴─────┴────────┘

        '''
    def eq_missing(self, other: Any) -> Self:
        '''
        Method equivalent of equality operator ``expr == other`` where `None` == None`.

        This differs from default ``eq`` where null values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [1.0, 2.0, float("nan"), 4.0, None, None],
        ...         "y": [2.0, 2.0, float("nan"), 4.0, 5.0, None],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").eq_missing(pl.col("y")).alias("x == y"),
        ... )
        shape: (6, 3)
        ┌──────┬──────┬────────┐
        │ x    ┆ y    ┆ x == y │
        │ ---  ┆ ---  ┆ ---    │
        │ f64  ┆ f64  ┆ bool   │
        ╞══════╪══════╪════════╡
        │ 1.0  ┆ 2.0  ┆ false  │
        │ 2.0  ┆ 2.0  ┆ true   │
        │ NaN  ┆ NaN  ┆ false  │
        │ 4.0  ┆ 4.0  ┆ true   │
        │ null ┆ 5.0  ┆ false  │
        │ null ┆ null ┆ true   │
        └──────┴──────┴────────┘

        '''
    def ge(self, other: Any) -> Self:
        '''
        Method equivalent of "greater than or equal" operator ``expr >= other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [5.0, 4.0, float("nan"), 2.0],
        ...         "y": [5.0, 3.0, float("nan"), 1.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").ge(pl.col("y")).alias("x >= y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬────────┐
        │ x   ┆ y   ┆ x >= y │
        │ --- ┆ --- ┆ ---    │
        │ f64 ┆ f64 ┆ bool   │
        ╞═════╪═════╪════════╡
        │ 5.0 ┆ 5.0 ┆ true   │
        │ 4.0 ┆ 3.0 ┆ true   │
        │ NaN ┆ NaN ┆ false  │
        │ 2.0 ┆ 1.0 ┆ true   │
        └─────┴─────┴────────┘

        '''
    def gt(self, other: Any) -> Self:
        '''
        Method equivalent of "greater than" operator ``expr > other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [5.0, 4.0, float("nan"), 2.0],
        ...         "y": [5.0, 3.0, float("nan"), 1.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").gt(pl.col("y")).alias("x > y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬───────┐
        │ x   ┆ y   ┆ x > y │
        │ --- ┆ --- ┆ ---   │
        │ f64 ┆ f64 ┆ bool  │
        ╞═════╪═════╪═══════╡
        │ 5.0 ┆ 5.0 ┆ false │
        │ 4.0 ┆ 3.0 ┆ true  │
        │ NaN ┆ NaN ┆ false │
        │ 2.0 ┆ 1.0 ┆ true  │
        └─────┴─────┴───────┘

        '''
    def le(self, other: Any) -> Self:
        '''
        Method equivalent of "less than or equal" operator ``expr <= other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [5.0, 4.0, float("nan"), 0.5],
        ...         "y": [5.0, 3.5, float("nan"), 2.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").le(pl.col("y")).alias("x <= y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬────────┐
        │ x   ┆ y   ┆ x <= y │
        │ --- ┆ --- ┆ ---    │
        │ f64 ┆ f64 ┆ bool   │
        ╞═════╪═════╪════════╡
        │ 5.0 ┆ 5.0 ┆ true   │
        │ 4.0 ┆ 3.5 ┆ false  │
        │ NaN ┆ NaN ┆ false  │
        │ 0.5 ┆ 2.0 ┆ true   │
        └─────┴─────┴────────┘

        '''
    def lt(self, other: Any) -> Self:
        '''
        Method equivalent of "less than" operator ``expr < other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [1.0, 2.0, float("nan"), 3.0],
        ...         "y": [2.0, 2.0, float("nan"), 4.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").lt(pl.col("y")).alias("x < y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬───────┐
        │ x   ┆ y   ┆ x < y │
        │ --- ┆ --- ┆ ---   │
        │ f64 ┆ f64 ┆ bool  │
        ╞═════╪═════╪═══════╡
        │ 1.0 ┆ 2.0 ┆ true  │
        │ 2.0 ┆ 2.0 ┆ false │
        │ NaN ┆ NaN ┆ false │
        │ 3.0 ┆ 4.0 ┆ true  │
        └─────┴─────┴───────┘

        '''
    def ne(self, other: Any) -> Self:
        '''
        Method equivalent of inequality operator ``expr != other``.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [1.0, 2.0, float("nan"), 4.0],
        ...         "y": [2.0, 2.0, float("nan"), 4.0],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").ne(pl.col("y")).alias("x != y"),
        ... )
        shape: (4, 3)
        ┌─────┬─────┬────────┐
        │ x   ┆ y   ┆ x != y │
        │ --- ┆ --- ┆ ---    │
        │ f64 ┆ f64 ┆ bool   │
        ╞═════╪═════╪════════╡
        │ 1.0 ┆ 2.0 ┆ true   │
        │ 2.0 ┆ 2.0 ┆ false  │
        │ NaN ┆ NaN ┆ true   │
        │ 4.0 ┆ 4.0 ┆ false  │
        └─────┴─────┴────────┘

        '''
    def ne_missing(self, other: Any) -> Self:
        '''
        Method equivalent of equality operator ``expr != other`` where `None` == None`.

        This differs from default ``ne`` where null values are propagated.

        Parameters
        ----------
        other
            A literal or expression value to compare with.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={
        ...         "x": [1.0, 2.0, float("nan"), 4.0, None, None],
        ...         "y": [2.0, 2.0, float("nan"), 4.0, 5.0, None],
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("x").ne_missing(pl.col("y")).alias("x != y"),
        ... )
        shape: (6, 3)
        ┌──────┬──────┬────────┐
        │ x    ┆ y    ┆ x != y │
        │ ---  ┆ ---  ┆ ---    │
        │ f64  ┆ f64  ┆ bool   │
        ╞══════╪══════╪════════╡
        │ 1.0  ┆ 2.0  ┆ true   │
        │ 2.0  ┆ 2.0  ┆ false  │
        │ NaN  ┆ NaN  ┆ true   │
        │ 4.0  ┆ 4.0  ┆ false  │
        │ null ┆ 5.0  ┆ true   │
        │ null ┆ null ┆ false  │
        └──────┴──────┴────────┘

        '''
    def add(self, other: Any) -> Self:
        '''
        Method equivalent of addition operator ``expr + other``.

        Parameters
        ----------
        other
            numeric or string value; accepts expression input.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        >>> df.with_columns(
        ...     pl.col("x").add(2).alias("x+int"),
        ...     pl.col("x").add(pl.col("x").cumprod()).alias("x+expr"),
        ... )
        shape: (5, 3)
        ┌─────┬───────┬────────┐
        │ x   ┆ x+int ┆ x+expr │
        │ --- ┆ ---   ┆ ---    │
        │ i64 ┆ i64   ┆ i64    │
        ╞═════╪═══════╪════════╡
        │ 1   ┆ 3     ┆ 2      │
        │ 2   ┆ 4     ┆ 4      │
        │ 3   ┆ 5     ┆ 9      │
        │ 4   ┆ 6     ┆ 28     │
        │ 5   ┆ 7     ┆ 125    │
        └─────┴───────┴────────┘

        >>> df = pl.DataFrame(
        ...     {"x": ["a", "d", "g"], "y": ["b", "e", "h"], "z": ["c", "f", "i"]}
        ... )
        >>> df.with_columns(pl.col("x").add(pl.col("y")).add(pl.col("z")).alias("xyz"))
        shape: (3, 4)
        ┌─────┬─────┬─────┬─────┐
        │ x   ┆ y   ┆ z   ┆ xyz │
        │ --- ┆ --- ┆ --- ┆ --- │
        │ str ┆ str ┆ str ┆ str │
        ╞═════╪═════╪═════╪═════╡
        │ a   ┆ b   ┆ c   ┆ abc │
        │ d   ┆ e   ┆ f   ┆ def │
        │ g   ┆ h   ┆ i   ┆ ghi │
        └─────┴─────┴─────┴─────┘

        '''
    def floordiv(self, other: Any) -> Self:
        '''
        Method equivalent of integer division operator ``expr // other``.

        Parameters
        ----------
        other
            Numeric literal or expression value.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        >>> df.with_columns(
        ...     pl.col("x").truediv(2).alias("x/2"),
        ...     pl.col("x").floordiv(2).alias("x//2"),
        ... )
        shape: (5, 3)
        ┌─────┬─────┬──────┐
        │ x   ┆ x/2 ┆ x//2 │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ f64 ┆ i64  │
        ╞═════╪═════╪══════╡
        │ 1   ┆ 0.5 ┆ 0    │
        │ 2   ┆ 1.0 ┆ 1    │
        │ 3   ┆ 1.5 ┆ 1    │
        │ 4   ┆ 2.0 ┆ 2    │
        │ 5   ┆ 2.5 ┆ 2    │
        └─────┴─────┴──────┘

        See Also
        --------
        truediv

        '''
    def mod(self, other: Any) -> Self:
        '''
        Method equivalent of modulus operator ``expr % other``.

        Parameters
        ----------
        other
            Numeric literal or expression value.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [0, 1, 2, 3, 4]})
        >>> df.with_columns(pl.col("x").mod(2).alias("x%2"))
        shape: (5, 2)
        ┌─────┬─────┐
        │ x   ┆ x%2 │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 0   ┆ 0   │
        │ 1   ┆ 1   │
        │ 2   ┆ 0   │
        │ 3   ┆ 1   │
        │ 4   ┆ 0   │
        └─────┴─────┘

        '''
    def mul(self, other: Any) -> Self:
        '''
        Method equivalent of multiplication operator ``expr * other``.

        Parameters
        ----------
        other
            Numeric literal or expression value.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [1, 2, 4, 8, 16]})
        >>> df.with_columns(
        ...     pl.col("x").mul(2).alias("x*2"),
        ...     pl.col("x").mul(pl.col("x").log(2)).alias("x * xlog2"),
        ... )
        shape: (5, 3)
        ┌─────┬─────┬───────────┐
        │ x   ┆ x*2 ┆ x * xlog2 │
        │ --- ┆ --- ┆ ---       │
        │ i64 ┆ i64 ┆ f64       │
        ╞═════╪═════╪═══════════╡
        │ 1   ┆ 2   ┆ 0.0       │
        │ 2   ┆ 4   ┆ 2.0       │
        │ 4   ┆ 8   ┆ 8.0       │
        │ 8   ┆ 16  ┆ 24.0      │
        │ 16  ┆ 32  ┆ 64.0      │
        └─────┴─────┴───────────┘

        '''
    def sub(self, other: Any) -> Self:
        '''
        Method equivalent of subtraction operator ``expr - other``.

        Parameters
        ----------
        other
            Numeric literal or expression value.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [0, 1, 2, 3, 4]})
        >>> df.with_columns(
        ...     pl.col("x").sub(2).alias("x-2"),
        ...     pl.col("x").sub(pl.col("x").cumsum()).alias("x-expr"),
        ... )
        shape: (5, 3)
        ┌─────┬─────┬────────┐
        │ x   ┆ x-2 ┆ x-expr │
        │ --- ┆ --- ┆ ---    │
        │ i64 ┆ i64 ┆ i64    │
        ╞═════╪═════╪════════╡
        │ 0   ┆ -2  ┆ 0      │
        │ 1   ┆ -1  ┆ 0      │
        │ 2   ┆ 0   ┆ -1     │
        │ 3   ┆ 1   ┆ -3     │
        │ 4   ┆ 2   ┆ -6     │
        └─────┴─────┴────────┘

        '''
    def truediv(self, other: Any) -> Self:
        '''
        Method equivalent of float division operator ``expr / other``.

        Parameters
        ----------
        other
            Numeric literal or expression value.

        Notes
        -----
        Zero-division behaviour follows IEEE-754:

        0/0: Invalid operation - mathematically undefined, returns NaN.
        n/0: On finite operands gives an exact infinite result, eg: ±infinity.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     data={"x": [-2, -1, 0, 1, 2], "y": [0.5, 0.0, 0.0, -4.0, -0.5]}
        ... )
        >>> df.with_columns(
        ...     pl.col("x").truediv(2).alias("x/2"),
        ...     pl.col("x").truediv(pl.col("y")).alias("x/y"),
        ... )
        shape: (5, 4)
        ┌─────┬──────┬──────┬───────┐
        │ x   ┆ y    ┆ x/2  ┆ x/y   │
        │ --- ┆ ---  ┆ ---  ┆ ---   │
        │ i64 ┆ f64  ┆ f64  ┆ f64   │
        ╞═════╪══════╪══════╪═══════╡
        │ -2  ┆ 0.5  ┆ -1.0 ┆ -4.0  │
        │ -1  ┆ 0.0  ┆ -0.5 ┆ -inf  │
        │ 0   ┆ 0.0  ┆ 0.0  ┆ NaN   │
        │ 1   ┆ -4.0 ┆ 0.5  ┆ -0.25 │
        │ 2   ┆ -0.5 ┆ 1.0  ┆ -4.0  │
        └─────┴──────┴──────┴───────┘

        See Also
        --------
        floordiv

        '''
    def pow(self, exponent: int | float | None | Series | Expr) -> Self:
        '''
        Method equivalent of exponentiation operator ``expr ** exponent``.

        Parameters
        ----------
        exponent
            Numeric literal or expression exponent value.

        Examples
        --------
        >>> df = pl.DataFrame({"x": [1, 2, 4, 8]})
        >>> df.with_columns(
        ...     pl.col("x").pow(3).alias("cube"),
        ...     pl.col("x").pow(pl.col("x").log(2)).alias("x ** xlog2"),
        ... )
        shape: (4, 3)
        ┌─────┬───────┬────────────┐
        │ x   ┆ cube  ┆ x ** xlog2 │
        │ --- ┆ ---   ┆ ---        │
        │ i64 ┆ f64   ┆ f64        │
        ╞═════╪═══════╪════════════╡
        │ 1   ┆ 1.0   ┆ 1.0        │
        │ 2   ┆ 8.0   ┆ 2.0        │
        │ 4   ┆ 64.0  ┆ 16.0       │
        │ 8   ┆ 512.0 ┆ 512.0      │
        └─────┴───────┴────────────┘

        '''
    def xor(self, other: Any) -> Self:
        '''
        Method equivalent of bitwise exclusive-or operator ``expr ^ other``.

        Parameters
        ----------
        other
            Integer or boolean value; accepts expression input.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"x": [True, False, True, False], "y": [True, True, False, False]}
        ... )
        >>> df.with_columns(pl.col("x").xor(pl.col("y")).alias("x ^ y"))
        shape: (4, 3)
        ┌───────┬───────┬───────┐
        │ x     ┆ y     ┆ x ^ y │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ true  ┆ true  ┆ false │
        │ false ┆ true  ┆ true  │
        │ true  ┆ false ┆ true  │
        │ false ┆ false ┆ false │
        └───────┴───────┴───────┘

        >>> def binary_string(n: int) -> str:
        ...     return bin(n)[2:].zfill(8)
        >>>
        >>> df = pl.DataFrame(
        ...     data={"x": [10, 8, 250, 66], "y": [1, 2, 3, 4]},
        ...     schema={"x": pl.UInt8, "y": pl.UInt8},
        ... )
        >>> df.with_columns(
        ...     pl.col("x").apply(binary_string).alias("bin_x"),
        ...     pl.col("y").apply(binary_string).alias("bin_y"),
        ...     pl.col("x").xor(pl.col("y")).alias("xor_xy"),
        ...     pl.col("x").xor(pl.col("y")).apply(binary_string).alias("bin_xor_xy"),
        ... )
        shape: (4, 6)
        ┌─────┬─────┬──────────┬──────────┬────────┬────────────┐
        │ x   ┆ y   ┆ bin_x    ┆ bin_y    ┆ xor_xy ┆ bin_xor_xy │
        │ --- ┆ --- ┆ ---      ┆ ---      ┆ ---    ┆ ---        │
        │ u8  ┆ u8  ┆ str      ┆ str      ┆ u8     ┆ str        │
        ╞═════╪═════╪══════════╪══════════╪════════╪════════════╡
        │ 10  ┆ 1   ┆ 00001010 ┆ 00000001 ┆ 11     ┆ 00001011   │
        │ 8   ┆ 2   ┆ 00001000 ┆ 00000010 ┆ 10     ┆ 00001010   │
        │ 250 ┆ 3   ┆ 11111010 ┆ 00000011 ┆ 249    ┆ 11111001   │
        │ 66  ┆ 4   ┆ 01000010 ┆ 00000100 ┆ 70     ┆ 01000110   │
        └─────┴─────┴──────────┴──────────┴────────┴────────────┘

        '''
    def is_in(self, other: Expr | Collection[Any] | Series) -> Self:
        '''
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

        '''
    def repeat_by(self, by: pl.Series | Expr | str | int) -> Self:
        '''
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

        '''
    def is_between(self, lower_bound: IntoExpr, upper_bound: IntoExpr, closed: ClosedInterval = ...) -> Self:
        '''
        Check if this expression is between the given start and end values.

        Parameters
        ----------
        lower_bound
            Lower bound value. Accepts expression input. Strings are parsed as column
            names, other non-expression inputs are parsed as literals.
        upper_bound
            Upper bound value. Accepts expression input. Strings are parsed as column
            names, other non-expression inputs are parsed as literals.
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

        '''
    def hash(self, seed: int = ..., seed_1: int | None = ..., seed_2: int | None = ..., seed_3: int | None = ...) -> Self:
        '''
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

        '''
    def reinterpret(self) -> Self:
        '''
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

        '''
    def inspect(self, fmt: str = ...) -> Self:
        '''
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

        '''
    def interpolate(self, method: InterpolationMethod = ...) -> Self:
        '''
        Fill null values using interpolation.

        Parameters
        ----------
        method : {\'linear\', \'nearest\'}
            Interpolation method.

        Examples
        --------
        Fill null values using linear interpolation.

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

        Fill null values using nearest interpolation.

        >>> df.select(pl.all().interpolate("nearest"))
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ f64 │
        ╞═════╪═════╡
        │ 1   ┆ 1.0 │
        │ 3   ┆ NaN │
        │ 3   ┆ 3.0 │
        └─────┴─────┘

        Regrid data to a new grid.

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

        '''
    def rolling_min(self, *args, **kwargs) -> Self:
        '''
        Apply a rolling min (moving min) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        If ``by`` has not been specified (the default), the window at a given row will
        include the row itself, and the `window_size - 1` elements before it.

        If you pass a ``by`` column ``<t_0, t_1, ..., t_n>``, then `closed="left"`
        means the windows will be:

            - [t_0 - window_size, t_0)
            - [t_1 - window_size, t_1)
            - ...
            - [t_n - window_size, t_n)

        With `closed="right"`, the left endpoint is not included and the right
        endpoint is included.

        Parameters
        ----------
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic
            temporal size indicated by a timedelta or the following string language:

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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

            By "calendar day", we mean the corresponding time on the next day
            (which may not be 24 hours, due to daylight savings). Similarly for
            "calendar week", "calendar month", "calendar quarter", and
            "calendar year".

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
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype Datetime.
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive); only
            applicable if `by` has been set.

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
        >>> df.with_columns(
        ...     rolling_min=pl.col("A").rolling_min(window_size=2),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_min │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 1.0         │
        │ 3.0 ┆ 2.0         │
        │ 4.0 ┆ 3.0         │
        │ 5.0 ┆ 4.0         │
        │ 6.0 ┆ 5.0         │
        └─────┴─────────────┘

        Specify weights to multiply the values in the window with:

        >>> df.with_columns(
        ...     rolling_min=pl.col("A").rolling_min(
        ...         window_size=2, weights=[0.25, 0.75]
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_min │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 0.25        │
        │ 3.0 ┆ 0.5         │
        │ 4.0 ┆ 0.75        │
        │ 5.0 ┆ 1.0         │
        │ 6.0 ┆ 1.25        │
        └─────┴─────────────┘

        Center the values in the window

        >>> df.with_columns(
        ...     rolling_min=pl.col("A").rolling_min(window_size=3, center=True),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_min │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 1.0         │
        │ 3.0 ┆ 2.0         │
        │ 4.0 ┆ 3.0         │
        │ 5.0 ┆ 4.0         │
        │ 6.0 ┆ null        │
        └─────┴─────────────┘

        Create a DataFrame with a datetime column and a row number column

        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> df_temporal = pl.DataFrame(
        ...     {"date": pl.date_range(start, stop, "1h", eager=True)}
        ... ).with_row_count()
        >>> df_temporal
        shape: (25, 2)
        ┌────────┬─────────────────────┐
        │ row_nr ┆ date                │
        │ ---    ┆ ---                 │
        │ u32    ┆ datetime[μs]        │
        ╞════════╪═════════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 │
        │ 1      ┆ 2001-01-01 01:00:00 │
        │ 2      ┆ 2001-01-01 02:00:00 │
        │ 3      ┆ 2001-01-01 03:00:00 │
        │ …      ┆ …                   │
        │ 21     ┆ 2001-01-01 21:00:00 │
        │ 22     ┆ 2001-01-01 22:00:00 │
        │ 23     ┆ 2001-01-01 23:00:00 │
        │ 24     ┆ 2001-01-02 00:00:00 │
        └────────┴─────────────────────┘
        >>> df_temporal.with_columns(
        ...     rolling_row_min=pl.col("row_nr").rolling_min(
        ...         window_size="2h", by="date", closed="left"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬─────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_min │
        │ ---    ┆ ---                 ┆ ---             │
        │ u32    ┆ datetime[μs]        ┆ u32             │
        ╞════════╪═════════════════════╪═════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ null            │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 0               │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 0               │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 1               │
        │ …      ┆ …                   ┆ …               │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 19              │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 20              │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 21              │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 22              │
        └────────┴─────────────────────┴─────────────────┘

        '''
    def rolling_max(self, *args, **kwargs) -> Self:
        '''
        Apply a rolling max (moving max) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        If ``by`` has not been specified (the default), the window at a given row will
        include the row itself, and the `window_size - 1` elements before it.

        If you pass a ``by`` column ``<t_0, t_1, ..., t_n>``, then `closed="left"`
        means the windows will be:

            - [t_0 - window_size, t_0)
            - [t_1 - window_size, t_1)
            - ...
            - [t_n - window_size, t_n)

        With `closed="right"`, the left endpoint is not included and the right
        endpoint is included.

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
            - 1d    (1 calendar day)
            - 1w    (1 calendar week)
            - 1mo   (1 calendar month)
            - 1q    (1 calendar quarter)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

            By "calendar day", we mean the corresponding time on the next day
            (which may not be 24 hours, due to daylight savings). Similarly for
            "calendar week", "calendar month", "calendar quarter", and
            "calendar year".

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
            If the `window_size` is temporal, for instance `"5h"` or `"3s"`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype Datetime.
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive); only
            applicable if `by` has been set.

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
        >>> df.with_columns(
        ...     rolling_max=pl.col("A").rolling_max(window_size=2),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_max │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 2.0         │
        │ 3.0 ┆ 3.0         │
        │ 4.0 ┆ 4.0         │
        │ 5.0 ┆ 5.0         │
        │ 6.0 ┆ 6.0         │
        └─────┴─────────────┘

        Specify weights to multiply the values in the window with:

        >>> df.with_columns(
        ...     rolling_max=pl.col("A").rolling_max(
        ...         window_size=2, weights=[0.25, 0.75]
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_max │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 1.5         │
        │ 3.0 ┆ 2.25        │
        │ 4.0 ┆ 3.0         │
        │ 5.0 ┆ 3.75        │
        │ 6.0 ┆ 4.5         │
        └─────┴─────────────┘

        Center the values in the window

        >>> df.with_columns(
        ...     rolling_max=pl.col("A").rolling_max(window_size=3, center=True),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_max │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 3.0         │
        │ 3.0 ┆ 4.0         │
        │ 4.0 ┆ 5.0         │
        │ 5.0 ┆ 6.0         │
        │ 6.0 ┆ null        │
        └─────┴─────────────┘

        Create a DataFrame with a datetime column and a row number column

        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> df_temporal = pl.DataFrame(
        ...     {"date": pl.date_range(start, stop, "1h", eager=True)}
        ... ).with_row_count()
        >>> df_temporal
        shape: (25, 2)
        ┌────────┬─────────────────────┐
        │ row_nr ┆ date                │
        │ ---    ┆ ---                 │
        │ u32    ┆ datetime[μs]        │
        ╞════════╪═════════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 │
        │ 1      ┆ 2001-01-01 01:00:00 │
        │ 2      ┆ 2001-01-01 02:00:00 │
        │ 3      ┆ 2001-01-01 03:00:00 │
        │ …      ┆ …                   │
        │ 21     ┆ 2001-01-01 21:00:00 │
        │ 22     ┆ 2001-01-01 22:00:00 │
        │ 23     ┆ 2001-01-01 23:00:00 │
        │ 24     ┆ 2001-01-02 00:00:00 │
        └────────┴─────────────────────┘

        Compute the rolling max with the default left closure of temporal windows

        >>> df_temporal.with_columns(
        ...     rolling_row_max=pl.col("row_nr").rolling_max(
        ...         window_size="2h", by="date", closed="left"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬─────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_max │
        │ ---    ┆ ---                 ┆ ---             │
        │ u32    ┆ datetime[μs]        ┆ u32             │
        ╞════════╪═════════════════════╪═════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ null            │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 0               │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 1               │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 2               │
        │ …      ┆ …                   ┆ …               │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 20              │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 21              │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 22              │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 23              │
        └────────┴─────────────────────┴─────────────────┘

        Compute the rolling max with the closure of windows on both sides

        >>> df_temporal.with_columns(
        ...     rolling_row_max=pl.col("row_nr").rolling_max(
        ...         window_size="2h", by="date", closed="both"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬─────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_max │
        │ ---    ┆ ---                 ┆ ---             │
        │ u32    ┆ datetime[μs]        ┆ u32             │
        ╞════════╪═════════════════════╪═════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ 0               │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 1               │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 2               │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 3               │
        │ …      ┆ …                   ┆ …               │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 21              │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 22              │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 23              │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 24              │
        └────────┴─────────────────────┴─────────────────┘

        '''
    def rolling_mean(self, *args, **kwargs) -> Self:
        '''
        Apply a rolling mean (moving mean) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their mean.

        If ``by`` has not been specified (the default), the window at a given row will
        include the row itself, and the `window_size - 1` elements before it.

        If you pass a ``by`` column ``<t_0, t_1, ..., t_n>``, then `closed="left"`
        means the windows will be:

            - [t_0 - window_size, t_0)
            - [t_1 - window_size, t_1)
            - ...
            - [t_n - window_size, t_n)

        With `closed="right"`, the left endpoint is not included and the right
        endpoint is included.

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
            - 1d    (1 calendar day)
            - 1w    (1 calendar week)
            - 1mo   (1 calendar month)
            - 1q    (1 calendar quarter)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

            By "calendar day", we mean the corresponding time on the next day
            (which may not be 24 hours, due to daylight savings). Similarly for
            "calendar week", "calendar month", "calendar quarter", and
            "calendar year".

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
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype Datetime.
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive); only
            applicable if `by` has been set.

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
        >>> df.with_columns(
        ...     rolling_mean=pl.col("A").rolling_mean(window_size=2),
        ... )
        shape: (6, 2)
        ┌─────┬──────────────┐
        │ A   ┆ rolling_mean │
        │ --- ┆ ---          │
        │ f64 ┆ f64          │
        ╞═════╪══════════════╡
        │ 1.0 ┆ null         │
        │ 2.0 ┆ 1.5          │
        │ 3.0 ┆ 2.5          │
        │ 4.0 ┆ 3.5          │
        │ 5.0 ┆ 4.5          │
        │ 6.0 ┆ 5.5          │
        └─────┴──────────────┘

        Specify weights to multiply the values in the window with:

        >>> df.with_columns(
        ...     rolling_mean=pl.col("A").rolling_mean(
        ...         window_size=2, weights=[0.25, 0.75]
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬──────────────┐
        │ A   ┆ rolling_mean │
        │ --- ┆ ---          │
        │ f64 ┆ f64          │
        ╞═════╪══════════════╡
        │ 1.0 ┆ null         │
        │ 2.0 ┆ 1.75         │
        │ 3.0 ┆ 2.75         │
        │ 4.0 ┆ 3.75         │
        │ 5.0 ┆ 4.75         │
        │ 6.0 ┆ 5.75         │
        └─────┴──────────────┘

        Center the values in the window

        >>> df.with_columns(
        ...     rolling_mean=pl.col("A").rolling_mean(window_size=3, center=True),
        ... )
        shape: (6, 2)
        ┌─────┬──────────────┐
        │ A   ┆ rolling_mean │
        │ --- ┆ ---          │
        │ f64 ┆ f64          │
        ╞═════╪══════════════╡
        │ 1.0 ┆ null         │
        │ 2.0 ┆ 2.0          │
        │ 3.0 ┆ 3.0          │
        │ 4.0 ┆ 4.0          │
        │ 5.0 ┆ 5.0          │
        │ 6.0 ┆ null         │
        └─────┴──────────────┘

        Create a DataFrame with a datetime column and a row number column

        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> df_temporal = pl.DataFrame(
        ...     {"date": pl.date_range(start, stop, "1h", eager=True)}
        ... ).with_row_count()
        >>> df_temporal
        shape: (25, 2)
        ┌────────┬─────────────────────┐
        │ row_nr ┆ date                │
        │ ---    ┆ ---                 │
        │ u32    ┆ datetime[μs]        │
        ╞════════╪═════════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 │
        │ 1      ┆ 2001-01-01 01:00:00 │
        │ 2      ┆ 2001-01-01 02:00:00 │
        │ 3      ┆ 2001-01-01 03:00:00 │
        │ …      ┆ …                   │
        │ 21     ┆ 2001-01-01 21:00:00 │
        │ 22     ┆ 2001-01-01 22:00:00 │
        │ 23     ┆ 2001-01-01 23:00:00 │
        │ 24     ┆ 2001-01-02 00:00:00 │
        └────────┴─────────────────────┘

        Compute the rolling mean with the default left closure of temporal windows

        >>> df_temporal.with_columns(
        ...     rolling_row_mean=pl.col("row_nr").rolling_mean(
        ...         window_size="2h", by="date", closed="left"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬──────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_mean │
        │ ---    ┆ ---                 ┆ ---              │
        │ u32    ┆ datetime[μs]        ┆ f64              │
        ╞════════╪═════════════════════╪══════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ null             │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 0.0              │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 0.5              │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 1.5              │
        │ …      ┆ …                   ┆ …                │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 19.5             │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 20.5             │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 21.5             │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 22.5             │
        └────────┴─────────────────────┴──────────────────┘

        Compute the rolling mean with the closure of windows on both sides

        >>> df_temporal.with_columns(
        ...     rolling_row_mean=pl.col("row_nr").rolling_mean(
        ...         window_size="2h", by="date", closed="both"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬──────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_mean │
        │ ---    ┆ ---                 ┆ ---              │
        │ u32    ┆ datetime[μs]        ┆ f64              │
        ╞════════╪═════════════════════╪══════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ 0.0              │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 0.5              │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 1.0              │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 2.0              │
        │ …      ┆ …                   ┆ …                │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 20.0             │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 21.0             │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 22.0             │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 23.0             │
        └────────┴─────────────────────┴──────────────────┘

        '''
    def rolling_sum(self, *args, **kwargs) -> Self:
        '''
        Apply a rolling sum (moving sum) over the values in this array.

        A window of length `window_size` will traverse the array. The values that fill
        this window will (optionally) be multiplied with the weights given by the
        `weight` vector. The resulting values will be aggregated to their sum.

        If ``by`` has not been specified (the default), the window at a given row will
        include the row itself, and the `window_size - 1` elements before it.

        If you pass a ``by`` column ``<t_0, t_1, ..., t_n>``, then `closed="left"`
        means the windows will be:

            - [t_0 - window_size, t_0)
            - [t_1 - window_size, t_1)
            - ...
            - [t_n - window_size, t_n)

        With `closed="right"`, the left endpoint is not included and the right
        endpoint is included.

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
            - 1d    (1 calendar day)
            - 1w    (1 calendar week)
            - 1mo   (1 calendar month)
            - 1q    (1 calendar quarter)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

            By "calendar day", we mean the corresponding time on the next day
            (which may not be 24 hours, due to daylight savings). Similarly for
            "calendar week", "calendar month", "calendar quarter", and
            "calendar year".

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
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
            set the column that will be used to determine the windows. This column must
            of dtype `{Date, Datetime}`
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive); only
            applicable if `by` has been set.

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
        >>> df.with_columns(
        ...     rolling_sum=pl.col("A").rolling_sum(window_size=2),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_sum │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 3.0         │
        │ 3.0 ┆ 5.0         │
        │ 4.0 ┆ 7.0         │
        │ 5.0 ┆ 9.0         │
        │ 6.0 ┆ 11.0        │
        └─────┴─────────────┘

        Specify weights to multiply the values in the window with:

        >>> df.with_columns(
        ...     rolling_sum=pl.col("A").rolling_sum(
        ...         window_size=2, weights=[0.25, 0.75]
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_sum │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 1.75        │
        │ 3.0 ┆ 2.75        │
        │ 4.0 ┆ 3.75        │
        │ 5.0 ┆ 4.75        │
        │ 6.0 ┆ 5.75        │
        └─────┴─────────────┘

        Center the values in the window

        >>> df.with_columns(
        ...     rolling_sum=pl.col("A").rolling_sum(window_size=3, center=True),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_sum │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 6.0         │
        │ 3.0 ┆ 9.0         │
        │ 4.0 ┆ 12.0        │
        │ 5.0 ┆ 15.0        │
        │ 6.0 ┆ null        │
        └─────┴─────────────┘

        Create a DataFrame with a datetime column and a row number column

        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> df_temporal = pl.DataFrame(
        ...     {"date": pl.date_range(start, stop, "1h", eager=True)}
        ... ).with_row_count()
        >>> df_temporal
        shape: (25, 2)
        ┌────────┬─────────────────────┐
        │ row_nr ┆ date                │
        │ ---    ┆ ---                 │
        │ u32    ┆ datetime[μs]        │
        ╞════════╪═════════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 │
        │ 1      ┆ 2001-01-01 01:00:00 │
        │ 2      ┆ 2001-01-01 02:00:00 │
        │ 3      ┆ 2001-01-01 03:00:00 │
        │ …      ┆ …                   │
        │ 21     ┆ 2001-01-01 21:00:00 │
        │ 22     ┆ 2001-01-01 22:00:00 │
        │ 23     ┆ 2001-01-01 23:00:00 │
        │ 24     ┆ 2001-01-02 00:00:00 │
        └────────┴─────────────────────┘

        Compute the rolling sum with the default left closure of temporal windows

        >>> df_temporal.with_columns(
        ...     rolling_row_sum=pl.col("row_nr").rolling_sum(
        ...         window_size="2h", by="date", closed="left"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬─────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_sum │
        │ ---    ┆ ---                 ┆ ---             │
        │ u32    ┆ datetime[μs]        ┆ u32             │
        ╞════════╪═════════════════════╪═════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ null            │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 0               │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 1               │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 3               │
        │ …      ┆ …                   ┆ …               │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 39              │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 41              │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 43              │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 45              │
        └────────┴─────────────────────┴─────────────────┘

        Compute the rolling sum with the closure of windows on both sides

        >>> df_temporal.with_columns(
        ...     rolling_row_sum=pl.col("row_nr").rolling_sum(
        ...         window_size="2h", by="date", closed="both"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬─────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_sum │
        │ ---    ┆ ---                 ┆ ---             │
        │ u32    ┆ datetime[μs]        ┆ u32             │
        ╞════════╪═════════════════════╪═════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ 0               │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 1               │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 3               │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 6               │
        │ …      ┆ …                   ┆ …               │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 60              │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 63              │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 66              │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 69              │
        └────────┴─────────────────────┴─────────────────┘

        '''
    def rolling_std(self, *args, **kwargs) -> Self:
        '''
        Compute a rolling standard deviation.

        If ``by`` has not been specified (the default), the window at a given row will
        include the row itself, and the `window_size - 1` elements before it.

        If you pass a ``by`` column ``<t_0, t_1, ..., t_n>``, then `closed="left"` means
        the windows will be:

            - [t_0 - window_size, t_0)
            - [t_1 - window_size, t_1)
            - ...
            - [t_n - window_size, t_n)

        With `closed="right"`, the left endpoint is not included and the right
        endpoint is included.

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
            - 1d    (1 calendar day)
            - 1w    (1 calendar week)
            - 1mo   (1 calendar month)
            - 1q    (1 calendar quarter)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

            By "calendar day", we mean the corresponding time on the next day
            (which may not be 24 hours, due to daylight savings). Similarly for
            "calendar week", "calendar month", "calendar quarter", and
            "calendar year".

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that determines the
            relative contribution of each value in a window to the output.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype Datetime.
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive); only
            applicable if `by` has been set.
        ddof
            "Delta Degrees of Freedom": The divisor for a length N window is N - ddof

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
        >>> df.with_columns(
        ...     rolling_std=pl.col("A").rolling_std(window_size=2),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_std │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 0.707107    │
        │ 3.0 ┆ 0.707107    │
        │ 4.0 ┆ 0.707107    │
        │ 5.0 ┆ 0.707107    │
        │ 6.0 ┆ 0.707107    │
        └─────┴─────────────┘

        Specify weights to multiply the values in the window with:

        >>> df.with_columns(
        ...     rolling_std=pl.col("A").rolling_std(
        ...         window_size=2, weights=[0.25, 0.75]
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_std │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 0.433013    │
        │ 3.0 ┆ 0.433013    │
        │ 4.0 ┆ 0.433013    │
        │ 5.0 ┆ 0.433013    │
        │ 6.0 ┆ 0.433013    │
        └─────┴─────────────┘

        Center the values in the window

        >>> df.with_columns(
        ...     rolling_std=pl.col("A").rolling_std(window_size=3, center=True),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_std │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 1.0         │
        │ 3.0 ┆ 1.0         │
        │ 4.0 ┆ 1.0         │
        │ 5.0 ┆ 1.0         │
        │ 6.0 ┆ null        │
        └─────┴─────────────┘

        Create a DataFrame with a datetime column and a row number column

        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> df_temporal = pl.DataFrame(
        ...     {"date": pl.date_range(start, stop, "1h", eager=True)}
        ... ).with_row_count()
        >>> df_temporal
        shape: (25, 2)
        ┌────────┬─────────────────────┐
        │ row_nr ┆ date                │
        │ ---    ┆ ---                 │
        │ u32    ┆ datetime[μs]        │
        ╞════════╪═════════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 │
        │ 1      ┆ 2001-01-01 01:00:00 │
        │ 2      ┆ 2001-01-01 02:00:00 │
        │ 3      ┆ 2001-01-01 03:00:00 │
        │ …      ┆ …                   │
        │ 21     ┆ 2001-01-01 21:00:00 │
        │ 22     ┆ 2001-01-01 22:00:00 │
        │ 23     ┆ 2001-01-01 23:00:00 │
        │ 24     ┆ 2001-01-02 00:00:00 │
        └────────┴─────────────────────┘

        Compute the rolling std with the default left closure of temporal windows

        >>> df_temporal.with_columns(
        ...     rolling_row_std=pl.col("row_nr").rolling_std(
        ...         window_size="2h", by="date", closed="left"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬─────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_std │
        │ ---    ┆ ---                 ┆ ---             │
        │ u32    ┆ datetime[μs]        ┆ f64             │
        ╞════════╪═════════════════════╪═════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ null            │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 0.0             │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 0.707107        │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 0.707107        │
        │ …      ┆ …                   ┆ …               │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 0.707107        │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 0.707107        │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 0.707107        │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 0.707107        │
        └────────┴─────────────────────┴─────────────────┘

        Compute the rolling std with the closure of windows on both sides

        >>> df_temporal.with_columns(
        ...     rolling_row_std=pl.col("row_nr").rolling_std(
        ...         window_size="2h", by="date", closed="both"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬─────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_std │
        │ ---    ┆ ---                 ┆ ---             │
        │ u32    ┆ datetime[μs]        ┆ f64             │
        ╞════════╪═════════════════════╪═════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ 0.0             │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 0.707107        │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 1.0             │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 1.0             │
        │ …      ┆ …                   ┆ …               │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 1.0             │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 1.0             │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 1.0             │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 1.0             │
        └────────┴─────────────────────┴─────────────────┘

        '''
    def rolling_var(self, *args, **kwargs) -> Self:
        '''
        Compute a rolling variance.

        If ``by`` has not been specified (the default), the window at a given row will
        include the row itself, and the `window_size - 1` elements before it.

        If you pass a ``by`` column ``<t_0, t_1, ..., t_n>``, then `closed="left"`
        means the windows will be:

            - [t_0 - window_size, t_0)
            - [t_1 - window_size, t_1)
            - ...
            - [t_n - window_size, t_n)

        With `closed="right"`, the left endpoint is not included and the right
        endpoint is included.

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
            - 1d    (1 calendar day)
            - 1w    (1 calendar week)
            - 1mo   (1 calendar month)
            - 1q    (1 calendar quarter)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

            By "calendar day", we mean the corresponding time on the next day
            (which may not be 24 hours, due to daylight savings). Similarly for
            "calendar week", "calendar month", "calendar quarter", and
            "calendar year".

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that determines the
            relative contribution of each value in a window to the output.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype Datetime.
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive); only
            applicable if `by` has been set.
        ddof
            "Delta Degrees of Freedom": The divisor for a length N window is N - ddof

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
        >>> df.with_columns(
        ...     rolling_var=pl.col("A").rolling_var(window_size=2),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_var │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 0.5         │
        │ 3.0 ┆ 0.5         │
        │ 4.0 ┆ 0.5         │
        │ 5.0 ┆ 0.5         │
        │ 6.0 ┆ 0.5         │
        └─────┴─────────────┘

        Specify weights to multiply the values in the window with:

        >>> df.with_columns(
        ...     rolling_var=pl.col("A").rolling_var(
        ...         window_size=2, weights=[0.25, 0.75]
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_var │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 0.1875      │
        │ 3.0 ┆ 0.1875      │
        │ 4.0 ┆ 0.1875      │
        │ 5.0 ┆ 0.1875      │
        │ 6.0 ┆ 0.1875      │
        └─────┴─────────────┘

        Center the values in the window

        >>> df.with_columns(
        ...     rolling_var=pl.col("A").rolling_var(window_size=3, center=True),
        ... )
        shape: (6, 2)
        ┌─────┬─────────────┐
        │ A   ┆ rolling_var │
        │ --- ┆ ---         │
        │ f64 ┆ f64         │
        ╞═════╪═════════════╡
        │ 1.0 ┆ null        │
        │ 2.0 ┆ 1.0         │
        │ 3.0 ┆ 1.0         │
        │ 4.0 ┆ 1.0         │
        │ 5.0 ┆ 1.0         │
        │ 6.0 ┆ null        │
        └─────┴─────────────┘

        Create a DataFrame with a datetime column and a row number column

        >>> from datetime import timedelta, datetime
        >>> start = datetime(2001, 1, 1)
        >>> stop = datetime(2001, 1, 2)
        >>> df_temporal = pl.DataFrame(
        ...     {"date": pl.date_range(start, stop, "1h", eager=True)}
        ... ).with_row_count()
        >>> df_temporal
        shape: (25, 2)
        ┌────────┬─────────────────────┐
        │ row_nr ┆ date                │
        │ ---    ┆ ---                 │
        │ u32    ┆ datetime[μs]        │
        ╞════════╪═════════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 │
        │ 1      ┆ 2001-01-01 01:00:00 │
        │ 2      ┆ 2001-01-01 02:00:00 │
        │ 3      ┆ 2001-01-01 03:00:00 │
        │ …      ┆ …                   │
        │ 21     ┆ 2001-01-01 21:00:00 │
        │ 22     ┆ 2001-01-01 22:00:00 │
        │ 23     ┆ 2001-01-01 23:00:00 │
        │ 24     ┆ 2001-01-02 00:00:00 │
        └────────┴─────────────────────┘

        Compute the rolling var with the default left closure of temporal windows

        >>> df_temporal.with_columns(
        ...     rolling_row_var=pl.col("row_nr").rolling_var(
        ...         window_size="2h", by="date", closed="left"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬─────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_var │
        │ ---    ┆ ---                 ┆ ---             │
        │ u32    ┆ datetime[μs]        ┆ f64             │
        ╞════════╪═════════════════════╪═════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ null            │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 0.0             │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 0.5             │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 0.5             │
        │ …      ┆ …                   ┆ …               │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 0.5             │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 0.5             │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 0.5             │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 0.5             │
        └────────┴─────────────────────┴─────────────────┘

        Compute the rolling var with the closure of windows on both sides

        >>> df_temporal.with_columns(
        ...     rolling_row_var=pl.col("row_nr").rolling_var(
        ...         window_size="2h", by="date", closed="both"
        ...     )
        ... )
        shape: (25, 3)
        ┌────────┬─────────────────────┬─────────────────┐
        │ row_nr ┆ date                ┆ rolling_row_var │
        │ ---    ┆ ---                 ┆ ---             │
        │ u32    ┆ datetime[μs]        ┆ f64             │
        ╞════════╪═════════════════════╪═════════════════╡
        │ 0      ┆ 2001-01-01 00:00:00 ┆ 0.0             │
        │ 1      ┆ 2001-01-01 01:00:00 ┆ 0.5             │
        │ 2      ┆ 2001-01-01 02:00:00 ┆ 1.0             │
        │ 3      ┆ 2001-01-01 03:00:00 ┆ 1.0             │
        │ …      ┆ …                   ┆ …               │
        │ 21     ┆ 2001-01-01 21:00:00 ┆ 1.0             │
        │ 22     ┆ 2001-01-01 22:00:00 ┆ 1.0             │
        │ 23     ┆ 2001-01-01 23:00:00 ┆ 1.0             │
        │ 24     ┆ 2001-01-02 00:00:00 ┆ 1.0             │
        └────────┴─────────────────────┴─────────────────┘

        '''
    def rolling_median(self, *args, **kwargs) -> Self:
        '''
        Compute a rolling median.

        If ``by`` has not been specified (the default), the window at a given row will
        include the row itself, and the `window_size - 1` elements before it.

        If you pass a ``by`` column ``<t_0, t_1, ..., t_n>``, then `closed="left"` means
        the windows will be:

            - [t_0 - window_size, t_0)
            - [t_1 - window_size, t_1)
            - ...
            - [t_n - window_size, t_n)

        With `closed="right"`, the left endpoint is not included and the right
        endpoint is included.

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
            - 1d    (1 calendar day)
            - 1w    (1 calendar week)
            - 1mo   (1 calendar month)
            - 1q    (1 calendar quarter)
            - 1y    (1 calendar year)
            - 1i    (1 index count)

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

            By "calendar day", we mean the corresponding time on the next day
            (which may not be 24 hours, due to daylight savings). Similarly for
            "calendar week", "calendar month", "calendar quarter", and
            "calendar year".

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that determines the
            relative contribution of each value in a window to the output.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype Datetime.
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive); only
            applicable if `by` has been set.

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
        >>> df.with_columns(
        ...     rolling_median=pl.col("A").rolling_median(window_size=2),
        ... )
        shape: (6, 2)
        ┌─────┬────────────────┐
        │ A   ┆ rolling_median │
        │ --- ┆ ---            │
        │ f64 ┆ f64            │
        ╞═════╪════════════════╡
        │ 1.0 ┆ null           │
        │ 2.0 ┆ 1.5            │
        │ 3.0 ┆ 2.5            │
        │ 4.0 ┆ 3.5            │
        │ 5.0 ┆ 4.5            │
        │ 6.0 ┆ 5.5            │
        └─────┴────────────────┘

        Specify weights for the values in each window:

        >>> df.with_columns(
        ...     rolling_median=pl.col("A").rolling_median(
        ...         window_size=2, weights=[0.25, 0.75]
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬────────────────┐
        │ A   ┆ rolling_median │
        │ --- ┆ ---            │
        │ f64 ┆ f64            │
        ╞═════╪════════════════╡
        │ 1.0 ┆ null           │
        │ 2.0 ┆ 1.5            │
        │ 3.0 ┆ 2.5            │
        │ 4.0 ┆ 3.5            │
        │ 5.0 ┆ 4.5            │
        │ 6.0 ┆ 5.5            │
        └─────┴────────────────┘

        Center the values in the window

        >>> df.with_columns(
        ...     rolling_median=pl.col("A").rolling_median(window_size=3, center=True),
        ... )
        shape: (6, 2)
        ┌─────┬────────────────┐
        │ A   ┆ rolling_median │
        │ --- ┆ ---            │
        │ f64 ┆ f64            │
        ╞═════╪════════════════╡
        │ 1.0 ┆ null           │
        │ 2.0 ┆ 2.0            │
        │ 3.0 ┆ 3.0            │
        │ 4.0 ┆ 4.0            │
        │ 5.0 ┆ 5.0            │
        │ 6.0 ┆ null           │
        └─────┴────────────────┘

        '''
    def rolling_quantile(self, *args, **kwargs) -> Self:
        '''
        Compute a rolling quantile.

        If ``by`` has not been specified (the default), the window at a given row will
        include the row itself, and the `window_size - 1` elements before it.

        If you pass a ``by`` column ``<t_0, t_1, ..., t_n>``, then `closed="left"`
        means the windows will be:

            - [t_0 - window_size, t_0)
            - [t_1 - window_size, t_1)
            - ...
            - [t_n - window_size, t_n)

        With `closed="right"`, the left endpoint is not included and the right
        endpoint is included.

        Parameters
        ----------
        quantile
            Quantile between 0.0 and 1.0.
        interpolation : {\'nearest\', \'higher\', \'lower\', \'midpoint\', \'linear\'}
            Interpolation method.
        window_size
            The length of the window. Can be a fixed integer size, or a dynamic
            temporal size indicated by a timedelta or the following string language:

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

            Suffix with `"_saturating"` to indicate that dates too large for
            their month should saturate at the largest date
            (e.g. 2022-02-29 -> 2022-02-28) instead of erroring.

            By "calendar day", we mean the corresponding time on the next day
            (which may not be 24 hours, due to daylight savings). Similarly for
            "calendar week", "calendar month", "calendar quarter", and
            "calendar year".

            If a timedelta or the dynamic string language is used, the `by`
            and `closed` arguments must also be set.
        weights
            An optional slice with the same length as the window that determines the
            relative contribution of each value in a window to the output.
        min_periods
            The number of values in the window that should be non-null before computing
            a result. If None, it will be set equal to window size.
        center
            Set the labels at the center of the window
        by
            If the `window_size` is temporal for instance `"5h"` or `"3s"`, you must
            set the column that will be used to determine the windows. This column must
            be of dtype Datetime.
        closed : {\'left\', \'right\', \'both\', \'none\'}
            Define which sides of the temporal interval are closed (inclusive); only
            applicable if `by` has been set.

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
        >>> df.with_columns(
        ...     rolling_quantile=pl.col("A").rolling_quantile(
        ...         quantile=0.25, window_size=4
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬──────────────────┐
        │ A   ┆ rolling_quantile │
        │ --- ┆ ---              │
        │ f64 ┆ f64              │
        ╞═════╪══════════════════╡
        │ 1.0 ┆ null             │
        │ 2.0 ┆ null             │
        │ 3.0 ┆ null             │
        │ 4.0 ┆ 2.0              │
        │ 5.0 ┆ 3.0              │
        │ 6.0 ┆ 4.0              │
        └─────┴──────────────────┘

        Specify weights for the values in each window:

        >>> df.with_columns(
        ...     rolling_quantile=pl.col("A").rolling_quantile(
        ...         quantile=0.25, window_size=4, weights=[0.2, 0.4, 0.4, 0.2]
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬──────────────────┐
        │ A   ┆ rolling_quantile │
        │ --- ┆ ---              │
        │ f64 ┆ f64              │
        ╞═════╪══════════════════╡
        │ 1.0 ┆ null             │
        │ 2.0 ┆ null             │
        │ 3.0 ┆ null             │
        │ 4.0 ┆ 2.0              │
        │ 5.0 ┆ 3.0              │
        │ 6.0 ┆ 4.0              │
        └─────┴──────────────────┘

        Specify weights and interpolation method

        >>> df.with_columns(
        ...     rolling_quantile=pl.col("A").rolling_quantile(
        ...         quantile=0.25,
        ...         window_size=4,
        ...         weights=[0.2, 0.4, 0.4, 0.2],
        ...         interpolation="linear",
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬──────────────────┐
        │ A   ┆ rolling_quantile │
        │ --- ┆ ---              │
        │ f64 ┆ f64              │
        ╞═════╪══════════════════╡
        │ 1.0 ┆ null             │
        │ 2.0 ┆ null             │
        │ 3.0 ┆ null             │
        │ 4.0 ┆ 1.625            │
        │ 5.0 ┆ 2.625            │
        │ 6.0 ┆ 3.625            │
        └─────┴──────────────────┘

        Center the values in the window

        >>> df.with_columns(
        ...     rolling_quantile=pl.col("A").rolling_quantile(
        ...         quantile=0.2, window_size=5, center=True
        ...     ),
        ... )
        shape: (6, 2)
        ┌─────┬──────────────────┐
        │ A   ┆ rolling_quantile │
        │ --- ┆ ---              │
        │ f64 ┆ f64              │
        ╞═════╪══════════════════╡
        │ 1.0 ┆ null             │
        │ 2.0 ┆ null             │
        │ 3.0 ┆ 2.0              │
        │ 4.0 ┆ 3.0              │
        │ 5.0 ┆ null             │
        │ 6.0 ┆ null             │
        └─────┴──────────────────┘

        '''
    def rolling_apply(self, function: Callable[[Series], Any], window_size: int, weights: list[float] | None = ..., min_periods: int | None = ...) -> Self:
        '''
        Apply a custom rolling window function.

        Prefer the specific rolling window functions over this one, as they are faster.

        Prefer:

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

        '''
    def rolling_skew(self, window_size: int) -> Self:
        '''
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
        >>> df = pl.DataFrame({"a": [1, 4, 2, 9]})
        >>> df.select(pl.col("a").rolling_skew(3))
        shape: (4, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ null     │
        │ null     │
        │ 0.381802 │
        │ 0.47033  │
        └──────────┘

        Note how the values match the following:

        >>> pl.Series([1, 4, 2]).skew(), pl.Series([4, 2, 9]).skew()
        (0.38180177416060584, 0.47033046033698594)

        '''
    def abs(self) -> Self:
        '''
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

        '''
    def rank(self, method: RankMethod = ...) -> Self:
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

        Use \'rank\' with \'over\' to rank within groups:

        >>> df = pl.DataFrame({"a": [1, 1, 2, 2, 2], "b": [6, 7, 5, 14, 11]})
        >>> df.with_columns(pl.col("b").rank().over("a").alias("rank"))
        shape: (5, 3)
        ┌─────┬─────┬──────┐
        │ a   ┆ b   ┆ rank │
        │ --- ┆ --- ┆ ---  │
        │ i64 ┆ i64 ┆ f32  │
        ╞═════╪═════╪══════╡
        │ 1   ┆ 6   ┆ 1.0  │
        │ 1   ┆ 7   ┆ 2.0  │
        │ 2   ┆ 5   ┆ 1.0  │
        │ 2   ┆ 14  ┆ 3.0  │
        │ 2   ┆ 11  ┆ 2.0  │
        └─────┴─────┴──────┘

        '''
    def diff(self, n: int = ..., null_behavior: NullBehavior = ...) -> Self:
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

        '''
    def pct_change(self, n: int = ...) -> Self:
        '''
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

        '''
    def skew(self) -> Self:
        '''
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

        '''
    def kurtosis(self) -> Self:
        '''
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

        '''
    def clip(self, lower_bound: int | float, upper_bound: int | float) -> Self:
        '''
        Clip (limit) the values in an array to a `min` and `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        lower_bound
            Lower bound.
        upper_bound
            Upper bound.

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

        '''
    def clip_min(self, lower_bound: int | float) -> Self:
        '''
        Clip (limit) the values in an array to a `min` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        lower_bound
            Lower bound.

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

        '''
    def clip_max(self, upper_bound: int | float) -> Self:
        '''
        Clip (limit) the values in an array to a `max` boundary.

        Only works for numerical types.

        If you want to clip other dtypes, consider writing a "when, then, otherwise"
        expression. See :func:`when` for more information.

        Parameters
        ----------
        upper_bound
            Upper bound.

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

        '''
    def lower_bound(self) -> Self:
        '''
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

        '''
    def upper_bound(self) -> Self:
        '''
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

        '''
    def sign(self) -> Self:
        '''
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

        '''
    def sin(self) -> Self:
        '''
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

        '''
    def cos(self) -> Self:
        '''
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

        '''
    def tan(self) -> Self:
        '''
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

        '''
    def arcsin(self) -> Self:
        '''
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

        '''
    def arccos(self) -> Self:
        '''
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

        '''
    def arctan(self) -> Self:
        '''
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

        '''
    def sinh(self) -> Self:
        '''
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

        '''
    def cosh(self) -> Self:
        '''
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

        '''
    def tanh(self) -> Self:
        '''
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

        '''
    def arcsinh(self) -> Self:
        '''
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

        '''
    def arccosh(self) -> Self:
        '''
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

        '''
    def arctanh(self) -> Self:
        '''
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

        '''
    def degrees(self) -> Self:
        '''
        Convert from radians to degrees.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> import math
        >>> df = pl.DataFrame({"a": [x * math.pi for x in range(-4, 5)]})
        >>> df.select(pl.col("a").degrees())
        shape: (9, 1)
        ┌────────┐
        │ a      │
        │ ---    │
        │ f64    │
        ╞════════╡
        │ -720.0 │
        │ -540.0 │
        │ -360.0 │
        │ -180.0 │
        │ 0.0    │
        │ 180.0  │
        │ 360.0  │
        │ 540.0  │
        │ 720.0  │
        └────────┘
        '''
    def radians(self) -> Self:
        '''
        Convert from degrees to radians.

        Returns
        -------
        Series of dtype Float64

        Examples
        --------
        >>> df = pl.DataFrame({"a": [-720, -540, -360, -180, 0, 180, 360, 540, 720]})
        >>> df.select(pl.col("a").radians())
        shape: (9, 1)
        ┌────────────┐
        │ a          │
        │ ---        │
        │ f64        │
        ╞════════════╡
        │ -12.566371 │
        │ -9.424778  │
        │ -6.283185  │
        │ -3.141593  │
        │ 0.0        │
        │ 3.141593   │
        │ 6.283185   │
        │ 9.424778   │
        │ 12.566371  │
        └────────────┘
        '''
    def reshape(self, dimensions: tuple[int, ...]) -> Self:
        '''
        Reshape this Expr to a flat Series or a Series of Lists.

        Parameters
        ----------
        dimensions
            Tuple of the dimension sizes. If a -1 is used in any of the dimensions, that
            dimension is inferred.

        Returns
        -------
        Expr
            If a single dimension is given, results in a flat Series of shape (len,).
            If a multiple dimensions are given, results in a Series of Lists with shape
            (rows, cols).

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

        See Also
        --------
        Expr.list.explode : Explode a list column.

        '''
    def shuffle(self, seed: int | None = ..., fixed_seed: bool = ...) -> Self:
        '''
        Shuffle the contents of this expression.

        Parameters
        ----------
        seed
            Seed for the random number generator. If set to None (default), a random
            seed is generated using the ``random`` module.
        fixed_seed
            If True, The seed will not be incremented between draws.
            This can make output predictable because draw ordering can
            change due to threads being scheduled in a different order.

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

        '''
    def sample(self, *args, **kwargs) -> Self:
        '''
        Sample from this expression.

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
        fixed_seed
            If True, The seed will not be incremented between draws.
            This can make output predictable because draw ordering can
            change due to threads being scheduled in a different order.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").sample(fraction=1.0, with_replacement=True, seed=1))
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

        '''
    def ewm_mean(self, com: float | None = ..., span: float | None = ..., half_life: float | None = ..., alpha: float | None = ...) -> Self:
        '''
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

        '''
    def ewm_std(self, com: float | None = ..., span: float | None = ..., half_life: float | None = ..., alpha: float | None = ...) -> Self:
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

        '''
    def ewm_var(self, com: float | None = ..., span: float | None = ..., half_life: float | None = ..., alpha: float | None = ...) -> Self:
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

        '''
    def extend_constant(self, value: PythonLiteral | None, n: int) -> Self:
        '''
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

        '''
    def value_counts(self) -> Self:
        '''
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

        '''
    def unique_counts(self) -> Self:
        '''
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

        '''
    def log(self, base: float = ...) -> Self:
        '''
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

        '''
    def log1p(self) -> Self:
        '''
        Compute the natural logarithm of each element plus one.

        This computes `log(1 + x)` but is more numerically stable for `x` close to zero.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.select(pl.col("a").log1p())
        shape: (3, 1)
        ┌──────────┐
        │ a        │
        │ ---      │
        │ f64      │
        ╞══════════╡
        │ 0.693147 │
        │ 1.098612 │
        │ 1.386294 │
        └──────────┘

        '''
    def entropy(self, base: float = ...) -> Self:
        '''
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

        '''
    def cumulative_eval(self, expr: Expr, min_periods: int = ...) -> Self:
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

        '''
    def set_sorted(self) -> Self:
        '''
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

        '''
    def shrink_dtype(self) -> Self:
        '''
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

        '''
    def cache(self) -> Self:
        """
        Cache this expression so that it only is executed once per context.

        This can actually hurt performance and can have a lot of contention.
        It is advised not to use it until actually benchmarked on your problem.

        """
    def map_dict(self, remapping: dict[Any, Any]) -> Self:
        '''
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
        return_dtype
            Set return dtype to override automatic return dtype determination.

        See Also
        --------
        map

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

        Override return dtype:

        >>> df.with_columns(
        ...     pl.col("row_nr")
        ...     .map_dict({1: 7, 3: 4}, default=3, return_dtype=pl.UInt8)
        ...     .alias("remapped")
        ... )
        shape: (4, 3)
        ┌────────┬──────────────┬──────────┐
        │ row_nr ┆ country_code ┆ remapped │
        │ ---    ┆ ---          ┆ ---      │
        │ u32    ┆ str          ┆ u8       │
        ╞════════╪══════════════╪══════════╡
        │ 0      ┆ FR           ┆ 3        │
        │ 1      ┆ null         ┆ 7        │
        │ 2      ┆ ES           ┆ 3        │
        │ 3      ┆ DE           ┆ 4        │
        └────────┴──────────────┴──────────┘

        '''
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
    def meta(self): ...
    @property
    def str(self): ...
    @property
    def struct(self): ...
def _prepare_alpha(com: float | int | None = ..., span: float | int | None = ..., half_life: float | int | None = ..., alpha: float | int | None = ...) -> float:
    """Normalise EWM decay specification in terms of smoothing factor 'alpha'."""
def _prepare_rolling_window_args(window_size: int | timedelta | str, min_periods: int | None = ...) -> tuple[str, int]: ...

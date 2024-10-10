import polars as pl


class NestedInitPackageNamespace:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def some_method(self, x: int) -> pl.LazyFrame:
        """my docstring"""
        return self._ldf

import polars as pl

class CustomNamespace:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def custom_method(self, x: int) -> pl.LazyFrame:
        """my docstring"""
        return self._ldf


class EntryPointNameSpace:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def custom_method(self, x: int) -> pl.LazyFrame:
        """my docstring"""
        return self._ldf




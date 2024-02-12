import polars as pl


@pl.api.register_lazyframe_namespace("selfregister")
class SelfRegisterNamespace:
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf

    def custom_method(self, x: int) -> pl.LazyFrame:
        """my docstring"""
        return self._ldf

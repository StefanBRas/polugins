from polars import LazyFrame
from polugins.main import register_namespaces


def test_custom():
    ldf = LazyFrame()
    from polugins.namespace import CustomNamespace
    register_namespaces(lazyframe_namespaces={'custom': CustomNamespace}, entrypoints=False)
    ldf.custom.custom_method(x=1)

def test_external():
    ldf = LazyFrame()
    register_namespaces(entrypoints=True)
    ldf.external.some_method(x=1)
    ldf.internal.custom_method(x=1)

import ast
import importlib
import inspect
import sys
from pathlib import Path

from polugins._types import ExtensionClass

_IMPORT_PATHS = {
    ExtensionClass.EXPR: Path("polars", "expr", "expr"),
    ExtensionClass.SERIES: Path("polars", "series", "series"),
    ExtensionClass.LAZYFRAME: Path("polars", "lazyframe", "frame"),
    ExtensionClass.DATAFRAME: Path("polars", "dataframe", "frame"),
}


class RemoveBodies(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        """Remove function bodies"""
        if ast.get_docstring(node) is not None:
            # Assumption: If there is a docstring, it's the first element of the body
            node.body = [node.body[0], ast.Expr(value=ast.Constant(value=...))]
        else:
            node.body = [ast.Expr(value=ast.Constant(value=...))]
        node.args.defaults = []
        return node

    def visit_AnnAssign(self, node):
        """Remove value assignemnets
        Turns x: int = 1 into x: int

        TODO: Reconsider this - it might remove type information.
        For example if there is `x = 1` in the source, it will be infered to int or literal 1
        but without it, it will be Any
        """
        node.value = None
        return node

    def visit_If(self, node: ast.If):
        """Unnest TYPE_CHECKING if statements"""
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return node.body
        return node

    def visit_With(self, node: ast.With):
        """Unnest contextlib.suppress(ImportError)

        Assumption is that these are only used to guard in doctests
        """
        if isinstance(node.items[0].context_expr, ast.Call):
            call = node.items[0].context_expr
            if (
                isinstance(call.func, ast.Attribute)
                and getattr(call.func.value, "id", None) == "contextlib"
                and call.func.attr == "suppress"
                and getattr(call.args[0], "id", None) == "ImportError"
            ):
                return node.body
        return node


def parse(module_source: str):
    parsed = ast.parse(module_source, type_comments=True)
    parsed = RemoveBodies().visit(parsed)
    return ast.unparse(parsed)


def parse_from_current_env(extension_class: ExtensionClass) -> ast.Module:
    module_import_path = ".".join(_IMPORT_PATHS[extension_class].parts)
    module = importlib.import_module(module_import_path)
    module_source = inspect.getsource(module)
    parsed = ast.parse(module_source, type_comments=False)
    return RemoveBodies().visit(parsed)


if __name__ == "__main__":
    output_dir = sys.argv[1]
    for extension_class in ExtensionClass:
        module_import_path = _IMPORT_PATHS[extension_class]
        parsed = parse_from_current_env(extension_class)
        output_path = (Path(output_dir) / module_import_path).with_suffix(".pyi")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(ast.unparse(parsed))

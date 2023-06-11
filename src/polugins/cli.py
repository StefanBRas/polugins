from pathlib import Path
import importlib_resources
import sys

from polugins.main import get_entrypoints
import ast


def create_stubs(version: str):
    output_dir = Path("typings")

    entry_points = get_entrypoints()
    # TODO: get from configs too

    for extension_class, entrypoints in entry_points.items():
        if entrypoints:
            files = importlib_resources.files("polugins")
            stub_path = files / "_stubs" / version / extension_class.import_path
            stub_ast = ast.parse(stub_path.read_text(), type_comments=True)
            new_class_nodes = []
            modules_to_import = set()
            for node in stub_ast.body:
                if (
                    isinstance(node, ast.ClassDef)
                    and node.name.casefold() == extension_class.value.casefold()
                ):
                    for ep in entrypoints:
                        annotation_name = ep.value.replace(":", ".")
                        new_node = ast.AnnAssign(
                            target=ast.Name(id=ep.name),
                            annotation=ast.Name(id=annotation_name),
                            simple=1,
                        )  # no idea what `simple` does, but it breaks without it
                        new_class_nodes.append(new_node)
                        modules_to_import.add(annotation_name.split(".")[0])
                    node.body.extend(new_class_nodes)
            for name in modules_to_import:
                # prepend to have imports first
                stub_ast.body.insert(0, ast.Import(names=[ast.Name(id=name)]))
            output_path = output_dir / extension_class.import_path
            output_path.parent.mkdir(exist_ok=True, parents=True)
            output_path.with_suffix(".pyi").write_text(ast.unparse(stub_ast))


def cli():
    if len(sys.argv) == 1:
        from polugins._version import __version__

        print(f"Polugins version: {__version__}")
    elif sys.argv[1] == "stubs":
        print("generating stubs at ./typings/")
        import polars as pl

        create_stubs(pl.__version__)
    else:
        print("Use `polugins stubs` to generate type stubs.")


if __name__ == "__main__":
    cli()

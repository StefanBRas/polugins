from pathlib import Path
import importlib

from polugins.main import get_entrypoints
import ast



def main(version: str):
    output_dir = Path("typings")

    entry_points = get_entrypoints()

    for extension_class, entrypoints in entry_points.items():
        if entrypoints:
            files = importlib.resources.files("polugins")
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


if __name__ == "__main__":
    import polars as pl
    main(pl.__version__)

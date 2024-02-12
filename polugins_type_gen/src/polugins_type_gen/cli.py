import ast
import importlib.resources as importlib_resources
import sys
from pathlib import Path

from polugins.main import _get_namespaces


class MissingVersionException(Exception):
    pass


class NoNamespaceRegisteredException(Exception):
    pass


def has_version(version: str) -> bool:
    files = importlib_resources.files("polugins_type_gen")
    return (files / "_stubs" / version).is_dir()


def create_stubs(version: str):
    output_dir = Path("typings")
    if not has_version(version):
        msg = (
            f"Type stubs for version {version} does not exist."
            " This is usually because the version has been yanked or because it's new."
            " Feel free to create an issue if you want types for this version."
        )
        raise MissingVersionException(msg)

    all_namespaces = _get_namespaces()

    if all(namespace == {} for namespace in all_namespaces.values()):
        msg = "No namespaces found. No types will be generated as this is usually an error."
        raise NoNamespaceRegisteredException(msg)

    for extension_class, namespaces in all_namespaces.items():
        if namespaces:
            files = importlib_resources.files("polugins_type_gen")
            stub_path = (
                Path(str(files)) / "_stubs" / version / extension_class.import_path
            ).with_suffix(".pyi")
            stub_ast = ast.parse(stub_path.read_text(), type_comments=True)
            new_class_nodes = []
            modules_to_import = set()
            for node in stub_ast.body:
                if (
                    isinstance(node, ast.ClassDef)
                    and node.name.casefold() == extension_class.value.casefold()
                ):
                    for name, namespace in namespaces.items():
                        annotation_name = namespace.replace(":", ".")
                        new_node = ast.AnnAssign(
                            target=ast.Name(id=name),
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
        from polugins_type_gen._version import __version__

        print(f"Polugins version: {__version__}")
    elif sys.argv[1] == "stubs":
        version = None
        if len(sys.argv) >= 3:
            version = sys.argv[2]
        else:
            print("No version given; trying to infer")
            try:
                import polars as pl

                version = pl.__version__
                print(f"Found polars version {version} in current python environment.")
                print("Using this verison for type stubs.")
            except Exception:
                pass
        if version is None:
            msg = "No version was given or could be infered"
            raise ValueError(msg)
        print("generating stubs at ./typings/")

        create_stubs(version)
    else:
        print("Use `polugins stubs` to generate type stubs.")


if __name__ == "__main__":
    cli()

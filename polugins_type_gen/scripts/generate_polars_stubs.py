import subprocess
import sys
from difflib import unified_diff
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging import version

IMPORT_PATHS = [
    Path("polars", "expr", "expr"),
    Path("polars", "series", "series"),
    Path("polars", "lazyframe", "frame"),
    Path("polars", "dataframe", "frame"),
]


def run_stubgen(version: str, no_docstring_stub_path: Path, stub_path: Path, tempdir_path: Path) -> None:
    venv_path = tempdir_path / f".venv{version}"
    bin_path = venv_path / "bin" / "python"
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
    subprocess.check_call([bin_path, "-m", "pip", "install", f"polars=={version}"])
    subprocess.check_call([bin_path, "scripts/stubgen_ast.py", stub_path, "true"])
    subprocess.check_call(
        [
            bin_path,
            "scripts/stubgen_ast.py",
            tempdir_path / no_docstring_stub_path,
            "false",
        ]
    )


def get_current_versions() -> set[version.Version]:
    stub_dir = Path(__file__).parent.parent / "src" / "polugins_type_gen" / "_stubs"
    return {version.parse(p.parts[-1]) for p in stub_dir.iterdir()}


def get_available_versions() -> set[version.Version]:
    res = subprocess.run(
        [sys.executable, "-m", "pip", "index", "versions", "polars"],
        capture_output=True,
        text=True,
    )
    version_line_start = "Available versions: "
    versions = None
    for line in res.stdout.splitlines():
        if line.startswith(version_line_start):
            versions = line.split(version_line_start)[1].split(",")
            break
    assert versions
    return {version.parse(version_str.strip()) for version_str in versions}


def get_missing_versions() -> set[version.Version]:
    # 0.16.13 -> 0.16.14 changes location of imports
    oldest_version = version.Version("0.16.13")
    return {v for v in (get_available_versions() - get_current_versions()) if v > oldest_version}


def clean_types(path: Path, version):
    stub_content = path.read_text()
    match path.parts[-2]:
        case "dataframe":
            if (txt := "P: Incomplete") in stub_content:
                stub_content = "from typing_extensions import ParamSpec, Generic\n" + stub_content.replace(
                    txt, "P = ParamSpec('P')"
                ).replace("class DataFrame", "class DataFrame(Generic[P])")
            stub_content = stub_content.replace("_df: Incomplete", "_df: PyDataFrame")
            stub_content = stub_content.replace("columns: Incomplete", "columns: list[str]")
            stub_content = stub_content.replace(
                "from builtins import PyDataFrame",
                "from polars.polars import PyDataFrame",
            )

        case "lazyframe":
            if (txt := "P: Incomplete") in stub_content:
                stub_content = "from typing_extensions import ParamSpec, Generic\n" + stub_content.replace(
                    txt, "P = ParamSpec('P')"
                ).replace("class LazyFrame", "class LazyFrame(Generic[P])")
            stub_content = stub_content.replace("_ldf: Incomplete", "_ldf: PyLazyFrame")
            stub_content = stub_content.replace(
                "from builtins import PyLazyFrame",
                "from polars.polars import PyLazyFrame",
            )
        case "expr":
            if (txt := "P: Incomplete") in stub_content:
                stub_content = "from typing_extensions import ParamSpec, Generic\n" + stub_content.replace(
                    txt, "P = ParamSpec('P')"
                ).replace("class Expr", "class Expr(Generic[P])")
            stub_content = stub_content.replace("_pyexpr: Incomplete", "_pyexpr: PyExpr")
            stub_content = stub_content.replace("from builtins import PyExpr", "from polars.polars import PyExpr")
        case "series":
            array_like = (
                'ArrayLike = Union[Sequence[Any], "Series", '
                '"pa.Array", "pa.ChunkedArray", "np.ndarray", "pd.Series", "pd.DatetimeIndex"]'
            )
            if (incomplete_str := "ArrayLike: Incomplete") in stub_content:
                stub_content = stub_content.replace(incomplete_str, array_like)
            if "class SeriesIter:" in stub_content:
                stub_content = stub_content.replace("len: Incomplete", "len: int").replace("s: Incomplete", "s: Series")
            stub_content = stub_content.replace("from builtins import PySeries", "from polars.polars import PySeries")
            stub_content = stub_content.replace("_s: Incomplete", "_s: PySeries")
            stub_content = stub_content.replace("_accessors: _ClassVar[set]", "_accessors: _ClassVar[set[str]]")
        case err:
            raise ValueError(err)
    stub_content = stub_content.replace("from _typeshed import Incomplete", "")
    stub_content = f"#: version {version}\n" + stub_content
    return stub_content


def is_incomplete(path: Path):
    return "Incomplete" in path.read_text()


def main(tmp_dir: Path):
    versions = get_missing_versions()
    print(f"Missing versions: {versions}")
    for version_ in versions.union([newest_version_not_in(versions)]):
        output_dir = Path("src", "polugins_type_gen", "_stubs", str(version_))
        no_docstring_output_dir = Path("no_docstring", str(version_))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_stubgen(str(version_), no_docstring_output_dir, output_dir, tmp_dir)
        for import_path in IMPORT_PATHS:
            stub_path = output_dir / import_path.with_suffix(".pyi")
            cleaned_stub_content = clean_types(stub_path, version_)
            stub_path.with_suffix(".pyi").write_text(cleaned_stub_content)
            # run ruff on the cleaned stub
            subprocess.check_call([sys.executable, "-m", "ruff", "format", str(stub_path)])
            if is_incomplete(stub_path):
                msg = f"File {stub_path} could not be cleaned and has Incomplete types."
                print(msg)
            no_docstring_stub_path = tmp_dir / no_docstring_output_dir / import_path.with_suffix(".pyi")
            cleaned_no_docstring_stub_content = clean_types(no_docstring_stub_path, version_)
            no_docstring_stub_path.write_text(cleaned_no_docstring_stub_content)
            subprocess.check_call([sys.executable, "-m", "ruff", "format", str(no_docstring_stub_path)])

    return versions


def diff_chunk(content: str):
    return f"```diff\n{content}\n```\n"


def comparison_section(
    version_1: version.Version,
    version_2: version.Version,
    comparisons: list[tuple[str, str]],
):
    header = f"# Changes from {version_1} to {version_2}\n"
    body = ""
    for extension_class, diff in comparisons:
        body += f"## {extension_class}\n{diff_chunk(diff)}"
    return header + body


def newest_version_not_in(versions: set[version.Version]) -> version.Version:
    current_versions = get_current_versions()
    return max(current_versions - versions)


def create_pr_body(versions: set[version.Version], tempdir_path: Path):
    newest_current_version = newest_version_not_in(versions)

    comparisons = {
        (version_1, version_2): compare_versions(version_1, version_2, tempdir_path)
        for version_1, version_2 in pairwise(sorted(versions.union([newest_current_version])))
    }
    header = "# Automatic stub gen\n Changes between new versions and last:\n"
    body = "\n".join(
        comparison_section(version_1, version_2, comparison)
        for (version_1, version_2), comparison in comparisons.items()
    )
    return header + body


def compare_versions(version_1: version.Version, version_2, tempdir_path: Path) -> list[tuple[str, str]]:
    results = []
    stub_dir_1 = tempdir_path / "no_docstring" / str(version_1)
    stub_dir_2 = tempdir_path / "no_docstring" / str(version_2)
    for extension_class in IMPORT_PATHS:
        stub_path1 = stub_dir_1 / extension_class.with_suffix(".pyi")
        stub_path2 = stub_dir_2 / extension_class.with_suffix(".pyi")
        result = "\n".join(
            unified_diff(
                stub_path1.read_text().splitlines(),
                stub_path2.read_text().splitlines(),
                fromfile=str(version_1),
                tofile=str(version_2),
            )
        )
        results.append((extension_class.parts[-2], result))
    return results


if __name__ == "__main__":
    with TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        new_versions = main(tempdir_path)
        body_content = create_pr_body(new_versions, tempdir_path)

        body_path = Path(sys.argv[1]) if len(sys.argv) > 1 else tempdir_path / "pr_body.md"
        print(f"Wrinting pr template to: {body_path}")

        body_path.write_text(body_content)

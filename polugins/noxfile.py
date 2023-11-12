"""Nox sessions.
From https://github.com/cjolowicz/cookiecutter-hypermodern-python-instance
"""
from pathlib import Path
from typing import Iterable

import nox

python_versions = ["3.12", "3.11", "3.10", "3.9"]


def install(session: nox.Session, *, groups: Iterable[str], root: bool = True) -> None:
    """Install the dependency groups using Poetry.

    This function installs the given dependency groups into the session's
    virtual environment. When ``root`` is true (the default), the function
    also installs the root package and its default dependencies.

    To avoid an editable install, the root package is not installed using
    ``poetry install``. Instead, the function invokes ``pip install .``
    to perform a PEP 517 build.

    Args:
        session: The Session object.
        groups: The dependency groups to install.
        root: Install the root package.
    """
    session.run(
        "poetry",
        "install",
        # "--no-root",
        "--sync",
        "--{}={}".format("only" if not root else "with", ",".join(groups)),
        external=True,
    )
    if root:
        session.install(".")


@nox.session(python=python_versions)
def test(session: nox.Session) -> None:
    """Run the test suite."""
    install(session, groups=["dev"], root=True)
    try:
        session.run("pytest", "--cov=polugins", *session.posargs)
        coverage_file = Path(".coverage")
        coverage_file.rename(coverage_file.with_suffix("." + session.python))

    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@nox.session(python=python_versions[0])
def coverage(session: nox.Session) -> None:
    session.install("coverage")
    session.run("coverage", "combine")
    session.run("coverage", "report")

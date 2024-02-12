from pathlib import Path

import pytest

stubs_dir = Path(__file__).parent.parent / Path("src", "polugins_type_gen", "_stubs")
all_stub_files = sorted(p for p in stubs_dir.rglob("*") if p.is_file())


def get_id(p: Path):
    return f"{p.parts[-4]}_{p.parts[-1]}"


@pytest.mark.parametrize("stub_path", all_stub_files, ids=get_id)
def test_stubs(stub_path: Path):
    assert "Incomplete" not in stub_path.read_text()

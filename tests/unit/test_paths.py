from corpbond_rv.utils.paths import project_root


def test_project_root_contains_pyproject():
    root = project_root()
    assert (root / "pyproject.toml").exists()

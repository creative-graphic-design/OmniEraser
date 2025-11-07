import pathlib

import pytest


@pytest.fixture
def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parents[2]


@pytest.fixture
def example_dir(project_root: pathlib.Path) -> pathlib.Path:
    return project_root / "example"


@pytest.fixture
def img_dir(example_dir: pathlib.Path) -> pathlib.Path:
    return example_dir / "image"


@pytest.fixture
def msk_dir(example_dir: pathlib.Path) -> pathlib.Path:
    return example_dir / "mask"


@pytest.fixture
def output_base_dir(project_root: pathlib.Path) -> pathlib.Path:
    output_path = project_root / "output"
    output_path.mkdir(exist_ok=True)
    return output_path

import io
import pathlib
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client() -> Iterator[TestClient]:
    from omnieraser.api.run import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    argnames=("img_name", "msk_name"),
    argvalues=(
        ("image (1).jpg", "image (1).png"),
        # (),
        # (),
    ),
)
def test_erase_control(
    client: TestClient,
    img_dir: pathlib.Path,
    msk_dir: pathlib.Path,
    img_name: str,
    msk_name: str,
):
    img = img_dir / img_name
    msk = msk_dir / msk_name

    with open(img, "rb") as img_file, open(msk, "rb") as msk_file:
        response = client.post(
            "/erase/controlnet",
            files={
                "image_file": ("image.jpg", img_file, "image/jpeg"),
                "mask_image_file": ("mask.png", msk_file, "image/png"),
            },
            params={"dilation_kernel_size": 5},
        )

    response.raise_for_status()

    result_image = Image.open(io.BytesIO(response.content))
    result_image.save("hoge.png")

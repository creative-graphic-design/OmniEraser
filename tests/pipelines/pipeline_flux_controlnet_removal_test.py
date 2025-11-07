import pathlib

import pytest
import torch
from diffusers.utils.loading_utils import load_image

from omnieraser.models import FluxControlNetModel, FluxTransformer2DModel
from omnieraser.pipelines import FluxControlNetInpaintingPipeline


@pytest.fixture
def output_dir(output_base_dir: pathlib.Path) -> pathlib.Path:
    output_path = output_base_dir / "controlnet"
    output_path.mkdir(exist_ok=True)
    return output_path


@pytest.fixture
def torch_dtype() -> torch.dtype:
    return torch.bfloat16


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def controlnet_model_name() -> str:
    return "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"


@pytest.fixture
def flux_model_name() -> str:
    return "black-forest-labs/FLUX.1-dev"


@pytest.fixture
def prompt() -> str:
    return "There is nothing here."


@pytest.fixture
def resize_size() -> tuple[int, int]:
    return (1024, 1024)


@pytest.fixture
def seed() -> int:
    return 24


@pytest.mark.parametrize(
    argnames="img_name",
    argvalues=(
        "0bce8e90-10f1-442e-8330-2917fc7fa486.png",
        "0e5124d8-fe43-4b5c-819f-7212f23a6d2a.png",
        "0f900fe8-6eab-4f85-8121-29cac9509b94.png",
        "1a8e07fa-0138-4c80-90e5-7cd020f6fa2c.png",
        "1dca0631-881a-4c70-a18f-c34b049db463.png",
        "1ea0ad33-d348-467e-a44a-0d8cea188a7c.png",
        "3c43156c-2b44-4ebf-9c47-7707ec60b166.png",
        "3ed1ee18-33b0-4964-b679-0e214a0d8848.png",
        "4bb11bef-d76f-476f-8791-d25d6e7421f5.png",
        "4cf0a0c9-7b0d-4d1a-b1f6-ba1805d90712.png",
        "6a6a6d7b-a8c7-4421-b997-797dcd47813b.png",
        "7b4846a3-5e5d-4a68-8278-500ec7d6dcd2.png",
    ),
)
def test_pipeline_flux_controlnet_removal(
    prompt: str,
    resize_size: tuple[int, int],
    img_name: str,
    img_dir: pathlib.Path,
    msk_dir: pathlib.Path,
    output_dir: pathlib.Path,
    seed: int,
    device: torch.device,
    torch_dtype: torch.dtype,
    flux_model_name: str,
    controlnet_model_name: str,
    lora_weight_path: str = "theSure/Omnieraser_Controlnet_version",
    lora_weight_name: str = "controlnet_flux_pytorch_lora_weights.safetensors",
):
    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_model_name,
        torch_dtype=torch_dtype,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        flux_model_name,
        subfolder="transformer",
        torch_dtype=torch_dtype,
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        flux_model_name,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)

    pipe.load_lora_weights(
        lora_weight_path,
        weight_name=lora_weight_name,
    )
    pipe.transformer.to(torch_dtype)
    pipe.controlnet.to(torch_dtype)

    # import cv2
    # import numpy as np
    # from PIL import Image

    # mask = cv2.imread(str(msk_dir / img_name), cv2.IMREAD_GRAYSCALE)
    # mask = Image.fromarray(mask.astype(np.uint8).repeat(3, -1)).convert("RGB")
    # mask = mask.resize((1024, 1024))

    img = load_image(image=str(img_dir / img_name))
    orig_size = img.size
    img = img.resize(resize_size)

    msk = load_image(image=str(msk_dir / img_name))
    msk = msk.resize(resize_size)

    result = pipe(
        prompt=prompt,
        width=resize_size[0],
        height=resize_size[1],
        control_image=img,
        control_mask=msk,
        num_inference_steps=28,
        true_guidance_scale=1.0,
        guidance_scale=3.5,
        generator=torch.manual_seed(seed),
        controlnet_conditioning_scale=0.9,
    )
    image = result.images[0]
    image = image.resize(orig_size)

    image.save(output_dir / img_name)

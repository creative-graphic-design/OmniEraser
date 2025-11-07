import io
from contextlib import asynccontextmanager
from typing import Final, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Request, Response, UploadFile
from PIL import Image

from omnieraser.models import FluxControlNetModel, FluxTransformer2DModel
from omnieraser.pipelines import FluxControlNetInpaintingPipeline

CONTROLNET_MODEL_NAME: Final[str] = (
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"
)

FLUX_MODEL_NAME: Final[str] = "black-forest-labs/FLUX.1-dev"

LORA_WEIGHT_PATH: Final[str] = "theSure/Omnieraser_Controlnet_version"
LORA_WEIGHT_NAME: Final[str] = "controlnet_flux_pytorch_lora_weights.safetensors"

RESIZE_SIZE: Final[Tuple[int, int]] = (1024, 1024)
PROMPT: Final[str] = "There is nothing here."


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16

    controlnet = FluxControlNetModel.from_pretrained(
        CONTROLNET_MODEL_NAME,
        torch_dtype=torch_dtype,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        FLUX_MODEL_NAME,
        subfolder="transformer",
        torch_dtype=torch_dtype,
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        FLUX_MODEL_NAME,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)

    pipe.load_lora_weights(
        LORA_WEIGHT_PATH,
        weight_name=LORA_WEIGHT_NAME,
    )
    pipe.transformer.to(torch_dtype)
    pipe.controlnet.to(torch_dtype)

    app.state.pipe = pipe

    yield


app = FastAPI(lifespan=lifespan)


async def load_image(image_file: UploadFile) -> Image.Image:
    data = await image_file.read()
    image = Image.open(io.BytesIO(data))
    return image.convert("RGB")


async def load_mask_image(mask_image_file: UploadFile) -> np.ndarray:
    data = await mask_image_file.read()

    mask_image_np = cv2.imdecode(
        buf=np.frombuffer(data, np.uint8),
        flags=cv2.IMREAD_GRAYSCALE,
    )
    return mask_image_np


def apply_mask_dilation(mask_image: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask_image, kernel, iterations=1)
    return dilated_mask


@app.get("/health")
def health_check() -> Response:
    return Response(content="OK", status_code=200)


@app.post(
    "/erase/controlnet",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def apply_omni_eraser(
    request: Request,
    image_file: UploadFile,
    mask_image_file: UploadFile,
    seed: int = 24,
    dilation_kernel_size: Optional[int] = None,
) -> Response:
    orig_image = await load_image(image_file)

    mask_image_arr = await load_mask_image(mask_image_file)

    if dilation_kernel_size is not None:
        mask_image_arr = apply_mask_dilation(mask_image_arr, dilation_kernel_size)

    mask_image = Image.fromarray(mask_image_arr.astype(np.uint8).repeat(3, -1))
    mask_image = mask_image.convert("RGB")

    orig_size = orig_image.size
    orig_image = orig_image.resize(RESIZE_SIZE)
    mask_image = mask_image.resize(RESIZE_SIZE)

    pipe: FluxControlNetInpaintingPipeline = request.app.state.pipe

    result = pipe(
        prompt=PROMPT,
        width=RESIZE_SIZE[0],
        height=RESIZE_SIZE[1],
        control_image=orig_image,
        control_mask=mask_image,
        num_inference_steps=28,
        true_guidance_scale=1.0,
        guidance_scale=3.5,
        generator=torch.manual_seed(seed),
        controlnet_conditioning_scale=0.9,
    )
    image = result.images[0]
    image = image.resize(orig_size)

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")

    return Response(content=image_bytes.getvalue(), media_type="image/png")

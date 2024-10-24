from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from diffusers import StableDiffusion3Pipeline
from io import BytesIO
from PIL import Image
import base64
import os
from datetime import datetime

app = FastAPI()

# Initialize the Stable Diffusion pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16)
pipe = pipe.to("mps")

class ImageRequest(BaseModel):
    prompt: str
    n: Optional[int] = 4

@app.post("/v1/images/generations")
async def generate_stable_diffusion(request: ImageRequest):
    try:
        image = pipe(
            request.prompt,
            guidance_scale=0.0,
            num_inference_steps=request.n,
        ).images[0]

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Save the image in the images directory with the current date and time as the filename
        if not os.path.exists("images"):
            os.makedirs("images")
        filename = datetime.now().strftime(
            "images/" + request.prompt.replace(" ", "_") + "_%Y%m%d_%H%M%S.png")
        with open(filename, "wb") as f:
            f.write(buffered.getvalue())

        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

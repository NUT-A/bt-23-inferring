import base64
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
import asyncio
import concurrent.futures
import time
import uvicorn
from model import AnimeModel, GenerateRequest
from print_info import print_stats
from PIL import Image
import io

import logging

class EndpointFilter(logging.Filter):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find(self.path) == -1

model = AnimeModel()
app = FastAPI()

# Apply the filter to the Uvicorn access logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(EndpointFilter("/healthcheck"))

# Create a semaphore to limit concurrency
semaphore = asyncio.Semaphore(1)
    
def pil_image_to_base64(image: Image.Image, format="JPEG") -> str:
    if format not in ["JPEG", "PNG"]:
        format = "JPEG"
    image_stream = io.BytesIO()
    image = image.convert("RGB")
    image.save(image_stream, format=format)
    base64_image = base64.b64encode(image_stream.getvalue()).decode("utf-8")
    return base64_image

@app.post("/generate_anime")
async def generate(request: GenerateRequest):
    if request.model_name != "AnimeV3":
        raise HTTPException(status_code=400, detail="Model name must be AnimeV3")
    
    request_time = time.time()
    
    # Define a function to perform the inference
    def inference():
        return model.infer(request.prompt, request.seed, request.pipeline_params)

    # Use a semaphore to limit concurrency
    async with semaphore:
        # Use a timeout to limit the execution time
        try:
            remaining_time = 12.0 - (time.time() - request_time)
            
            if remaining_time < 1: # Generation time
                raise HTTPException(status_code=504, detail="Request timed out after 12 seconds")
            
            image = await asyncio.wait_for(run_in_threadpool(inference), timeout=remaining_time)
            # base_64_image = base64.b64encode(image).decode('utf-8')
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out after 12 seconds")
        except concurrent.futures.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out after 12 seconds")

    return {"image": pil_image_to_base64(image)}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

if __name__ == "__main__":
    print_stats()
    model.preheat_model()
    
    uvicorn.run(app, host="0.0.0.0", port=3000, timeout_keep_alive=12)
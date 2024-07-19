import logging
import os
from io import BytesIO

import aiohttp
import oaas_sdk_py as oaas
import uvicorn
from PIL import Image
from fastapi import Request, FastAPI, HTTPException
from oaas_sdk_py import OaasInvocationCtx
from rembg import remove, new_session

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
level = logging.getLevelName(LOG_LEVEL)
logging.basicConfig(level=level)

IMAGE_KEY = "image"

rembg_session = new_session("u2net").download_models()


class RemoveBackgroundHandler(oaas.Handler):
    async def handle(self, ctx: OaasInvocationCtx):
        inplace = ctx.task.output_obj is None or ctx.task.output_obj.id is None
        async with aiohttp.ClientSession() as session:
            async with await ctx.load_main_file(session, IMAGE_KEY) as resp:
                image_bytes = await resp.read()
                with Image.open(BytesIO(image_bytes)) as img:
                    output_image = remove(img)
                    byte_io = BytesIO()
                    output_image.save(byte_io, format=img.format)
                    resized_image_bytes = byte_io.getvalue()
                    if inplace:
                        await ctx.upload_main_byte_data(session, IMAGE_KEY, resized_image_bytes)
                    else:
                        await ctx.upload_byte_data(session, IMAGE_KEY, resized_image_bytes)


app = FastAPI()
router = oaas.Router()
router.register(RemoveBackgroundHandler())


@app.post('/')
async def handle(request: Request):
    body = await request.json()
    logging.debug("request %s", body)
    resp = await router.handle_task(body)
    logging.debug("completion %s", resp)
    if resp is None:
        raise HTTPException(status_code=404, detail="No handler matched")
    return resp


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)

# python -m uvicorn main:app --reload --port 8010
import io
import os
import base64
import subprocess
import time
import tensorflow as tf

import cv2
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse, FileResponse, StreamingResponse
from cihp_pgn.inf_pgn import get_humanparse

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/parse", status_code=200)
async def human_parse(file: UploadFile = File(...)):
    print(" >>>>> Let's start Human Parse !")
    init = tf.global_variables_initializer()
    sess = tf.compat.v1.Session()
    try:
        sess.run(init)
        
        model_fname = f"cihp_pgn/data/image/{file.filename}"
        os.chmod(model_fname, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        with open(model_fname, "wb") as f:
            f.write(file.file.read())
        img_name, img_ext = os.path.splitext(file.filename)
        os.chdir("cihp_pgn")
        get_humanparse(file.filename)
        os.chdir("../")
        await server_reload()
        parsing_maps_path = Image.open(f"cihp_pgn/output/cihp_parsing_maps/{img_name}.png")
        parsing_color_path = Image.open(f"cihp_pgn/output/parse_image/{img_name}_color.png")
        parsing_maps_converted = from_image_to_bytes(parsing_maps_path)
        parsing_color_converted = from_image_to_bytes(parsing_color_path)
        img_list = [parsing_maps_converted, parsing_color_converted]

    finally:
        sess.close()
        return JSONResponse(img_list)


def from_image_to_bytes(img):
    # Pillow 이미지 객체를 Bytes로 변환
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format=img.format)
    imgByteArr = imgByteArr.getvalue()
    # Base64로 Bytes를 인코딩
    encoded = base64.b64encode(imgByteArr)
    # Base64로 ascii로 디코딩
    decoded = encoded.decode('ascii')
    return decoded


async def server_reload():
    time.sleep(5)
    subprocess.Popen(["python", "-m", "uvicorn", "main:app", "--reload", "--port", "8010"])


""" 
# test code
@app.post("/human-parse", status_code=200)
async def human_parse(request_body: Optional[Dict[str, Any]] = None):
    model_fname = request_body.get('model_fname')
    os.chdir("cihp_pgn")
    start = time.time()
    get_humanparse(model_fname)
    print(f"human parse 소요시간 : {round((time.time() - start), 2)}")
    os.chdir("../")
    return JSONResponse(status_code=200,
                        content={"msg": "human-parse done"})
"""

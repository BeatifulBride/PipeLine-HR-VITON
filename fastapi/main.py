#  python -m uvicorn main:app --reload --port 8000
import asyncio

import cv2
import os
import json

from typing import List

import io


import pose
import base64
import requests
import subprocess
from PIL import Image
from io import BytesIO
from starlette.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi import FastAPI, File, UploadFile,Form

from agnostics.get_img_agnostic import getting_img_agnostic
from agnostics.get_parse_agnostic import get_parse_agnostic
from utils.time import current_time
from cloth_mask import get_cloth_mask
from humanparse.get_human_parse import get_human_parse
from img_paste import merge

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(os.getcwd())
@app.get("/")
async def home():
    return HTMLResponse(content=f"""
    <body>
    <div>
        <h1 style="width:400px;margin:50px auto">
            {current_time()} <br/>
            현재 서버 구동 중 입니다. 
         </h1>
    </div>
    </body>
        """)


@app.get("/image-resize")
async def async_image_resize():

    print(" >>>>> Image Resize ! ")
    image = cv2.imread("data/test/image/test_1.jpg")
    width = 768
    height = 1024
    dim = (width, height)
    resized_image = cv2.resize(image, dim)
    cv2.imwrite("data/test/image/test_1.jpg", resized_image)


@app.post("/cloth-preprocess")
async def cloth_preprocess(image: UploadFile = File(...)):

    print(" >>>>> Start Cloth Preprocss !")
    cloth_path = "data/test/cloth"
    cloth_fname = 'test_1.jpg'
    cloth_location = os.path.join(cloth_path, cloth_fname)
    print(os.listdir(os.path.join(cloth_paeth)))

    with open(cloth_location, "wb") as buffer:
        buffer.write(await image.read())
    img = cv2.imread(cloth_location)
    resized_img = cv2.resize(img, (768, 1024))
    cv2.imwrite(cloth_location, resized_img)

    get_cloth_mask(cloth_fname)

    with open(f"data/test/cloth-mask/{cloth_fname}", "rb") as img_file:
        image_bytes = img_file.read()

    output = base64.b64encode(image_bytes).decode()
    print(" >>>>> Success Cloth Preprocess !")
    os.chdir("../")
    return {"data": output}


@app.post("/human-parse")
async def async_human_parse(img_name):
    print(" >>>>> Start Preprocess :: Human Parse")
    print(" >>>>> Request Human Parse to 8010 port server")
    print(os.getcwd())
    image_path = f"data/test/image/{img_name}"
    humanparse_url = "http://127.0.0.1:8010/parse"
    with open(image_path, "rb") as f:
        files = {"file": (img_name, f)}
        name, ext = img_name.split('.')
        result = requests.post(humanparse_url, files=files)
        if result.status_code == 200:
            response = json.loads(result.content)
            img_parse_maps = Image.open(BytesIO(base64.b64decode(response[0])))
            img_parse_color = Image.open(BytesIO(base64.b64decode(response[1])))
            parse_path = "data/test/image-parse-v3"
            parse_maps = os.path.join(parse_path, f"{name}.png")
            parse_color = os.path.join(parse_path, f"{name}_color.png")
            img_parse_maps.save(parse_maps)
            img_parse_color.save(parse_color)
            #im = Image.fromarray(im).convert('L')

            print(" >>>>> Return Human Parse Image successfully")
            print(" >>>>> Success Human Parse ! ")
        else:
            print("Error occurred during image transfer")


@app.get("/densepose")
async def async_densepose():
    print(" >>>>> Start Preprocess :: Densepose")
    terminnal_command = "python DensePose/apply_net.py " \
                        "show DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml " \
                        "DensePose/model_densepose.pkl data/test/image " \
                        "dp_segm -v --opts MODEL.DEVICE cpu"
    os.system(terminnal_command)
    os.chdir("../")
    print(" >>>>> Success Densepose ! ")


@app.get("/openpose")
async def async_openpose():
    os.chdir(r"D:\myProjects\project-Howsfit\fastapi\openpose\openpose")

    # os.chdir("D:\myProjects\project-Howsfit\fastapi\openpose\openpose")
    dir = os.getcwd()
    print(dir)
    # bin\OpenPoseDemo.exe --image_dir example\media --write_json output\json

    subprocess.call(['/openpose/bin/OpenPoseDemo.exe',
                     '--image_dir', r'C:\Users\hi\PycharmProjects\hr-vton-pipline\fastapi\data\test\image',
                     '--write_json', r'C:\Users\hi\PycharmProjects\hr-vton-pipline\fastapi\data\test\openpose_json',
                     '--write_images', r'C:\Users\hi\PycharmProjects\hr-vton-pipline\fastapi\data\test\openpose_img',
                     '--display', '0'])

    #
    # print(" >>>>> Start Preprocess :: Openpose")
    # os.chdir("./openpose")
    # subprocess.call('artifacts/bin/OpenPoseDemo.exe '
    #                 '--image_dir ../data/test/image '
    #                 '--write_json ../data/test/openpose_json '
    #                 '--write_images ../data/test/openpose_img '
    #                 '--display 0'
    #                 )
    os.chdir("../")
    os.chdir("C:\\Users\\hi\\PycharmProjects\\hr-vton-pipline\\fastapi")
    print(os.getcwd())
    print(" >>>>> Success Openpose ! ")


@app.get("/agnostic")
async def async_agnostic():
    print(" >>>>> Start Preprocess :: Agnostics")
    print(os.getcwd())
    os.chdir("./agnostics")
    getting_img_agnostic("test_1.jpg")
    get_parse_agnostic("test_1.jpg")
    print(os.getcwd())
    os.chdir("../")
    print(" >>>>> Success Agnostics ! ")


@app.post("/model-preprocess-local")
async def model_preprocess_local(image: UploadFile = File(...)):
    print(" >>>>> Start Model Preprocess !")
    model_path = "data/test/image"
    model_fname = "test_1.jpg"
    name, ext = model_fname.split('.')
    model_location = os.path.join(model_path, model_fname)
    with open(model_location, "wb") as buffer:
        buffer.write(await image.read())

    # model preprocess 0 :: image-resize
    await async_image_resize()

    # model preprocess 1 :: human-parse
    await async_human_parse(model_fname)

    # model preprocess 2 :: Densepose
    await async_densepose()

    # model preprocess 3 :: Openpose
    await async_openpose()
    print(os.getcwd())
    # model preprocess 4 :: Agnostic
    await async_agnostic()

    merge(f'./data/test/image-densepose/{model_fname}',
          f'./data/test/openpose_img/{name}_rendered.png',
          f'./data/test/image-parse-v3/{name}_color.png',
          f'./data/test/agnostic-v3.2/{name}.jpg'
          )
    print(" >>>>> Success All Preprocess ! ")

    with open(f"./data/merged_img/merged_image.jpg", "rb") as img_file:
        image_bytes = img_file.read()
    output = base64.b64encode(image_bytes).decode()

    print(" >>>>> Success return preprocess image to Web Page !")

    return {"data": output}


@app.post("/model-preprocess-aws-cpu")
async def model_preprocess_cpu(image: UploadFile = File(...)):
    model_path = "data/test/image"
    model_fname = "test_1.jpg"
    file_location = os.path.join(model_path, model_fname)
    with open(file_location, "wb") as buffer:
        buffer.write(await image.read())
    img = cv2.imread(file_location)
    resized_img = cv2.resize(img, (768, 1024))
    cv2.imwrite(file_location, resized_img)

    os.chdir("./humanparse")
    get_human_parse(model_fname)
    os.chdir("../")
    print('finish')

    pose.get_posenet(file_location)

    terminnal_command = "python DensePose/apply_net.py " \
                        "show DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml " \
                        "DensePose/model_densepose.pkl " \
                        r"data/test/image dp_segm -v --opts MODEL.DEVICE cpu"
    os.system(terminnal_command)

    os.chdir("./agnostics")
    getting_img_agnostic(model_fname)
    get_parse_agnostic(model_fname)
    os.chdir("../")

    #remove_back(file_location)

    merge('./data/test/image-densepose/test_1.jpg',
          './data/test/openpose_img/test_1_rendered.png',
          './data/test/image-parse-v3/test_1_color.png',
          './data/test/agnostic-v3.2/test_1.jpg')

    with open(f"./data/merged_img/merged_image.jpg", "rb") as img_file:
        print(img_file)
        image_bytes = img_file.read()

    output = base64.b64encode(image_bytes).decode()
    return {"data": output}


@app.post("/try-on")
async def try_on():

    print(" >>>>> Start Virtual Try On !")
    weight_path = "try_on/eval_models/weights/v0.1"
    terminnal_command = f"python try_on/my_test_generator.py "
    # weight_path = "try_on/eval_models/weights/v0.1"
    # terminnal_command = f"python try_on/my_test_generator.py " \
    #                     f"--test_name viton " \
    #                     f"--tocg_checkpoint {weight_path}/tocg_final.pth " \
    #                     f"--gpu_ids 0 " \
    #                     f"--gen_checkpoint {weight_path}/gen_model_final.pth " \
    #                     f"--datasetting unpaired " \
    #                     f"--data_list test_pairs.txt " \
    #                     f"--dataroot ./data"
    os.system(terminnal_command)

    print(" >>>>> Success Virtual Try On ! ")

    fname = 'test_1_test_1.png'
    with open(f"output/viton/{fname}", "rb") as img_file:
        image_bytes = img_file.read()

    output = base64.b64encode(image_bytes).decode()

    print(" >>>>> Success return VITON image to Web Page !")
    return {"data": output}




#json형식으로
#dress index값과
#이 dress index는 spring에서 이미지파일을 가져와야 한다!
#이 두개의 이미지를 전달받아 사용하는데 이
#이미지파일을 전달받는다
#현재 출력되는 이미지를 return하여 던져주어야 한다
@app.post("/getimg")
async def get_img(
    multipartfile: UploadFile = File(...),
    #dressCompanyName: str = Form(...),
    dressIndex: str = Form(...),
    dressPath: str = Form(...)
):
    print(os.getcwd())
    model_path = "data/test/image"
    model_fname = "test_1.jpg"
    name, ext = model_fname.split('.')
    model_location = os.path.join(model_path, model_fname)
    with open(model_location, "wb") as buffer:
        buffer.write(await multipartfile.read())

    # await async_image_resize()
    # await asyncio.gather(
    #     async_densepose(),
    #     async_openpose(),
    #     # async_human_parse(model_fname)
    # )
    # await async_agnostic()
    # #확인용 어그노스틱 굳이 필요하진 않음
    #
    # #드레스 로직 처리하자 성근아
    # server_address = "http://lamdahi.iptime.org:6900/"
    # dresspath_url = server_address+dressPath
    # cloth_path = "data/test/cloth"
    # cloth_fname = 'test_1.jpg'
    # cloth_location = os.path.join(cloth_path, cloth_fname)
    # response = requests.get(dresspath_url)
    #
    # # 에러가 없다면 이미지를 저장
    # if response.status_code == 200:
    #     with open(cloth_location, 'wb') as file:
    #         file.write(response.content)
    #
    # img = cv2.imread(cloth_location)
    # resized_img = cv2.resize(img, (768, 1024))
    # cv2.imwrite(cloth_location, resized_img)
    #
    # get_cloth_mask(cloth_fname)
    # print(os.getcwd())
    # #os.chdir("../")
    print(os.getcwd())
    terminnal_command = f"python try_on/my_test_generator.py "
    os.system(terminnal_command)
    print(os.getcwd())
    fname = 'test_1_test_1.png'
    fname = '_029880_0.png'
    with open(f"try_on/output/viton/{fname}", "rb") as img_file:
        image_bytes = img_file.read()



    #image_content = await multipartfile.read()  # 이미지 파일의 내용을 읽습니다.
    print(multipartfile)
    #print(dressCompanyName)
    print(dressIndex)
    print(dressPath)
    return StreamingResponse(io.BytesIO(image_bytes), media_type=multipartfile.content_type)


    # return StreamingResponse(io.BytesIO(image_content), media_type=multipartfile.content_type)
    #return None
#def __main__():
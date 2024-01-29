import json
from os import path as osp
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
def draw_img_RGB(agnostic,processindex):
    # agnostic_np = np.array(agnostic.convert('RGB'))
    # agnostic_np = cv2.cvtColor(agnostic_np, cv2.COLOR_RGB2BGR)
    # cv2.imshow(processindex, agnostic_np)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()
    return None

def get_img_agnostic(im, parse, pose_data):

    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                  (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                   (parse_array == 12).astype(np.float32) +
                   (parse_array == 16).astype(np.float32) +
                   (parse_array == 17).astype(np.float32) +
                   (parse_array == 18).astype(np.float32) +
                   (parse_array == 19).astype(np.float32))

    agnostic = im.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)
    draw_img_RGB(agnostic,'1')
    #왼쪽 어깨 오른쪽 어깨의 길이
    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    #오른쪽 엉덩이~ 왼쪽엉덩이
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    #골반 중앙지점
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a / 16) + 1

    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
    draw_img_RGB(agnostic,'2')

    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        draw_img_RGB(agnostic, '3')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        draw_img_RGB(agnostic, '4')


    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    draw_img_RGB(agnostic,'5')
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    draw_img_RGB(agnostic,'6')
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    draw_img_RGB(agnostic,'7')
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')
    draw_img_RGB(agnostic,'8')
    #2 오른쪽 어깨 5 왼쪽 어깨 12왼쪽엉덩이 9오른쪽 엉덩이
    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')

    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    #agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    #draw_img(agnostic, 'head')


    #9,12 엉덩이
    #9 오른쪽엉덩이 10 오른쪽 무릎 11 오른쪽 발목
    # agnostic_draw.line([tuple(pose_data[i]) for i in [9, 10]], 'gray', width=r * 12)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9]], 'gray', width=r * 12)
    draw_img_RGB(agnostic,'9')
    # for i in [9, 10,11]:
    for i in [9]:

        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*6, pointy-r*6, pointx+r*6, pointy+r*6), 'gray', 'gray')
        draw_img_RGB(agnostic, '10')
    # agnostic_draw.line([tuple(pose_data[i]) for i in [10, 11]], 'gray', width=r * 12)
    draw_img_RGB(agnostic,'11')

    #12 왼쪽엉덩이,13 왼쪽 무릎,14 왼쪽 발목
    # for i in [12, 13,14]:
    for i in [12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*6, pointy-r*6, pointx+r*6, pointy+r*6), 'gray', 'gray')
    # agnostic_draw.line([tuple(pose_data[i]) for i in [12, 13]], 'gray', width=r * 12)
    agnostic_draw.line([tuple(pose_data[i]) for i in [12]], 'gray', width=r * 12)
    draw_img_RGB(agnostic,'12')
    # agnostic_draw.line([tuple(pose_data[i]) for i in [13, 14]], 'gray', width=r * 12)
    draw_img_RGB(agnostic,'13')


    # agnostic_draw.polygon([tuple(pose_data[i]) for i in [9, 12, 14, 11]], 'gray', 'gray')

    # agnostic_draw.polygon([tuple(pose_data[i]) for i in [9]], 'gray', 'gray')

    #2 오른쪽 어깨 5 왼쪽 어깨 12왼쪽엉덩이 9오른쪽 엉덩이

    # agnostic_draw.line([tuple(pose_data[i]) for i in [21, 24]], 'gray', width=r * 12)

    #
    # #발끝 마스킹
    # for i in [21, 24]:
    #     pointx, pointy = pose_data[i]
    #     agnostic_draw.ellipse((pointx-r*6, pointy-r*6, pointx+r*6, pointy+r*6), 'gray', 'gray')
    #

    parse_cloth =((parse_array == 5).astype(np.float32) +
                   (parse_array == 6).astype(np.float32) +
                   (parse_array == 7).astype(np.float32) +
                   (parse_array == 9).astype(np.float32) +
                   (parse_array == 12).astype(np.float32) +
                   (parse_array == 18).astype(np.float32) +
                   (parse_array == 3).astype(np.float32) +
                   (parse_array == 11).astype(np.float32) +
                  (parse_array == 16).astype(np.float32) +
                  (parse_array == 17).astype(np.float32) +
                   (parse_array == 19).astype(np.float32))

    cloth_mask = Image.fromarray(np.uint8(parse_cloth * 255), 'L')

    # cloth_mask를 회색으로 색칠
    cloth_mask_colored = Image.new("RGB", cloth_mask.size)
    cloth_mask_colored.paste('gray', mask=cloth_mask)

    # 원본 이미지에 마스크 적용
    agnostic.paste(cloth_mask_colored, (0, 0), cloth_mask)


    return agnostic


def getting_img_agnostic(model_fname):
    data_path = '../data/test'
    output_path = "../data/test/agnostic-v3.2"

    print(os.getcwd())
    os.makedirs(output_path, exist_ok=True)

    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        print(os.getcwd())
        print(im_name)
        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                print(pose_label)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                print(pose_data)
                pose_data = np.array(pose_data)
                print(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
                print(pose_data)
        except IndexError:
            continue

        # load parsing image
        im = Image.open(osp.join(data_path, 'image', im_name))
        label_name = im_name.replace('.jpg', '.png')
        im_label = Image.open(osp.join(data_path, 'image-parse-v3', label_name))

        agnostic = get_img_agnostic(im, im_label, pose_data)

        agnostic.save(osp.join(output_path, im_name))





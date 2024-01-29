from PIL import Image, ImageDraw, ImageFont

def merge(img1,img2,img3,img4):
    # 이미지 네 개 불러오기
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img3 = Image.open(img3)
    img4 = Image.open(img4)


    width, height = img1.size

    # 새로운 이미지 생성
    merged_img = Image.new('RGB', (width * 2, height * 2), color=(255, 255, 255))

    # 이미지 병합
    merged_img.paste(img1, (0, 0))
    merged_img.paste(img2, (width, 0))
    merged_img.paste(img3, (0, height))
    merged_img.paste(img4, (width, height))

    # 제목 폰트, 크기, 색상 설정
    #font = ImageFont.truetype('arial.ttf', 50)
    title_color = (0, 0, 0)
    title_color2 = (255, 255, 255)

    # 이미지 위에 제목 삽입
    draw = ImageDraw.Draw(merged_img)
    draw.text((10, 10), 'Densepose', fill=title_color2)
    draw.text((width + 10, 10), 'Openpose', fill=title_color)
    draw.text((10, height + 10), 'Human_Parse', fill=title_color2)
    draw.text((width + 10, height + 10), 'Agnostic', fill=title_color)
    merged_img = merged_img.resize((768, 1024))
    # 이미지 저장
    merged_img.save('./data/merged_img/merged_image.jpg')

if __name__ == '__main__':
    merge('./data/test/image-densepose/test_1.jpg',
          './data/test/openpose_img/test_1_rendered.png',
          './data/test/image-parse-v3/test_1_resize_color.png',
          './data/test/agnostic-v3.2/test_1.jpg')
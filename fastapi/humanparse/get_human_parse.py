import os
import cv2


def get_human_parse(fname):
    img = cv2.resize(cv2.imread(f'../data/test/image/{fname}'),
                     (300, 400))
    cv2.imwrite(f'../data/test/image/{fname[:-4]}_resize.jpg', img)
    terminnal_command = f'python inference.py ' \
                        f'--loadmodel inference.pth ' \
                        f'--img_path ../data/test/image/{fname[:-4]}_resize.jpg ' \
                        f'--output_path ../data/test/image-parse-v3 ' \
                        f'--output_name test_1'
    os.system(terminnal_command)

    #color resize & save
    color = cv2.resize(cv2.imread(f'../data/test/image-parse-v3/{fname[:-4]}_color.png'),
               (768, 1024), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(f'../data/test/image-parse-v3/{fname[:-4]}_color.png',color)

    #gray resize & save
    img2 = cv2.resize(cv2.imread(f'../data/test/image-parse-v3/{fname[:-4]}.png'),
                      (768, 1024), interpolation=cv2.INTER_LANCZOS4)

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'../data/test/image-parse-v3/{fname[:-4]}.png', img2)
    os.remove(f'../data/test/image/{fname[:-4]}_resize.jpg')



if __name__ == '__main__':
    folder_path = "../data/test/image"
    files = os.listdir(folder_path)
    get_human_parse('test_1.jpg')

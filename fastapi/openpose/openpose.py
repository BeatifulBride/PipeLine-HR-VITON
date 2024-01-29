import os


def get_openpose():
    terminnal_command = 'C:/new_proj/try_on/preprocess/openpose/artifacts/bin/OpenPoseDemo.exe ' \
                        '--image_dir ./test ' \
                        '--write_json ../openpose_json ' \
                        '--write_images ../openpose_img'
    os.system(terminnal_command)

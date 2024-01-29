import cv2
import numpy as np


def draw_img_RGB(agnostic,processindex):
    agnostic_np = np.array(agnostic.convert('RGB'))
    agnostic_np = cv2.cvtColor(agnostic_np, cv2.COLOR_RGB2BGR)
    cv2.imshow(processindex, agnostic_np)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()



def draw_img_GRAY(agnostic,processindex):
    agnostic_np = np.array(agnostic.convert('RGB'))
    agnostic_np = cv2.cvtColor(agnostic_np, cv2.COLOR_RGB2BGR)
    cv2.imshow(processindex, agnostic_np)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
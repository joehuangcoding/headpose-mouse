
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix

import matplotlib.pyplot as plt
from skimage import io
import numpy as np
# from IPython.display import Image


# load config
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

face_boxes = FaceBoxes()
tddfa = TDDFA(gpu_mode=False, **cfg)

    # Given a still image path and load to BGR channel
img = cv2.imread('./examples/inputs/emma.jpg')
if img is not None:
    # face detection
    boxes = face_boxes(img)
    print(f'Detect {len(boxes)} faces')
    print(boxes)
    # regress 3DMM params
    param_lst, roi_box_lst = tddfa(img, boxes)

    dense_flag = False
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

    viz_pose(img, param_lst, ver_lst, show_flag=True, wfp=None)

    plt.show(block=False)
    input("Press Enter to close the plot...")

cv2.waitKey(0)
cv2.destroyAllWindows()
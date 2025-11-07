import io
import argparse

import cv2
import easyocr

# CRAFT-pytorch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from data.CRAFT import craft_utils
from data.CRAFT import imgproc
from data.CRAFT import file_utils
from data.CRAFT.craft import CRAFT
import json
import zipfile

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def _net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):

    # resize
    img_resized, target_ratio = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    return boxes, polys
#CRAFT-pytorch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='food_data arguments')
    parser.add_argument('--test_image_in', default='ERROR', type=str, help='path to test image of receipt')
    parser.add_argument('--test_image_out', default='ERROR', type=str, help='path to write test image')
    parser.add_argument('--trained_model', default='ERROR', type=str, help='path to CRAFT model')
    parser.add_argument('--test_folder', default='ERROR', type=str, help='path test folder for CRAFT')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score CRAFT')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold CRAFT')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference for CRAFT')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference CRAFT')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio CRAFT')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type CRAFT')

    args = parser.parse_args()

    #TESTS

    # load CRAFT
    net = CRAFT()

    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # end load CRAFT


    _image = cv2.imread(args.test_image_in)
    _image_sample = _image[16:60,16:807]
    cv2.imwrite(args.test_image_out,_image_sample)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(_image_sample)

    for (bbox, text, prob) in result:
        print(f'Text: {text}, Probability: {prob}, box:{bbox}')

    print("'food_data' EXIT")

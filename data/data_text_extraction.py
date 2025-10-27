
import os
import json
import numpy as np
import cv2 as cv
import math
import pytesseract
from data import text_detection

def main(_config="config.json"):
    #Load config file [TODO: error handling]
    with open(os.path.join(os.path.dirname(__file__),_config)) as config_fp:
        config = json.load(config_fp)

    pytesseract.pytesseract.tesseract_cmd = config["tesseract_cmd"]


    #OpenCV
    text_detector = text_detection.CV_Text_Detector(
                                                    config["cv_text_detector_model"],
                                                    config["cv_image_resolution_W"],
                                                    config["cv_image_resolution_H"]
                                                   )
 
#TESTS
def main_test():
    main("test_config.json")
    print("'data_text_extraction' main_test() EXIT")

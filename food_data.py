import io
import argparse

import cv2
import easyocr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='food_data arguments')
    parser.add_argument('--test_image_in', default='ERROR', type=str, help='path to test image of receipt')
    parser.add_argument('--test_image_out', default='ERROR', type=str, help='path to write test image')

    args = parser.parse_args()

    #TESTS
    _image = cv2.imread(args.test_image_in)
    _image_sample = _image[16:60,16:807]
    cv2.imwrite(args.test_image_out,_image_sample)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(_image_sample)

    for (bbox, text, prob) in result:
        print(f'Text: {text}, Probability: {prob}, box:{bbox}')

    print("'food_data' EXIT")

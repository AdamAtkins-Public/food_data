import io
import argparse

import easyocr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='food_data arguments')
    parser.add_argument('--test_image', default='ERROR', type=str, help='path to test image of receipt')

    args = parser.parse_args()

    #TESTS
    reader = easyocr.Reader(['en'])
    result = reader.readtext(args.test_image)

    for (bbox, text, prob) in result:
        print(f'Text: {text}, Probability: {prob}')

    print("'food_data' EXIT")

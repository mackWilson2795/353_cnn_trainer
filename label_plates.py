#! /usr/bin/env python3
import cv2
import os
import sys

FILE_PATH = "/home/fizzer/plate_images"
DEST_PATH = "/home/fizzer/labelled_plates"

def main(args):
    file_list = os.listdir(FILE_PATH)
    for file in file_list:
        img = cv2.imread(f"{FILE_PATH}/{file}")
        print(f"{file}")
        cv2.imshow(f"Image", img)
        cv2.waitKey(1)
        label = input("Plate label: ")
        # TODO: slice and join
        output_name = f"{file[:-4]}_{label}.png"
        print(output_name)
        cv2.imwrite(f"{DEST_PATH}/{output_name}", img)

if __name__ == '__main__':
    main(sys.argv)
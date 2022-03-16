"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: img_process_demo.py
    Description: This file is a coding experiment for machine vision, it'll
        be packed into metaflow's image processor module in the future.

    Created by Melrose-Lbt 2022-3-16
"""
import os

from PIL import Image
import numpy as np

img_root = './img_process_data/'


def get_img_name(img_root):
    filelist = os.listdir(img_root)
    img = []
    for file in filelist:
        img.append(img_root+file)
    if img_root + '.DS_Store' in img:
        img.remove(img_root+'.DS_Store')
    return img


def move(image, dx, dy):
    """
        Move image to some direction. When dx > 0, it will move to right.
    When dy > 0, it will move to above.
    :param image: image data read by PIL Image.open
    :param dx: number of pixels
    :param dy: number of pixels
    Warning:
            dx and dy has to be integer.
    """
    if isinstance(dx, int) and isinstance(dy, int):
        pass
    else:
        raise TypeError("dx and dy should be integer!")
    image = np.array(image)
    print(image.shape)
    image = np.roll(image, dx, axis=1)
    if dx > 0:
        image[:, :dx] = 255
    else:
        image[:, image.shape[1]+dx:image.shape[1]] = 255
    image = np.roll(image, dy, axis=0)
    if dy > 0:
        image[:dy, :] = 255
    else:
        image[image.shape[0]+dy:image.shape[0], :] = 255
    return image


def dot(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    img1[img1 == 0] = 1
    img1[img1 == 255] = 0
    img2[img2 == 0] = 1
    img2[img2 == 255] = 0
    img = img1 * img2
    return img


def add(img1, img2):
    img1 = np.array(img1, dtype=int)
    img2 = np.array(img2, dtype=int)
    img = img1 + img2
    img[img > 255] = 255
    img = np.array(img, dtype=np.uint8)
    return img


def minus(img1, img2):
    img1 = np.array(img1, dtype=int)
    img2 = np.array(img2, dtype=int)
    img = img1 - img2
    img[img <= 0] = 0
    img = np.array(img, dtype=np.uint8)
    return img


if __name__ == "__main__":
    img_name = get_img_name(img_root)
    print(img_name)
    img1 = Image.open(img_name[3]).resize((400, 400))
    img2 = Image.open(img_name[4]).resize((400, 400))
    img = dot(img1, img2)
    img[img == 0] = 255
    img[img == 1] = 0
    img = Image.fromarray(img)
    img.show()

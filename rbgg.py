import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(
    description='Generating random backgrounds for a given image.')
parser.add_argument('path', help='path to an image to process')
parser.add_argument('-o', '--out', default='out', help='output directory')
parser.add_argument('-n', '--number', type=int, default=2,
                    help='number of images to generate')
parser.add_argument('-m', '--mask', action="store_true",
                    help='whether to save masks')
parser.add_argument('-mo', '--maskout', default='mask',
                    help='mask output directory')


def get_random_cutout(im):
    y = random.randint(0, im.shape[0]//15)
    x = random.randint(0, im.shape[1]//15)
    width = im.shape[1]-x - \
        random.randint(-(im.shape[1]-y)//15, (im.shape[1]-x)//15)
    height = im.shape[0]-y - \
        random.randint(-(im.shape[0]-y)//15, (im.shape[0]-y)//15)
    cutout = im[y:y+height, x:x+width]
    scale = min(random.random()+0.2, 1)
    cutout = cv2.resize(cutout, tuple(int(scale*x)
                                      for x in im.shape)[:-1][-1::-1])
    if random.random() > 0.8:
        blur_rate = max(1, random.randrange(
            min(cutout.shape[0], cutout.shape[1])//15))
        cutout = cv2.blur(cutout, (blur_rate, blur_rate))
    return cutout


def get_random_solid_background(im):
    random_color = [random.randrange(
        256), random.randrange(256), random.randrange(256)]
    return np.full((im.shape[0]+random.randrange(-im.shape[0]//4, im.shape[0]//3), im.shape[1]+random.randrange(-im.shape[1]//4, im.shape[1]//3), 3), random_color)


def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]
    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1],
                         1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * \
        background[y:y+h, x:x+w] + mask * overlay_image
    full_mask = np.full(background.shape, 0.)
    full_mask[y:y+h, x:x+w] = mask
    full_mask *= 255
    return background, full_mask


if __name__ == '__main__':
    args = parser.parse_args()
    im = cv2.imread(args.path, cv2.IMREAD_UNCHANGED)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.maskout, exist_ok=True)

    for i in range(args.number):
        cutout = get_random_cutout(im)
        background = get_random_solid_background(cutout)
        x = random.randrange(max(1, background.shape[1]-cutout.shape[1]))
        y = random.randrange(max(1, background.shape[0]-cutout.shape[0]))
        added_image, mask = overlay_transparent(background, cutout, x, y)
        added_image = added_image.astype('uint8')

        cv2.imwrite(f'{args.out}/{i}.jpg', added_image)
        if args.mask:
            cv2.imwrite(f'{args.maskout}/{i}.png', mask)

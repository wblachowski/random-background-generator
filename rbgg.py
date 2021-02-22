import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import argparse
import colorsys
import urllib.request

parser = argparse.ArgumentParser(
    description='Generating random backgrounds for a given image.')
parser.add_argument('path', help='path to an image to process')
parser.add_argument('-o', '--out-dir', default='out', help='output directory')
parser.add_argument('-n', '--number', type=int, default=2,
                    help='number of images to generate')
parser.add_argument('-p', '--photos', type=float, default=0.,
                    help='percentage of images to have photographs as background')
parser.add_argument('-m', '--mask', action="store_true",
                    help='whether to save masks')
parser.add_argument('-mo', '--mask-out-dir', default='mask',
                    help='mask output directory')
parser.add_argument('--margin-low', type=float, help='margin low')
parser.add_argument('--margin-high', type=float, help='margin high')
parser.add_argument('--margins-equal', action="store_true",
                    help='equal margins')
parser.add_argument('--scale-low', type=float, help='scale low')
parser.add_argument('--scale-high', type=float, help='scale high')
parser.add_argument('--blur-probability', type=float,
                    default=0., help='blur probability')
parser.add_argument('--blur-low', type=float, default=0.05,
                    help='blur strength low')
parser.add_argument('--blur-high', type=float,
                    help='blur strength high')


def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else c[1]
        slices.append(slice(-c[0], e))
    return x[tuple(slices)]


def get_random_cutout(image, scale_low, scale_high, margin_low, margin_high, margins_equal, blur_probability, blur_low, blur_high):
    if scale_low or scale_high:
        if not scale_low:
            scale_low = scale_high
        if not scale_high:
            scale_high = scale_low
        scale = random.uniform(scale_low, scale_high)
        image = cv2.resize(
            image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))

    wm_shape = image.shape

    if margin_low or margin_high:
        if not margin_low:
            margin_low = margin_high
        if not margin_high:
            margin_high = margin_low
        if margins_equal:
            margin = random.unform(margin_low/2, margin_high/2)
            a, b, c, d = int(margin*image.shape[0]), int(margin*image.shape[0]), int(
                margin*image.shape[1]), int(margin*image.shape[1])
            image = np.pad(image, ((max(0, a), max(0, b)),
                                   (max(0, c), max(0, d)), (0, 0)))
            image = unpad(image, ((a, b), (c, d), (0, 0)))
        else:
            def rndm(): return random.uniform(margin_low/2, margin_high/2)
            a, b, c, d = int(rndm()*image.shape[0]), int(rndm()*image.shape[0]), int(
                rndm()*image.shape[1]), int(rndm()*image.shape[1])
            image = np.pad(image, ((max(0, a), max(0, b)),
                                   (max(0, c), max(0, d)), (0, 0)))
            image = unpad(image, ((min(0, a), min(0, b)),
                                  (min(0, c), min(0, d)), (0, 0)))

    if random.random() <= blur_probability:
        if not blur_high:
            blur_high = blur_low
        if not blur_low:
            blur_low = blur_high
        blur_low, blur_high = int(blur_low*min(
            wm_shape[0], wm_shape[1])), int(blur_high*min(wm_shape[0], wm_shape[1]))
        blur = random.randint(blur_low, blur_high)
        image = cv2.blur(image, (blur, blur))

    return image


def get_random_solid_background(im):
    random_color = np.array(colorsys.hsv_to_rgb(
        random.random(), random.random(), random.random()))*255
    return np.full((im.shape[0], im.shape[1], 3), random_color.astype(np.int32))


def get_random_photo_background(im):
    with urllib.request.urlopen(f"https://picsum.photos/{im.shape[1]}/{im.shape[0]}") as url:
        image_array = np.asarray(bytearray(url.read()), dtype="uint8")
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image


def overlay_transparent(background, overlay):
    bg_height, bg_width = background.shape[0], background.shape[1]
    h, w = overlay.shape[0], overlay.shape[1]

    if w > bg_width:
        w = bg_width
        overlay = overlay[:, :w]

    if h > bg_height:
        h = bg_height
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

    background[:h, :w] = (1.0 - mask) * \
        background[:h, :w] + mask * overlay_image
    full_mask = np.full(background.shape, 0.)
    full_mask[:h, :w] = mask
    full_mask *= 255
    return background, full_mask


if __name__ == '__main__':
    args = parser.parse_args()
    im = cv2.imread(args.path, cv2.IMREAD_UNCHANGED)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.mask_out_dir, exist_ok=True)

    photo_bg_number = int(100*args.photos*args.number)
    solid_bg_number = args.number - photo_bg_number

    for i in range(args.number):
        cutout = get_random_cutout(im, args.scale_low, args.scale_high,
                                   args.margin_low, args.margin_high, args.margins_equal, args.blur_probability, args.blur_low, args.blur_high)
        background = get_random_solid_background(
            cutout) if i < solid_bg_number else get_random_photo_background(cutout)
        added_image, mask = overlay_transparent(background, cutout)
        added_image = added_image.astype('uint8')

        cv2.imwrite(f'{args.out_dir}/{i}.jpg', added_image)
        if args.mask:
            cv2.imwrite(f'{args.mask_out_dir}/{i}.png', mask)

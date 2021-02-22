import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import colorsys
from random import randint, uniform, random
from urllib.request import urlopen

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


class ImagePermutationGenerator:
    def __init__(self, img, args):
        self.img = img
        self.scale_processor = ScaleProcessor(args)
        self.margin_processor = MarginProcessor(args)
        self.blur_processor = BlurProcessor(args)

    def next(self):
        img = self.img.copy()
        shape_after_scaling = img.shape
        if self.scale_processor.should_process():
            img = self.scale_processor.process(img)
            shape_after_scaling = img.shape
        if self.margin_processor.should_process():
            img = self.margin_processor.process(img)
        if self.blur_processor.should_process():
            img = self.blur_processor.process(img, shape_after_scaling)
        return img


class Processor:
    def should_process(self):
        return self.low is not None or self.high is not None


class ScaleProcessor(Processor):
    def __init__(self, args):
        self.low = args.scale_low
        self.high = args.scale_high
        if not self.low:
            self.low = self.high
        if not self.high:
            self.high = self.low

    def process(self, img):
        scale = uniform(self.low, self.high)
        return cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))


class MarginProcessor(Processor):
    def __init__(self, args):
        self.low = args.margin_low
        self.high = args.margin_high
        if not self.low:
            self.low = self.high
        if not self.high:
            self.high = self.low
        self.equal = args.margins_equal

    def process(self, img):
        if(self.equal):
            margin = unform(self.low/2, self.high/2)
            a, b, c, d = int(margin*img.shape[0]), int(margin*img.shape[0]), int(
                margin*img.shape[1]), int(margin*img.shape[1])
            img = np.pad(img, ((max(0, a), max(0, b)),
                               (max(0, c), max(0, d)), (0, 0)))
            img = MarginProcessor.unpad(img, ((a, b), (c, d), (0, 0)))
        else:
            def rndm(): return uniform(self.low/2, self.high/2)
            a, b, c, d = int(rndm()*img.shape[0]), int(rndm()*img.shape[0]), int(
                rndm()*img.shape[1]), int(rndm()*img.shape[1])
            img = np.pad(img, ((max(0, a), max(0, b)),
                               (max(0, c), max(0, d)), (0, 0)))
            img = MarginProcessor.unpad(img, ((min(0, a), min(0, b)),
                                              (min(0, c), min(0, d)), (0, 0)))
        return img

    @staticmethod
    def unpad(x, pad_width):
        slices = []
        for c in pad_width:
            e = None if c[1] == 0 else c[1]
            slices.append(slice(-c[0], e))
        return x[tuple(slices)]


class BlurProcessor(Processor):
    def __init__(self, args):
        self.low = args.blur_low
        self.high = args.blur_high
        if not self.low:
            self.low = self.high
        if not self.high:
            self.high = self.low
        self.probability = args.blur_probability

    def process(self, img, wm_shape):
        if random() <= self.probability:
            blur_low, blur_high = int(self.low*min(
                wm_shape[0], wm_shape[1])), int(self.high*min(wm_shape[0], wm_shape[1]))
            blur = randint(blur_low, blur_high)
            img = cv2.blur(img, (blur, blur))
        return img


def get_random_solid_background(im):
    random_color = np.array(colorsys.hsv_to_rgb(
        random(), random(), random()))*255
    return np.full((im.shape[0], im.shape[1], 3), random_color.astype(np.int32))


def get_random_photo_background(im):
    with urlopen(f"https://picsum.photos/{im.shape[1]}/{im.shape[0]}") as url:
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
    return background.astype('uint8'), full_mask


if __name__ == '__main__':
    args = parser.parse_args()

    im = cv2.imread(args.path, cv2.IMREAD_UNCHANGED)
    generator = ImagePermutationGenerator(im, args)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.mask_out_dir, exist_ok=True)

    photo_bg_number = int(args.photos*args.number)
    solid_bg_number = args.number - photo_bg_number

    for i in range(args.number):
        cutout = generator.next()
        background = get_random_solid_background(
            cutout) if i < solid_bg_number else get_random_photo_background(cutout)
        combined_image, mask = overlay_transparent(background, cutout)

        cv2.imwrite(f'{args.out_dir}/{i}.jpg', combined_image)
        if args.mask:
            cv2.imwrite(f'{args.mask_out_dir}/{i}.png', mask)

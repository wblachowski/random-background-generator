import cv2
import numpy as np
import os
import argparse
import colorsys
from random import randint, uniform, random
from urllib.request import urlopen
from multiprocessing import cpu_count, Pool
from itertools import repeat
import asyncio
import aiohttp
import signal

parser = argparse.ArgumentParser(
    description='Generating random backgrounds for a given image.')
parser.add_argument('path', help='path to an image to process')
parser.add_argument('-o', '--out-dir', default='out',
                    help='output directory. Default: out')
parser.add_argument('-n', '--number', type=int, default=100,
                    help='number of images to generate. Default: 100')
parser.add_argument('-p', '--photos', type=float, default=0.,
                    help='percentage of images to have photographs as background (should be between 0.0 and 1.0). Default: 0.0')
parser.add_argument('-m', '--mask', action='store_true',
                    help='whether to save masks. Default: false')
parser.add_argument('-mo', '--mask-out-dir', default='mask',
                    help='mask output directory. Default: mask')
parser.add_argument('--scale-low', type=float,
                    help='lower limit for relative scaling (should be higher than 0.0)')
parser.add_argument('--scale-high', type=float,
                    help='upper limit for relative scaling (should be higher than 0.0)')
parser.add_argument('--margin-low', type=float,
                    help='lower limit for relative margins (should be higher than -0.5)')
parser.add_argument('--margin-high', type=float,
                    help='upper limit for relative margins (should be higher than -0.5)')
parser.add_argument('--margin-equal', action='store_true',
                    help='whether to keep margins equal. Default: false')
parser.add_argument('--blur-probability', type=float,
                    default=0., help='probability of blurring an image (should be between 0.0 and 1.0). Default: 0.0')
parser.add_argument('--blur-low', type=float, default=0.05,
                    help='lower limit for relative blurring strength (should be between 0.0 and 1.0). Default: 0.05')
parser.add_argument('--blur-high', type=float,
                    help='upper limit for relative blurring strength (should be between 0.0 and 1.0)')


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
    def __init__(self, low, high):
        if high is not None and low is None:
            low = high
            low, high = min(low, high), max(low, high)
        elif low is not None and high is None:
            high = low
            low, high = min(low, high), max(low, high)
        self.low = low
        self.high = high

    def should_process(self):
        return self.low is not None or self.high is not None


class ScaleProcessor(Processor):
    def __init__(self, args):
        super().__init__(args.scale_low, args.scale_high)

    def process(self, img):
        scale = uniform(self.low, self.high)
        return cv2.resize(img, (int(max(1, img.shape[1]*scale)), max(1, int(img.shape[0]*scale))))


class MarginProcessor(Processor):
    def __init__(self, args):
        super().__init__(args.margin_low, args.margin_high)
        self.equal = args.margin_equal

    def process(self, img):
        def rndm(): return max(-0.49, uniform(self.low, self.high))
        margins = [rndm()]*4 if self.equal else [rndm() for _ in range(4)]
        a, b, c, d = [int(x*img.shape[:2][i//2])for i, x in enumerate(margins)]
        img = np.pad(img, ((max(0, a), max(0, b)),
                           (max(0, c), max(0, d)), (0, 0)))
        img = MarginProcessor.unpad(
            img, ((min(0, a), min(0, b)), (min(0, c), min(0, d)), (0, 0)))
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
        super().__init__(args.blur_low, args.blur_high)
        self.probability = args.blur_probability

    def process(self, img, wm_shape):
        if random() <= self.probability:
            blur_low, blur_high = int(self.low*min(
                wm_shape[0], wm_shape[1])), int(self.high*min(wm_shape[0], wm_shape[1]))
            blur = max(1, randint(blur_low, blur_high))
            img = cv2.blur(img, (blur, blur))
        return img


def get_random_solid_background(shape):
    random_color = np.array(colorsys.hsv_to_rgb(
        random(), random(), random()))*255
    return np.full((shape[0], shape[1], 3), random_color.astype(np.int32))


async def get_random_photo_background(shape):
    url = f'https://picsum.photos/{shape[1]}/{shape[0]}'
    async with aiohttp.ClientSession() as session, session.get(url=url) as response:
        resp = await response.read()
        image_array = np.asarray(bytearray(resp), dtype='uint8')
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image


def combine(background, overlay):
    mask = overlay[..., 3:] / 255.0
    overlay = overlay[..., :3]
    result = (1.0 - mask) * background + mask * overlay
    return result.astype('uint8'), mask*255


def create_output_dirs(args):
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mask:
        os.makedirs(args.mask_out_dir, exist_ok=True)


async def generate_imgs(ids, solid_bg_number, generator, args):
    overlays = [generator.next() for _ in ids]
    backgrounds_solid = [get_random_solid_background(
        overlays[i].shape) for i, id in enumerate(ids) if id < solid_bg_number]
    backgrounds_photo = await asyncio.gather(*[get_random_photo_background(
        overlays[i].shape) for i, id in enumerate(ids) if id >= solid_bg_number])
    for i, background, overlay in zip(ids, backgrounds_solid+backgrounds_photo, overlays):
        combined_image, mask = combine(background, overlay)
        cv2.imwrite(f'{args.out_dir}/{i}.jpg', combined_image)
        if args.mask:
            cv2.imwrite(f'{args.mask_out_dir}/{i}.png', mask)


def run(ids, solid_bg_number, generator, args):
    asyncio.run(generate_imgs(ids, solid_bg_number, generator, args))


def run_singlethread(solid_bg_number, generator, args):
    run(range(args.number), solid_bg_number, generator, args)


def run_multithread(solid_bg_number, generator, args):
    with Pool() as pool:
        parts = np.array_split(range(args.number), cpu_count())
        data = zip(parts, repeat(solid_bg_number),
                   repeat(generator), repeat(args))
        pool.starmap(run, data)


if __name__ == '__main__':
    args = parser.parse_args()
    create_output_dirs(args)
    im = cv2.imread(args.path, cv2.IMREAD_UNCHANGED)
    generator = ImagePermutationGenerator(im, args)

    solid_bg_number = args.number - int(args.photos*args.number)

    if args.number < 100:
        run_singlethread(solid_bg_number, generator, args)
    else:
        run_multithread(solid_bg_number, generator, args)

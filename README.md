# Random Background Generator 🖼
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![grid](/assets/grid.png)

A python script adding random backgrounds to a PNG image. Two types of backgrounds are supported - solid color and photographs. A set of transformations can be applied to the input image prior to it being superimposed over the final background. Together with the final images, the script can produce corresponding mask images.

The script was created to produce datasets for the background removal task. Data produced can be directly used, for instance, to train a U2-Net neural network.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
$ py rbgg.py [-h] [-o OUT_DIR] [-n NUMBER] [-p PHOTOS] [-m] [-mo MASK_OUT_DIR] [-sl SCALE_LOW] [-sh SCALE_HIGH] [-ml MARGIN_LOW] [-mh MARGIN_HIGH] [-me] [-bp BLUR_PROBABILITY] [-bl BLUR_LOW] [-bh BLUR_HIGH] [--single-core] path
```

Arguments:

```
positional arguments:
  path                  path to the image to process

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_DIR, --out-dir OUT_DIR
                        output directory. Default: out
  -n NUMBER, --number NUMBER
                        number of images to generate. Default: 100
  -p PHOTOS, --photos PHOTOS
                        percentage of images to have photographs as background (should be between 0.0 and 1.0). Default: 0.0
  -m, --mask            whether to save masks. Default: false
  -mo MASK_OUT_DIR, --mask-out-dir MASK_OUT_DIR
                        mask output directory. Default: mask
  -sl SCALE_LOW, --scale-low SCALE_LOW
                        lower limit for relative scaling (should be higher than 0.0)
  -sh SCALE_HIGH, --scale-high SCALE_HIGH
                        upper limit for relative scaling (should be higher than 0.0)
  -ml MARGIN_LOW, --margin-low MARGIN_LOW
                        lower limit for relative margins (should be higher than -0.5)
  -mh MARGIN_HIGH, --margin-high MARGIN_HIGH
                        upper limit for relative margins (should be higher than -0.5)
  -me, --margin-equal   whether to keep margins equal. Default: false
  -bp BLUR_PROBABILITY, --blur-probability BLUR_PROBABILITY
                        probability of blurring an image (should be between 0.0 and 1.0). Default: 0.0
  -bl BLUR_LOW, --blur-low BLUR_LOW
                        lower limit for relative blurring strength (should be between 0.0 and 1.0). Default: 0.05
  -bh BLUR_HIGH, --blur-high BLUR_HIGH
                        upper limit for relative blurring strength (should be between 0.0 and 1.0)
  --single-core         Run on a single CPU core. Default: false
```

## Background types

There are two types of backgrounds that the script can produce - solid colors and photographs. By default, all backgrounds will be a solid colour. The percentage of images to have photographs as their background can be controlled with the `-p` parameter. Setting the parameter to 1 will result in all images having photograph backgrounds, 0.5 will mean 50% etc.

#### Solid

Each background is randomly drawn from a uniform distribution of the color space.

![Scale](/assets/solid.png)

#### Photograph

The photos are downloaded from https://picsum.photos.

![Photo](/assets/photo.png)

## Mask

Use `--mask` to save masks along with the final images. You can specify the target directory for masks using `--mask-out-dir` (the default directory is `mask`).

For instance, for the following output images:
![Final imgs](/assets/mask_org.png)

The masks are:
![Final imgs with masks](/assets/mask_mask.png)

## Transformations

Scaling, blurring, or adding margins can be applied to the input image before combining it with the background. The options allow for either constant or random transformations, in which case the user can specify the range of transformation coefficients.

#### Scale

Use `--scale-low` and `--scale-high` to specify scaling of the input image. The scaling factor will be drawn from a uniform distribution between the two values specified. If only a single value is specified (either `low` or `high`), the scaling factor is constant. The scaling parameter should be greater than zero.

![Scaling](/assets/scale.png)

#### Margin

Use `--margin-low` and `--margin-high` to specify the range of margins. The margin for each side will be drawn randomly from a uniform distribtion between the two values. If `--margin-equal` flag is specified, the margin the same for all four sides. If only a single value is specified (either `low` or `high`), the margin is constant.

The values passed are relative to the base image after scaling (if scaling is applied). For example, if you set the margin to be 1.0, the left and right margins will be equal to the width of the base image, and the top and bottom margins equal to the height of the image.

You can specify negative values as well to crop out a part of the input image. Note that the values should be greater than -0.5, otherwise the negative margins would eat up the entirety of the base image.

![Margin](/assets/margin.png)

#### Blur

Use `--blur-probability` to set a probability of blurring the final image (the default is zero). Use `--blur-low` and `--blur-high` to specify the range of blurring strength. If only a single value is specified (either `low` or `high`), the blurring strength is constant.

The blurring strength can take values between 0.0 and 1.0, with 0.0 meaning no blur at all, and 1.0 meaning the maximal blur possible.

![Blur](/assets/blur.png)

## Multithreading

Use `--single-core` to run on a single CPU core. Otherwise, the tool uses multithreading if the number of images to generate is larger than 100 (for smaller number of images the multithreading overhead is greater than the benefits - although the exact number is machine-specific). The number of threads is set to the number of CPU cores available.

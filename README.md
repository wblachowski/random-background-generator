# Random Background Generator

![grid](/assets/grid.png)

A python script adding random backgrounds to a given image. Two types of backgrounds are supported - solid color and photographs. A set of transformations can be applied to the input image prior to it being superimposed over the final background. Together with the final images, the script can produce corresponding mask images.

The script was created to produce datasets for the background removal task. Data produced can be directly used, for instance, to train a U2-Net neural network.

## Usage

```bash
$ py rbgg.py [-h] [-o OUT_DIR] [-n NUMBER] [-p PHOTOS] [-m] [-mo MASK_OUT_DIR] [--scale-low SCALE_LOW] [--scale-high SCALE_HIGH] [--margin-low MARGIN_LOW] [--margin-high MARGIN_HIGH] [--margin-equal] [--blur-probability BLUR_PROBABILITY] [--blur-low BLUR_LOW] [--blur-high BLUR_HIGH] path
```

Arguments:

```
positional arguments:
  path                  path to an image to process

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_DIR, --out-dir OUT_DIR
                        output directory
  -n NUMBER, --number NUMBER
                        number of images to generate
  -p PHOTOS, --photos PHOTOS
                        percentage of images to have photographs as background
  -m, --mask            whether to save masks
  -mo MASK_OUT_DIR, --mask-out-dir MASK_OUT_DIR
                        mask output directory
  --scale-low SCALE_LOW
                        lower limit for scaling
  --scale-high SCALE_HIGH
                        upper limit for scaling
  --margin-low MARGIN_LOW
                        lower limit for margins
  --margin-high MARGIN_HIGH
                        upper limit for margins
  --margin-equal        whether to keep margins equal
  --blur-probability BLUR_PROBABILITY
                        blur probability
  --blur-low BLUR_LOW   lower limit for blurring strength
  --blur-high BLUR_HIGH
                        upper limit for blurring strength
```

## Background types

There are two types of backgrounds that the script can produce - solid colors and photographs. By default, all backgrounds will be solid colour. The percentage of images to have photographs as their background can be controlled with the `-p` parameter. Setting the parameter to 1 will result in all images having photograph backgrounds, 0.5 will mean 50% etc.

#### Solid

Each background is randomly drawn from a uniform distribution of the color space.

![Scale](/assets/solid.png)

#### Photograph

The photos are downloaded from https://picsum.photos.

![Photo](/assets/photo.png)

## Mask

Use `--mask` to save masks along the final images. You can specify the target directory for masks using `--mask-out-dir`.

For instance, for the following output images:
![Final imgs](/assets/mask_org.png)

The masks are:
![Final imgs with masks](/assets/mask_mask.png)

## Transformations

Scaling, blurring, or adding margins can be applied to the input image before combining it with the background. The options allow for either constant or random transformations, in which case the user can specify the range of transformation coefficients.

#### Scale

Use `--scale-low` and `--scale-high` to specify scaling of the input image. The scaling factor will be drawn from a uniform distribution between the two values specified. If only a single value is specified (either `low` or `high`), the scaling factor is constant.

![Scaling](/assets/scale.png)

#### Margin

Use `--margin-low` and `--margin-high` to specify the range of margins. Margin for each side will be drawn randomly from a uniform distrubtion between the two values. If `--margin-equal` flag is specified, the margin the same for all four sides. If only a single value is specified (either `low` or `high`), the margin is constant.

You can specify negative values as well to crop out a part of the input image.

![Margin](/assets/margin.png)

#### Blur

Use `--blur-probability` to set a probability of blurring the final image (the default is zero). Use `--blur-low` and `--blur-high` to specify the range of blurring strength. If only a single value is specified (either `low` or `high`), the blurring strength is constant.

![Blur](/assets/blur.png)

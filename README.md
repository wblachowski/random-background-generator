# Random Background Generator

![grid](/assets/grid.png)


## Mask
Use `--mask` to save masks along the final images. You can specify the target directory for masks using `--mask-out-dir`.

For instance, for the following output images:
![Final imgs](/assets/mask_org.png)

The masks are:
![Final imgs with masks](/assets/mask_mask.png)

## Scale
Use `--scale-low` and `--scale-high` to specify scaling of the input image. The scaling factor will be drawn from a uniform distribution between the two values specified. If only a single value is specified (either `low` or `high`), the scaling factor is contant.

![Scaling](/assets/scale.png)

## Margin
Use `--margin-low` and `--margin-high` to specify the range of margins. Margin for each side will be drawn randomly from a uniform distrubtion between the two values. If `--margin-equal` flag is specified, the margin the same for all four sides. If only a single value is specified (either `low` or `high`), the margin is contant.

You can specify negative values as well to crop out a part of the input image.

![Margin](/assets/margin.png)

## Blur
Use `--blur-probability` to set a probability of blurring the final image (the default is zero). Use `--blur-low` and `--blur-high` to specify the range of blurring strength. If only a single value is specified (either `low` or `high`), the blurring strength is contant.

![Blur](/assets/blur.png)

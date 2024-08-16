import argparse
from synthetic_images.anisotropic_transforms import anisotropic_resize_reconstruct, smooth_horizontal, smooth_vertical
from imageio.v3 import imread, imwrite
from glob import glob
import numpy as np
from functools import partial
import os
from tqdm import tqdm


def transform_all(input_files, output_dir, transform):
    for input_file in input_files:
        # Read image, should be in range [0, 255]
        img = imread(input_file)

        # Convert to float in range [0, 255] to avoid intermediate rounding errors
        img = img.astype(float)

        # Transform image, should be in range [0, 255]
        transformed_img = transform(img)

        # Cast to uint8
        transformed_img = np.clip(np.round(transformed_img), 0, 255).astype(np.uint8)

        # Construct output filepath
        output_filepath = os.path.join(output_dir, os.path.basename(input_file))

        # Store image
        imwrite(output_filepath, transformed_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str, help="Directory containing the input images to be transformed")
    parser.add_argument("--output_dir", type=str, help="Directory where to store the results")
    args = vars(parser.parse_args())

    input_files = sorted(glob(os.path.join(args["input_dir"], "*.png")))

    # Anisotropic resize in vertical direction
    for scale_y in tqdm([0.5, 0.25, 0.125], desc="Anisotropic resize in vertical direction"):
        # Create output directory
        transform_output_dir = os.path.join(args["output_dir"], f"anisotropic_resize_vertical_scale_y_{scale_y}")
        os.makedirs(transform_output_dir, exist_ok=False)

        # Set up transform
        transform = partial(anisotropic_resize_reconstruct, scale_y=scale_y, scale_x=1)

        # Transform all images
        transform_all(input_files=input_files, output_dir=transform_output_dir, transform=transform)

    # Anisotropic resize in horizontal direction
    for scale_x in tqdm([0.5, 0.25, 0.125], desc="Anisotropic resize in horizontal direction"):
        # Create output directory
        transform_output_dir = os.path.join(args["output_dir"], f"anisotropic_resize_horizontal_scale_x_{scale_x}")
        os.makedirs(transform_output_dir, exist_ok=False)

        # Set up transform
        transform = partial(anisotropic_resize_reconstruct, scale_y=1, scale_x=scale_x)

        # Transform all images
        transform_all(input_files=input_files, output_dir=transform_output_dir, transform=transform)

    # Horizontal smoothing
    for kernel_size in tqdm([5, 10, 20], desc="Horizontal smoothing"):
        # Create output directory
        transform_output_dir = os.path.join(args["output_dir"], f"horizontal_smoothing_kernel_{kernel_size}")
        os.makedirs(transform_output_dir, exist_ok=False)

        # Set up transform
        transform = partial(smooth_horizontal, kernel_size=kernel_size)

        # Transform all images
        transform_all(input_files=input_files, output_dir=transform_output_dir, transform=transform)

    # Vertical smoothing
    for kernel_size in tqdm([5, 10, 20], desc="Vertical smoothing"):
        # Create output directory
        transform_output_dir = os.path.join(args["output_dir"], f"vertical_smoothing_kernel_{kernel_size}")
        os.makedirs(transform_output_dir, exist_ok=False)

        # Set up transform
        transform = partial(smooth_vertical, kernel_size=kernel_size)

        # Transform all images
        transform_all(input_files=input_files, output_dir=transform_output_dir, transform=transform)

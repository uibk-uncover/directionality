import numpy as np
import argparse
from tqdm import tqdm
from imageio.v3 import imwrite
import os


def create_symmetric_cosine_image(side_length, num_cosines=2, seed=6020):
    """
    Create a symmetric image showing overlapping cosine patterns.
    :param side_length: side length of images to generate
    :param num_cosines: number of cosine patterns
    :param seed: seed for random number generator
    :return: grayscale image of shape [side_length, side_length] with intensities in range [0, 1]
    """
    rng = np.random.default_rng(seed)

    img = 0.5 * np.ones((side_length, side_length))

    # xs ranges between [0, 1]
    xs = np.arange(side_length) / (side_length - 1)

    # Generate cosine curves
    for _ in range(num_cosines):
        amplitude = rng.normal(loc=0, scale=0.1)
        wavelength = 2 ** rng.integers(low=0, high=4)

        img += amplitude * np.cos(wavelength * xs[:, None] * 2 * np.pi)
        img += amplitude * np.cos(wavelength * xs[None, :] * 2 * np.pi)

    # Clip pixel values to range [0, 1]
    img = np.clip(img, 0, 1)

    return img


def create_symmetric_random_image(side_length, seed=6020):
    """
    Create a symmetric noise image
    :param side_length: side length of image to generate
    :param seed: seed for random number generator
    :return: grayscale image of shape [side_length, side_length] with intensities in range [0, 1]
    """
    rng = np.random.default_rng(seed)

    # Create a random synthetic image with desired dimensions
    img = rng.normal(loc=0.5, scale=0.2, size=(side_length, side_length))

    # Ensure horizontal symmetry
    img = (img + np.fliplr(img)) / 2

    # Ensure vertical symmetry
    img = (img + np.flipud(img)) / 2

    # Ensure both horizontal and vertical symmetry
    img = (img + np.fliplr(np.flipud(img))) / 2

    # Ensure rotation/transposition symmetry
    img = (img + np.rot90(img)) / 2

    # Assert transverse symmetry
    assert np.allclose(img, np.transpose(img))
    assert np.allclose(img, np.rot90(np.transpose(img), k=2))

    # Clip intensities to range [0, 1]
    img = np.clip(img, 0, 1)

    return img


def create_symmetric_image(image_size, rng):
    """
    Create a symmetric image that contains a cosine pattern with additive noise.
    :param image_size: side length of the image to generate
    :param rng: instance of a random number generator
    :return: ndarray of shape [image_size, image_size] with intensities in range [0, 1]
    """
    num_cosines = rng.integers(low=1, high=8)
    seed = rng.integers(low=0, high=2 ** 32)

    cosine_img = create_symmetric_cosine_image(
        side_length=image_size,
        num_cosines=num_cosines,
        seed=seed,
    )

    noise_img = create_symmetric_random_image(
        side_length=image_size,
        seed=seed,
    )

    # Create a random noise amplitude between [0, 0.2)
    noise_amplitude = rng.random() * 0.2

    # Add cosine and noise images
    img = (1 * cosine_img + noise_amplitude * noise_img) / (1. + noise_amplitude)

    # Convert to uint8
    img = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)

    # Assert symmetry
    assert np.allclose(img, np.fliplr(img))
    assert np.allclose(img, np.flipud(img))
    assert np.allclose(img, np.fliplr(np.flipud(img)))
    assert np.allclose(img, np.rot90(img, k=1))
    assert np.allclose(img, np.rot90(img, k=3))

    return img


def create_symmetric_images(output_dir, num_images, image_size, seed):
    """
    Create a given number of synthetic symmetric images
    :param output_dir: output directory
    :param num_images: number of images to generate
    :param image_size: side length of the images to generate
    :param seed: seed for random number generator
    """
    rng = np.random.default_rng(seed)

    for i in tqdm(range(num_images)):
        img = create_symmetric_image(image_size, rng)

        pattern = "{:0" + str(int(np.ceil(np.log10(num_images)))) + "d}.png"

        output_filename = pattern.format(i)
        output_filepath = os.path.join(output_dir, output_filename)

        imwrite(output_filepath, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Directory where to store synthetic images", default="/home/bene/data/directionality-matters/synthetic-symmetric-images/original")
    parser.add_argument("--num_images", type=int, help="Number of images", default=1000)
    parser.add_argument("--image_size", type=int, help="Image side length", default=512)
    parser.add_argument("--seed", type=int, help="Random seed", default=6020)

    args = vars(parser.parse_args())

    create_symmetric_images(
        output_dir=args["output_dir"],
        num_images=args["num_images"],
        image_size=args["image_size"],
        seed=args["seed"],
    )

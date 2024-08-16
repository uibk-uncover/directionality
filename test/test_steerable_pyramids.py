import unittest
import itertools
from copy import deepcopy
import numpy as np
import pyrtools as pt
from parameterized import parameterized
from imageio.v3 import imread
from directionality.steerable_pyramids_utils import calculate_band_energies_original, calculate_band_energies_fast
import os


ASSETS_DIR = "assets/uncompressed_gray"


class TestSteerablePyramids(unittest.TestCase):
    @parameterized.expand([
        os.path.join(ASSETS_DIR, "seal1.png"),
        os.path.join(ASSETS_DIR, "seal2.png"),
        os.path.join(ASSETS_DIR, "seal3.png"),
    ])
    def test_reconstruction(self, img_filepath, num_scales=2, num_orientations=16):
        """
        Verify that pyr.recon_pyr(levels=l, bands=o) is identical to zeroing all undesired bands and levels
        :param img_filepath:
        :param num_scales:
        :param num_orientations:
        :return:
        """
        img = imread(img_filepath)
        pyr = pt.pyramids.SteerablePyramidFreq(img, height=num_scales, order=num_orientations - 1)
        pyr_original = deepcopy(pyr)

        # Disable the low-pass residual
        pyr.pyr_coeffs['residual_lowpass'][:, :] = 0

        # Disable the high-pass residual
        pyr.pyr_coeffs['residual_highpass'][:, :] = 0

        for scale_idx_keep in range(pyr.num_scales):
            for orientation_idx_keep in np.arange(0, pyr.num_orientations, step=pyr.num_orientations // 2):
                pyr_local = deepcopy(pyr)

                for s in range(pyr.num_scales):
                    for o in range(pyr.num_orientations):
                        # Keep only selected subband
                        if o == orientation_idx_keep and s == scale_idx_keep:
                            continue

                        # Otherwise zero out subband
                        pyr_local.pyr_coeffs[(s, o)][:, :] = 0

                reconstruction_local = pyr_local.recon_pyr()
                reconstruction_original = pyr_original.recon_pyr(levels=scale_idx_keep, bands=orientation_idx_keep)

                np.testing.assert_allclose(reconstruction_original, reconstruction_local)

    @parameterized.expand([
        (os.path.join(ASSETS_DIR, "seal1.png")),
        (os.path.join(ASSETS_DIR, "seal2.png")),
        (os.path.join(ASSETS_DIR, "seal3.png")),
    ])
    def test_height(self, img_filepath, num_orientations=16):
        img = imread(img_filepath)
        pyr_height_1 = pt.pyramids.SteerablePyramidFreq(img, height=1, order=num_orientations - 1)
        pyr_height_2 = pt.pyramids.SteerablePyramidFreq(img, height=2, order=num_orientations - 1)
        pyr_height_3 = pt.pyramids.SteerablePyramidFreq(img, height=3, order=num_orientations - 1)

        for orientation in range(num_orientations):
            # Compare scale 0
            np.testing.assert_allclose(pyr_height_1.pyr_coeffs[(0, orientation)], pyr_height_2.pyr_coeffs[(0, orientation)])
            np.testing.assert_allclose(pyr_height_1.pyr_coeffs[(0, orientation)], pyr_height_3.pyr_coeffs[(0, orientation)])

            # Compare scale 1
            np.testing.assert_allclose(pyr_height_2.pyr_coeffs[(1, orientation)], pyr_height_3.pyr_coeffs[(1, orientation)])

    @parameterized.expand(itertools.product(
        [
            os.path.join(ASSETS_DIR, "seal1.png"),
            os.path.join(ASSETS_DIR, "seal2.png"),
            os.path.join(ASSETS_DIR, "seal3.png"),
        ],  # filepath
        range(1, 5),  # height
        [4, 8, 16],  # num_orientations
    ))
    def test_simplified_energy_calculation(self, img_filepath, height, num_orientations):
        """
        Verify that our calculation of the band energies based on a partial decomposition implementation matches the band energies calculated from a full decomposition.
        :param img_filepath: path to grayscale image
        :param height: number of scales
        :param num_orientations: number of orientations
        """
        img = imread(img_filepath)

        band_energies_original = calculate_band_energies_original(img, height=height, order=num_orientations - 1, num_orientations_of_interest=2)
        band_energies_fast = calculate_band_energies_fast(img, height=height, order=num_orientations - 1, num_orientations_of_interest=2)

        np.testing.assert_allclose(band_energies_original, band_energies_fast)


__all__ = ["TestSteerablePyramids"]

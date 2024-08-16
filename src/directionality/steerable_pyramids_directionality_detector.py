import numpy as np
import pyrtools as pt
from directionality.directionality_detector import DirectionalityDetector
from directionality.steerable_pyramids_utils import calculate_band_energies_fast


class SteerablePyramidsDirectionalityDetector(DirectionalityDetector):
    def __init__(self, height=2, num_orientations=16):
        """
        :param height: number of scales
        :param num_orientations: number of orientations must be a multiple of 2
        """

        assert num_orientations % 2 == 0, "Number of orientations must be an integer multiple of 2"

        self.height = height
        self.num_orientations = num_orientations

    def evaluate_directional_statistics(self, img):
        """
        Evaluate directional statistics for each scale separately
        :param img: 2D image
        :return ndarray of shape [height, 2]
            The two columns represent energies in directions
            (0) west + east
            (1) north + south
        """
        return calculate_band_energies_fast(img, height=self.height, order=self.num_orientations - 1, num_orientations_of_interest=2)

    def evaluate_directionality_score(self, img):
        """
        Quantify directionality of a given image
        :param img: 2D image
        :return: scalar value d, where
            d > 0 means that the image is dominated by vertical edges (= horizontal frequency coefficients),
            d < 0 means that the images is dominated by horizontal edges (= vertical frequency coefficients).
        """
        directional_stats = self.evaluate_directional_statistics(img)
        assert directional_stats.shape == (self.height, 2)

        # Per scale: (west + east) - (north + south)
        directionality_score_per_scale = (directional_stats[:, 0] - directional_stats[:, 1]) / (directional_stats[:, 0] + directional_stats[:, 1])

        return np.sum(directionality_score_per_scale)

    def reconstruct_individual_components(self, img):
        """
        Reconstruct images with individual subbands only and transform into Fourier space
        :param img: 2D ndarray
        :return: 4D ndarray of shape [num_scales, num_orientations, height, width]
            The order of the orientations is
                (0) west + east
                (1) north + south
        """
        pyr = pt.pyramids.SteerablePyramidFreq(img, height=self.height, order=self.num_orientations - 1)

        img_height, img_width = img.shape

        # We only use two orientations
        individual_components = np.zeros((self.height, 2, img_height, img_width), dtype=float)

        for scale in range(pyr.num_scales):
            for orientation_idx_keep in np.arange(0, pyr.num_orientations, step=pyr.num_orientations // 2):
                output_orientation_idx = orientation_idx_keep * 2 // pyr.num_orientations

                band_reconstruction = pyr.recon_pyr(levels=scale, bands=orientation_idx_keep)

                # Fourier transform
                band_reconstruction_fft = np.abs(np.fft.fftshift(np.fft.fft2(band_reconstruction)))

                individual_components[scale, output_orientation_idx] = band_reconstruction_fft

        return individual_components

    def show_individual_components(self, img, figsize=None, fontsize=None):
        """
        Create a figure that visualizes the individual components
        :param img: grayscale image
        :return: fig
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        from mpl_toolkits.axes_grid1 import ImageGrid

        # Add small constant to the image's Fourier transform to avoid log(0)
        eps = 1e-9

        individual_components = self.reconstruct_individual_components(img)
        directional_stats = self.evaluate_directional_statistics(img)

        # Create figure and axis grid (= image grid)
        if figsize is None:
            figsize = (8, 4 * self.height)

        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(
            fig,
            rect=111,
            nrows_ncols=(self.height, 2),
            axes_pad=0.3, # Padding between axes in inch
            share_all=True,
            cbar_location="right",
            cbar_mode="edge",
            cbar_size="7%",
            cbar_pad="5%",
        )

        # Create one row per scale
        for scale in range(self.height):
            # Each row shares the colorbar
            vmin = individual_components[scale, :].min() + eps
            vmax = individual_components[scale, :].max() + eps

            for orientation in [0, 1]:
                # Grid index is linear
                idx = scale * 2 + orientation
                ax = grid[idx]

                # Display the Fourier transform in log space
                im = ax.imshow(
                    individual_components[scale, orientation] + eps,
                    cmap="gray",
                    norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                )

                # Add colorbar to the right
                if orientation == 1:
                    grid.cbar_axes[scale].colorbar(im)

                # Display relative energy in the axis title
                subband_energy = directional_stats[scale, orientation] / np.sum(directional_stats[scale, :])
                ax.set_title(f"Energy: {subband_energy:4.3f}", fontsize=fontsize)

        return fig, grid

    @classmethod
    def name(cls):
        """
        :return: Return the detector name
        """
        return "SteerablePyramidsDirectionalityDetector"

    def params(self):
        return {
            "height": self.height,
            "num_orientations": self.num_orientations,
        }

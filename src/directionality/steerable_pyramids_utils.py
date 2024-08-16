"""
This code is carved out from pyrtools.
We removed parts that are not needed for our purposes to speed up the computation.
Original code: https://github.com/LabForComputationalVision/pyrtools/blob/main/src/pyrtools/pyramids/SteerablePyramidFreq.py
"""
import warnings
import numpy as np
from scipy.special import factorial
import pyrtools as pt
from pyrtools.pyramids.c.wrapper import pointOp
from pyrtools.tools.utils import rcosFn


def calculate_band_energies_fast(image, height, order, num_orientations_of_interest=2, twidth=1):
    """
    Calculate the energy of each frequency subband.

    This code is carved out from pyrtools.
    This modified implementation skips parts of the decomposition that are not needed for our purposes.

    :param image: square grayscale image
    :param height: number of scales
    :param order: Gaussian derivative order. This is the number of orientations - 1.
    :param num_orientations_of_interest: number of equally-spaced orientation bands of interest. Must be a divisor of the number orientations. The default is 2.
    :param twidth: width of the transition region of the radial lowpass function, in octaves
    :return: energies of shape [num_scales, 2]. The two columns contain the horizontal and vertical frequency subbands at each scale.
    """

    #
    # --------------------
    # Sanitize input arguments
    # --------------------
    #
    assert len(image.shape) == 2, "Expected grayscale image"

    img_height, img_width = image.shape
    assert img_height == img_width, "Expected square input image"

    max_ht = np.floor(np.log2(min(image.shape))) - 2
    if height == "auto" or height is None:
        num_scales = int(max_ht)
    elif height > max_ht:
        raise Exception("Cannot build pyramid higher than %d levels." % (max_ht))
    else:
        num_scales = int(height)

    # Parse order and convert to number of orientations
    if order < 0:
        raise Exception("order must be a positive integer")

    num_orientations = int(order + 1)
    assert num_orientations % num_orientations_of_interest == 0

    if twidth <= 0:
        warnings.warn("twidth must be positive. Setting to 1.")
        twidth = 1
    twidth = int(twidth)

    #
    # --------------------
    # Set up buffer for intermediate results
    # --------------------
    #
    # Store frequency bands into a dict
    pyr_bands = dict()
    # Energies
    band_energies = np.zeros((height, num_orientations_of_interest), dtype=float)

    # Low-pass, high-pass and angular masks
    all_anglemasks = []
    himasks = []
    lomasks = []

    # Keep track of bounds and dimensions at individual scales
    bound_list = []
    dim_list = []
    bound_list.append((0, 0, image.shape[0], image.shape[1]))
    dim_list.append(image.shape)

    #
    # --------------------
    # Initial low- and high-pass filters
    # --------------------
    #

    dims = np.array(image.shape)
    # Center position [257, 257]
    ctr = np.ceil((np.array(dims) + 0.5) / 2).astype(int)

    # xramp is a 2D array with values in range [-1, +1] increasing from left to right
    # yramp is a 2D array with values in range [-1, +1] increasing from top to bottom
    (xramp, yramp) = np.meshgrid(np.linspace(-1, 1, dims[1] + 1)[:-1], np.linspace(-1, 1, dims[0] + 1)[:-1])

    # Radial angle, starting with -pi in the top left quadrant and then radially moving towards +pi
    angle = np.arctan2(yramp, xramp)

    # Circular ramp, with 0 in the center and sqrt(2) at the border
    log_rad = np.sqrt(xramp ** 2 + yramp ** 2)
    # To avoid a zero value in the center, copy the left neighbor
    log_rad[ctr[0] - 1, ctr[1] - 1] = log_rad[ctr[0] - 1, ctr[1] - 2]
    # Take the log
    log_rad = np.log2(log_rad)

    # Radial transition function (a raised cosine in log-frequency):
    (Xrcos, Yrcos) = rcosFn(twidth, (-twidth / 2.0), np.array([0, 1]))
    Yrcos = np.sqrt(Yrcos)

    # Create low-pass filter mask
    YIrcos = np.sqrt(1.0 - Yrcos ** 2)
    lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1] - Xrcos[0])

    # Transform image into Fourier domain
    imdft = np.fft.fftshift(np.fft.fft2(image))

    # Apply low-pass mask to Fourier spectrum
    lo0mask = lo0mask.reshape(imdft.shape[0], imdft.shape[1])
    lodft = imdft * lo0mask
    lomasks.append(lo0mask)

    #
    # --------------------
    # Recursive decomposition
    # --------------------
    #
    for scale_idx in range(num_scales):
        # Keep on working on lodft
        Xrcos -= np.log2(2)

        # Create lookup table from pi * [-2 - eps, 1 + eps] where eps = 1 / lutsize
        lutsize = 1024
        Xcosn = np.pi * np.arange(-(2 * lutsize + 1), (lutsize + 2)) / lutsize

        const = (2 ** (2 * order)) * (factorial(order, exact=True) ** 2) / float(num_orientations * factorial(2 * order, exact=True))
        Ycosn = np.sqrt(const) * (np.cos(Xcosn)) ** order

        # Create next high-pass filter mask: 0 in the center, 1 at the margins, blurry transition. The zero disk in the center becomes smaller with every scale.
        log_rad_test = np.reshape(log_rad, (1, log_rad.shape[0] * log_rad.shape[1]))
        himask = pointOp(log_rad_test, Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0])
        himask = himask.reshape((lodft.shape[0], lodft.shape[1]))
        himasks.append(himask)

        anglemasks = []
        orientations_of_interest = np.arange(num_orientations_of_interest) / num_orientations_of_interest

        for band_idx, b in enumerate(orientations_of_interest):
            angle_tmp = np.reshape(angle, (1, angle.shape[0] * angle.shape[1]))
            anglemask = pointOp(angle_tmp, Ycosn, Xcosn[0] + np.pi * b, Xcosn[1] - Xcosn[0])

            anglemask = anglemask.reshape(lodft.shape[0], lodft.shape[1])
            # that (-1j)**order term in the beginning will be 1, -j, -1, j for order 0, 1, 2,
            # 3, and will then loop again

            # Combine high-pass and angular mask, multiply with low-pass DCT spectrum
            banddft = (-1j) ** order * lodft * anglemask * himask

            pyr_bands[(scale_idx, band_idx)] = banddft

            # Preserve angular mask
            anglemasks.append(anglemask)

        # Now store all angular masks
        all_anglemasks.append(anglemasks)

        # Create next low-pass mask: The next almost remains the same, but the resolution is halved.
        dims = np.array(lodft.shape)
        ctr = np.ceil((dims + 0.5) / 2).astype(int)
        lodims = np.ceil((dims - 0.5) / 2).astype(int)
        loctr = np.ceil((lodims + 0.5) / 2).astype(int)
        lostart = ctr - loctr
        loend = lostart + lodims

        # Keep track of bounds and dimensions
        bounds = (lostart[0], lostart[1], loend[0], loend[1])
        bound_list.append(bounds)
        dim_list.append(lodft.shape)

        log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]

        # Subsampling
        lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.abs(np.sqrt(1.0 - Yrcos ** 2))
        log_rad_tmp = np.reshape(log_rad, (1, log_rad.shape[0] * log_rad.shape[1]))
        lomask = pointOp(log_rad_tmp, YIrcos, Xrcos[0], Xrcos[1] - Xrcos[0])
        lomask = lomask.reshape(lodft.shape[0], lodft.shape[1])
        lomasks.append(lomask)

        lodft = lodft * lomask

    # Store the smallest bounds as well
    dims = np.array(lodft.shape)
    ctr = np.ceil((dims + 0.5) / 2).astype(int)
    lodims = np.ceil((dims - 0.5) / 2).astype(int)
    loctr = np.ceil((lodims + 0.5) / 2).astype(int)
    lostart = ctr - loctr
    loend = lostart + lodims
    bounds = (lostart[0], lostart[1], loend[0], loend[1])
    bound_list.append(bounds)
    dim_list.append(lodft.shape)

    # Make sure we have stored everyting

    assert len(himasks) == height
    assert len(all_anglemasks) == height

    # Lopass-filter masks
    # lomasks[0] is the lo0mask
    assert len(lomasks) == height + 1

    #
    # --------------------
    # Re-composition
    # --------------------
    #

    # Compute energy for each frequency subband
    for (scale_idx, orientation_idx), banddft in pyr_bands.items():
        anglemask = all_anglemasks[scale_idx][orientation_idx]
        himask = himasks[scale_idx]
        resdft = ((np.power(-1 + 0j, 0.5)) ** order * banddft * anglemask * himask)

        # Upsampling
        for s in range(scale_idx, 0, -1):
            lomask = lomasks[s]

            # This should be the next higher resolution
            nresdft = np.zeros(dim_list[s]) + 0j

            # Copy the previous spectrum into the center
            nresdft[bound_list[s][0]:bound_list[s][2], bound_list[s][1]:bound_list[s][3]] = resdft * lomask

            resdft = nresdft

        # Finally, apply the lo0 mask
        resdft = resdft * lo0mask

        # Calculate the energy
        band_energies[(scale_idx, orientation_idx)] = np.mean(np.abs(resdft) ** 2)

    return band_energies


def calculate_energy(pyr, scale, band):
    """
    Calculate the energy in a specific frequency sub-band
    :param pyr: steerable pyramid object
    :param scale: level index
    :param band: orientation index
    :return: energy in given subband
    """
    # Reconstruct given subband by its index
    band_reconstruction = pyr.recon_pyr(levels=scale, bands=band)

    # Fourier transform
    band_reconstruction_fft = np.fft.fftshift(np.fft.fft2(band_reconstruction))

    # Calculate energy
    band_energy = np.mean(np.abs(band_reconstruction_fft) ** 2)

    return band_energy


def calculate_band_energies_original(image, height, order, num_orientations_of_interest=2, twidth=1):
    """
    Calculate band energies using the full decomposition provided by pyrtools
    :param image: square grayscale image
    :param height: number of scales
    :param order: Gaussian derivative order. This is the number of orientations - 1.
    :param num_orientations_of_interest: number of equally-spaced orientation bands of interest. Must be a divisor of the number orientations. The default is 2.
    :param twidth: width of the transition region of the radial lowpass function, in octaves
    :return: energies of shape [num_scales, 2]. The two columns contain the horizontal and vertical frequency subbands at each scale.
    """
    num_orientations = order + 1

    pyr = pt.pyramids.SteerablePyramidFreq(image, height=height, order=order, twidth=twidth)

    all_energies = []

    for scale_idx in range(pyr.num_scales):
        scale_energies = []

        # If `num_orientations_of_interest = 2`, then band 0 is west and east, and band 1 is north and south.
        for orientation_idx in np.arange(start=0, stop=num_orientations, step=num_orientations // num_orientations_of_interest):
            band = calculate_energy(pyr, scale=scale_idx, band=orientation_idx)
            scale_energies.append(band)

        all_energies.append(scale_energies)

    # Concatenate column vectors
    energies = np.array(all_energies)

    return energies

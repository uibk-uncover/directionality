import jpeglib
import argparse
from glob import glob
import os
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from utils.quantization_table import StandardQuantizationTableChecker


def collect_jpeg_parameters(filepath, standard_qt_checker=None):
    """
    Extract properties from a given JPEG image.

    We assume that the image is stored in YCbCr and that there are only one luminance and one chrominance quantization table.

    :param filepath: path to a JPEG image
    :param standard_qt_checker: instance of a standard quantization table checker. If None, a new instance will be created.
    :return: dict with several JPEG properties
    """

    # Create quantization table checker
    if standard_qt_checker is None:
        standard_qt_checker = StandardQuantizationTableChecker()

    result = {
        "filepath": filepath,
    }

    # Read metadata (without decoding the image)
    im = jpeglib.read_dct(filepath)

    # Resolution
    result["height"] = im.height
    result["width"] = im.width

    # Chroma subsampling factors
    result["sampling_factors"] = im.samp_factor

    # Quantization tables
    luma_qt = im.qt[im.quant_tbl_no[0]]
    result["luma_qt"] = luma_qt

    # Is the luminance quantization table symmetric?
    result["is_luma_qt_symmetric"] = np.all(luma_qt == luma_qt.T)

    # Is the luminance QT a scaled version of the "standard" luminance QT?
    result["is_luma_qt_standard"] = standard_qt_checker.is_standard_luma_qt(luma_qt)

    # If yes, what is the scaling factor, i.e., the JPEG quality factor?
    result["luma_qf"] = standard_qt_checker.identify_luma_qf(luma_qt)

    # For color images, repeat the steps for the first chroma quantization table
    if im.has_chrominance:
        chroma_qt = im.qt[im.quant_tbl_no[1]]
        result["chroma_qt"] = chroma_qt

        # Is the chrominance quantization table symmetric?
        result["is_chroma_qt_symmetric"] = np.all(chroma_qt == chroma_qt.T)

        # Is the luminance QT a scaled version of the "standard" luminance QT?
        result["is_chroma_qt_standard"] = standard_qt_checker.is_standard_chroma_qt(chroma_qt)

        # If yes, what is the scaling factor, i.e., the JPEG quality factor?
        result["chroma_qf"] = standard_qt_checker.identify_chroma_qf(chroma_qt)

    # Read the EXIF orientation flag
    img = Image.open(filepath)
    exif = img._getexif()

    orientation_exif_key = 0x0112
    if exif and orientation_exif_key in exif:
        orientation_flag = exif.get(0x0112)

        # See https://jdhao.github.io/2019/07/31/image_rotation_exif_info/ for an explanation
        orientation_str = {
            1: "normal", # Normal orientation
            2: "flip_left_right", # Horizontal flip
            3: "rotate_180", # Rotate by 180 degrees before displaying
            4: "flip_top_bottom", # Vertical flip
            5: "transpose", # Rotate by 270 degrees in counter-clockwise direction, then flip horizontally
            6: "rotate_270", # Rotate by 270 degrees in counter-clockwise direction before displaying
            7: "transverse", # Rotate by 90 degrees in counter-clockwise direction, then flip horizontally
            8: "rotate_90" # Rotate by 90 degrees in counter-clockwise direction before displaying
        }.get(orientation_flag)

        result["exif_orientation"] = orientation_str
    else:
        result["exif_orientation"] = None

    return result


def collect_jpeg_parameters_loop(filepaths):
    """
    Iterates over a list of JPEG images, extracts parameters from each image, and concatenates the results to a data frame.
    :param filepaths: list of filepaths
    :return: data frame
    """
    standard_qt_checker = StandardQuantizationTableChecker()
    buffer = []

    for filepath in tqdm(filepaths):
        try:
            result = collect_jpeg_parameters(filepath, standard_qt_checker)
            buffer.append(result)
        except Exception as e:
            print(f"Skipping image \"{filepath}\"")
            continue

    return pd.DataFrame(buffer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str, help="Directory where to look for input files")
    parser.add_argument("--output_dir", type=str, help="(Optional) Directory where to store resulting csv file")
    args = vars(parser.parse_args())

    # Recursively search for JPEG files in the input directory
    filepaths = []
    for file_type in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        filepaths.extend(glob(os.path.join(args["input_dir"], "**", "*" + file_type), recursive=True))

    if len(filepaths) == 0:
        print("No images found")
        exit(1)

    # Sort files
    filepaths = sorted(filepaths)

    # Collect statistics for each file
    df = collect_jpeg_parameters_loop(filepaths)

    # Make image filepaths relative to given input directory
    df["filepath"] = df["filepath"].map(lambda f: os.path.relpath(f, args["input_dir"]))

    # Concatenate output filepath
    output_filename = time.strftime("%Y_%m_%d") + "-collect_jpeg_statistics.csv"
    if args["output_dir"] is not None:
        output_dir = args["output_dir"]
    else:
        output_dir = args["input_dir"]

    output_filepath = os.path.join(output_dir, output_filename)
    df.to_csv(output_filepath, index=False)

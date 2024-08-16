import numpy as np
import tempfile
import jpeglib
import ast
import re


# Sample quantization tables given in Annex K of Recommendation ITU-T T.81 (1992) | ISO/IEC 10918-1:1994.
std_luminance_quant_tbl = np.array([
  [16,  11,  10,  16,  24,  40,  51,  61],
  [12,  12,  14,  19,  26,  58,  60,  55],
  [14,  13,  16,  24,  40,  57,  69,  56],
  [14,  17,  22,  29,  51,  87,  80,  62],
  [18,  22,  37,  56,  68, 109, 103,  77],
  [24,  35,  55,  64,  81, 104, 113,  92],
  [49,  64,  78,  87, 103, 121, 120, 101],
  [72,  92,  95,  98, 112, 100, 103,  99],
])

std_chrominance_quant_tbl = np.array([
  [16,  18,  24,  47,  99,  99,  99,  99],
  [18,  21,  26,  66,  99,  99,  99,  99],
  [24,  26,  56,  99,  99,  99,  99,  99],
  [47,  66,  99,  99,  99,  99,  99,  99],
  [99,  99,  99,  99,  99,  99,  99,  99],
  [99,  99,  99,  99,  99,  99,  99,  99],
  [99,  99,  99,  99,  99,  99,  99,  99],
  [99,  99,  99,  99,  99,  99,  99,  99],
])


def qf_to_std_qt(qf):
    """
    The scaling is defined in libjpeg's jcparam.c with the following description:

    The basic table is used as-is (scaling 100) for a quality of 50.
    Qualities 50..100 are converted to scaling percentage 200 - 2*Q;
    note that at Q=100 the scaling is 0, which will cause jpeg_add_quant_table to make all the table entries 1 (hence, minimum quantization loss).
    Qualities 1..50 are converted to scaling percentage 5000/Q.

    :param qf: JPEG quality factor
    :return: quantization table of shape [8, 8]
    """
    # Scale quality
    if qf <= 0:
        qf = 1
    if qf > 100:
        qf = 100

    if qf < 50:
        scale_factor = 5000 / qf
    else:
        scale_factor = 200 - qf * 2

    luma_tbl = (std_luminance_quant_tbl * scale_factor + 50) // 100
    chroma_tbl = (std_chrominance_quant_tbl * scale_factor + 50) // 100

    # Clip to valid range
    luma_tbl = np.clip(luma_tbl, a_min=1, a_max=32767)
    chroma_tbl = np.clip(chroma_tbl, a_min=1, a_max=32767)

    return luma_tbl, chroma_tbl


def load_qt(filepath):
    """
    Loads quantization tables from given image using jpeglib.
    """
    return jpeglib.read_dct(filepath).qt


def qf_to_qt(qf, libjpeg_version="6b"):
    """
    Compress a dummy image with the given quality factor and load its quantization table
    :param qf: JPEG quality factor
    :param libjpeg_version: libjpeg version to be passed to jpeglib
    :return: quantization tables created by jpeglib
    """
    dummy_img = np.random.randint(low=0, high=256, dtype=np.uint8, size=(64, 64, 3))

    im = jpeglib.from_spatial(dummy_img)

    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        with jpeglib.version(libjpeg_version):
            im.write_spatial(f.name, qt=qf)

        return load_qt(f.name)


def create_qt_to_qf_mapping(libjpeg_version="6b"):
    """
    Iterate over all JPEG quality factors and store the quantization tables in a dictionary.
    :param libjpeg_version: libjpeg version to be passed to jpeglib
    :return: dict where the keys are the quantization tables encoded as string, and the values are the corresponding quality factors.
    """
    dummy_img = np.random.randint(low=0, high=256, dtype=np.uint8, size=(64, 64, 3))

    mapping = {}

    for quality in range(0, 101):
        im = jpeglib.from_spatial(dummy_img)

        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            with jpeglib.version(libjpeg_version):
                im.write_spatial(f.name, qt=quality)

            qt = qf_to_qt(f.name)
            key = str(qt)

            mapping[key] = quality

    return mapping


def estimate_qf(filepath, qt_to_qf_map):
    """
    Reconstruct the JPEG quality factor from a given JPEG file by comparing it to a set of known quantization tables
    :param filepath: path to JPEG file
    :param qt_to_qf_map: dict where the keys are the quantization tables encoded as string, and the values are the corresponding quality factors.
    :return: JPEG quality factor, or None if not present in the given map
    """
    qt = load_qt(filepath)
    key = str(qt)

    if key not in qt_to_qf_map:
        print(f"Could not find quality for image \"{filepath}\"")
        return None

    return qt_to_qf_map.get(key, None)


def compute_qt_dissimilarity(qt1, qt2):
    """
    Computes the quantization table "dissimilarity" semi-metric defined in Yousfi, Fridrich: JPEG Steganalysis Detectors Scalable With Respect to Compression Quality.
    http://www.ws.binghamton.edu/fridrich/research/scalable_jpeg_steganalysis.pdf

    The semi-metric computes the weighted sum of relative differences between the quantization factors.
    The weights are larger for low spatial frequencies (upper left of the QT) nad lower for high spatial frequencies (lower right of the QT).
    If QT1 and QT2 are identical, the dissimilarity score is 0.

    :param qt1: quantization table of shape [8, 8]
    :param qt2: quantization table of shape [8, 8]
    :return: scalar dissimilarity score
    """
    squared_relative_diffs = ((qt1 - qt2) / (qt1 + qt2)) ** 2

    # Low spatial frequencies are assigned higher weights. High spatial frequencies are assigned lower weights.
    ks = np.arange(1, 9)
    weights = ks[:, None] + ks[None, :]
    weights = 1. / (weights ** 2)

    squared_dissimilarity = np.sum(weights * squared_relative_diffs)

    return np.sqrt(squared_dissimilarity)


def str_to_array(s):
    """
    Parse an array from a string.
    This can be used to reconstruct arrays stored in csv files.
    :param s: string
    :return: ndarray
    """

    # Remove space after [
    s = re.sub('\[ +', '[', s.strip())

    # Replace commas and spaces
    s = re.sub('[,\s]+', ', ', s)

    return np.array(ast.literal_eval(s))


class QuantizationTableStore(object):
    """
    Simple database objects that stores a set of quantization tables.
    This can be used to check whether a given quantization table has already been seen.
    """
    def __init__(self):
        self.qt = dict()

    @staticmethod
    def _to_key(qt):
        assert qt.shape == (8, 8)
        return str(qt)

    def add(self, qt, val):
        key = self._to_key(qt)
        existing_val = self.qt.get(key, None)

        # key is not yet known
        if existing_val is None:
            self.qt[key] = val

        # key is known, and value differs
        elif existing_val != val:
            raise ValueError("Duplicate QT with with different value")

    def __contains__(self, qt):
        assert qt.shape == (8, 8)
        key = self._to_key(qt)
        return key in self.qt

    def get(self, qt, default=None):
        """
        Return the value with the specific key
        :param qt: 8x8 quantization table
        :param default: a value to return if the specified key does not exist
        :return:
        """
        assert qt.shape == (8, 8)

        key = self._to_key(qt)
        return self.qt.get(key, default)


class StandardQuantizationTableChecker(object):
    def __init__(self):
        self.luma_store = QuantizationTableStore()
        self.chroma_store = QuantizationTableStore()

        # Add standard quantization table in all variants to our database of known tables
        for qf in range(1, 101):
            qts = qf_to_qt(qf, libjpeg_version="6b")
            self.luma_store.add(qts[0], qf)
            self.chroma_store.add(qts[1], qf)

            # libjpeg 9e changed the chroma DC quantization factor
            qts = qf_to_qt(qf, libjpeg_version="9e")
            self.luma_store.add(qts[0], qf)
            self.chroma_store.add(qts[1], qf)

    def is_standard_luma_qt(self, qt):
        assert qt.shape == (8, 8), "Expected single luminance QT"
        return qt in self.luma_store

    def is_standard_chroma_qt(self, qt):
        assert qt.shape == (8, 8), "Expected single chroma QT"
        return qt in self.chroma_store

    def identify_luma_qf(self, qt):
        assert qt.shape == (8, 8), "Expected single luminance QT"
        return self.luma_store.get(qt)

    def identify_chroma_qf(self, qt):
        assert qt.shape == (8, 8), "Expected single chroma QT"
        return self.chroma_store.get(qt)

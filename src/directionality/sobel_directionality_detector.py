import numpy as np
from skimage.filters import sobel_h, sobel_v
from directionality.directionality_detector import DirectionalityDetector


class SobelDirectionalityDetector(DirectionalityDetector):

    def evaluate_directional_statistics(self, img):
        """
        Compute directional statistics
        :param img: grayscale image
        :return: 2-tuple
            Gradient magnitude of the vertical edges (corresponds to horizontal frequencies)
            Gradient magnitude of the horizontal edges (corresponds to vertical frequencies)
        """

        # Find the vertical edges (horizontal frequencies)
        # sobel_v uses the following kernel (scale factor 1/4):
        # [ 1,  0, -1]
        # [ 2,  0, -2]
        # [ 1,  0, -1]
        grad_vertical_edges = sobel_v(img)

        # Find the horizontal edges (vertical frequencies)
        # sobel_h uses the following kernel (scale factor 1/4):
        # [ 1,  2,  1]
        # [ 0,  0,  0]
        # [-1, -2, -1]
        grad_horizontal_edges = sobel_h(img)

        # grad_magnitude = np.sqrt(grad_v ** 2 + grad_h ** 2)
        grad_magnitude_vertical_edges = np.mean(np.sqrt(grad_vertical_edges ** 2))
        grad_magnitude_horizontal_edges = np.mean(np.sqrt(grad_horizontal_edges ** 2))

        return grad_magnitude_vertical_edges, grad_magnitude_horizontal_edges

    def evaluate_directionality_score(self, img):
        """
        Quantify directionality of a given image
        :param img: 2D image
        :return: scalar value d, where
            d > 0 means that the image is dominated by vertical edges,
            d < 0 means that the image is dominated by horizontal edges.
        """
        grad_magnitude_vertical_edges, grad_magnitude_horizontal_edges = self.evaluate_directional_statistics(img)
        return (grad_magnitude_vertical_edges - grad_magnitude_horizontal_edges) / (grad_magnitude_vertical_edges + grad_magnitude_horizontal_edges)

    @classmethod
    def name(cls):
        """
        :return: Return the detector name
        """
        return "SobelDirectionalityDetector"

    def params(self):
        return {}

import abc


class DirectionalityDetector(abc.ABC):

    @abc.abstractmethod
    def evaluate_directional_statistics(self, img):
        """
        Evaluates statistics in several image directions
        :param img: grayscale image
        :return: exact return value varies between detectors
        """
        pass

    @abc.abstractmethod
    def evaluate_directionality_score(self, img):
        """
        Quantifies with a single scalar number how directional the statistics of the given image are
        :param img: grayscale image
        :return: directionality score
        """
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls):
        """
        :return: Return the detector name
        """
        pass

    @abc.abstractmethod
    def params(self):
        """
        Return a dict with the detector's configuration
        """
        pass

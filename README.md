# Directionality Matters

*Benedikt Lorch and Rainer Böhme, "Landscape More Secure Than Portrait? Zooming Into the Directionality of Digital Images With Security Implications", USENIX Security Symposium, 2024*. [PDF](https://arxiv.org/pdf/2406.15206)

This repository contains the following code:
- Directionality detector based on [steerable pyramids](src/directionality/steerable_pyramids_directionality_detector.py) and based on the [Sobel filter](src/directionality/sobel_directionality_detector.py).
- Code to [produce the synthetic symmetric images](src/synthetic_images/create_symmetric_images.py) and the [transformed versions](src/synthetic_images/create_transformed_images.py), which were used to compare the directionality detectors based on steerable pyramids and the Sobel filter (Appendix B).
- [Script](src/collect_jpeg_parameters.py) to collect JPEG parameters from a directory.

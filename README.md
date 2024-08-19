# Directionality Matters

*Benedikt Lorch and Rainer Böhme, "Landscape More Secure Than Portrait? Zooming Into the Directionality of Digital Images With Security Implications", USENIX Security Symposium, 2024*. [PDF](https://arxiv.org/pdf/2406.15206)

This repository contains the following code:
- Directionality detector based on [steerable pyramids](src/directionality/steerable_pyramids_directionality_detector.py) and based on the [Sobel filter](src/directionality/sobel_directionality_detector.py).
- Code to [produce the synthetic symmetric images](src/synthetic_images/create_symmetric_images.py) and the [transformed versions](src/synthetic_images/create_transformed_images.py), which were used to compare the directionality detectors based on steerable pyramids and the Sobel filter (Appendix B).
- [Script](src/collect_jpeg_parameters.py) to collect JPEG parameters from a directory.

Additionally, we provide the predictions for CNN-based experiments. While the paper only reports the accuracy, the predictions allow you evaluating additional metrics, such as false positive and false negative rates.

| Task                        | Experimental Setup                     | Test image orientation | Prediction vectors  |
|-----------------------------|----------------------------------------|------------------------|---------------------|
| Steganalysis                | Standard QF-75 QT, J-UNIWARD, no-rot   | org                    | TODO                |
| Steganalysis                | Standard QF-75 QT, J-UNIWARD, no-rot   | rot 90                 | TODO                |
| Steganalysis                | Standard QF-75 QT, J-UNIWARD, base-rot | org                    | TODO                |
| Steganalysis                | Standard QF-75 QT, J-UNIWARD, base-rot | rot 90                 | TODO                |
| Steganalysis                | Standard QF-75 QT, J-UNIWARD, aug-rot  | org                    | TODO                |
| Steganalysis                | Standard QF-75 QT, J-UNIWARD, aug-rot  | rot 90                 | TODO                |
| Steganalysis                | Standard QF-75 QT, UERD, no-rot        | org                    | TODO                |
| Steganalysis                | Standard QF-75 QT, UERD, no-rot        | rot 90                 | TODO                |
| Steganalysis                | Standard QF-75 QT, UERD, base-rot      | org                    | TODO                |
| Steganalysis                | Standard QF-75 QT, UERD, base-rot      | rot 90                 | TODO                |
| Steganalysis                | Standard QF-75 QT, UERD, aug-rot       | org                    | TODO                |
| Steganalysis                | Standard QF-75 QT, UERD, aug-rot       | rot 90                 | TODO                |
| Steganalysis                | Standard QF-75 QT, nsF5, no-rot        | org                    | TODO                |
| Steganalysis                | Standard QF-75 QT, nsF5, no-rot        | rot 90                 | TODO                |
| Steganalysis                | Standard QF-75 QT, nsF5, base-rot      | org                    | TODO                |
| Steganalysis                | Standard QF-75 QT, nsF5, base-rot      | rot 90                 | TODO                |
| Steganalysis                | Standard QF-75 QT, nsF5, aug-rot       | org                    | TODO                |
| Steganalysis                | Standard QF-75 QT, nsF5, aug-rot       | rot 90                 | TODO                |
| Camera model identification | no-rot                                 | org                    | TODO                |
| Camera model identification | no-rot                                 | rot 90                 | TODO                |
| Camera model identification | aug-rot                                | org                    | TODO                |
| Camera model identification | aug-rot                                | rot 90                 | TODO                |
| Synthetic image detection   | DALL-E Mini, no-rot                    | org                    | *not available*[^1] |
| Synthetic image detection   | DALL-E Mini, no-rot                    | rot 90                 | *not available*[^1] |
| Synthetic image detection   | Stable Diffusion XL, no-rot            | org                    | TODO                |
| Synthetic image detection   | Stable Diffusion XL, no-rot            | rot 90                 | TODO                |

[^1]: We already cleaned up the 
"""
MirrorMind
Preprocessing Module

Author: Nistha Agrawal
Project: MirrorMind

This module handles image preprocessing before OCR.
"""

import cv2


class ImagePreprocessor:
    """
    Image preprocessing pipeline.
    """

    def __init__(
        self,
        max_width=1024,
        grayscale=True,
        denoise=True,
        clahe=True,
        sharpen=True
    ):

        self.max_width = max_width
        self.grayscale = grayscale
        self.denoise = denoise
        self.clahe = clahe
        self.sharpen = sharpen

    def resize(self, image):

        height, width = image.shape[:2]

        if width <= self.max_width:
            return image

        ratio = self.max_width / width

        new_height = int(height * ratio)

        return cv2.resize(
            image,
            (self.max_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    def convert_grayscale(self, image):

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    def denoise_image(self, image):

        if len(image.shape) == 2:

            return cv2.fastNlMeansDenoising(
                image,
                None,
                10,
                7,
                21
            )

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            10,
            10,
            7,
            21
        )

    def enhance_contrast(self, image):

        if len(image.shape) == 2:

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            return clahe.apply(image)

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        return cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

    def sharpen_image(self, image):

        blurred = cv2.GaussianBlur(
            image,
            (0, 0),
            3
        )

        return cv2.addWeighted(
            image,
            1.5,
            blurred,
            -0.5,
            0
        )

    def process(self, image):

        image = self.resize(image)

        if self.grayscale:
            image = self.convert_grayscale(image)

        if self.denoise:
            image = self.denoise_image(image)

        if self.clahe:
            image = self.enhance_contrast(image)

        if self.sharpen:
            image = self.sharpen_image(image)

        return image

    def process_file(self, input_path, output_path):

        image = cv2.imread(input_path)

        if image is None:
            raise FileNotFoundError(
                f"Unable to read {input_path}"
            )

        image = self.process(image)

        cv2.imwrite(output_path, image)

        return output_path
"""Small helpers for the image experiments: sample photographs, error measures, file sizes."""
import os

import numpy as np
from PIL import Image
from sklearn.datasets import load_sample_image


def load_sample(name="china.jpg"):
    """One of the two photographs that ship with scikit-learn, as a (427, 640, 3) uint8 array.

    "china.jpg" has fine detail everywhere; "flower.jpg" has large smooth regions. Both are
    licensed Creative Commons Attribution 2.0, so the experiments need no image files.
    """
    return np.asarray(load_sample_image(name), dtype=np.uint8)


def psnr(reference, image, mask=None, peak=255.0):
    """Peak signal-to-noise ratio in dB (higher is closer to the reference).

    With a boolean (H, W) mask the error is measured on the masked pixels only.
    """
    reference = np.asarray(reference, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    if mask is not None:
        reference, image = reference[mask], image[mask]
    mse = np.mean((reference - image) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(peak ** 2 / mse))


def apply_palette(labels, palette, height, width):
    """Build an image in which pixel n has the colour palette[labels[n]].

    Output:
    - ndarray (Shape: (height, width, C), dtype uint8). The palette is rounded to the nearest
      intensity; writing floats straight into a uint8 array would truncate them and darken
      the image by half a level on average.
    """
    colours = np.clip(np.rint(palette), 0, 255).astype(np.uint8)
    return colours[np.asarray(labels)].reshape(height, width, -1)


def png_size(image, path):
    """Save a (H, W, 3) uint8 image as a 24-bit PNG and return the file size in bytes."""
    Image.fromarray(image).save(path, format="PNG")
    return os.path.getsize(path)


def indexed_png_size(labels, palette, height, width, path):
    """Save the image as a palette PNG (one index per pixel plus the colours) and return its size.

    This is the format that stores exactly what colour quantization produces. It needs at
    most 256 colours.
    """
    colours = np.clip(np.rint(palette), 0, 255).astype(np.uint8)
    assert len(colours) <= 256, "A palette PNG holds at most 256 colours."
    indexed = Image.fromarray(np.asarray(labels, dtype=np.uint8).reshape(height, width), mode="P")
    indexed.putpalette(colours.flatten().tolist())
    indexed.save(path, format="PNG")
    return os.path.getsize(path)

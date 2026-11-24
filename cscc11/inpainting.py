"""Image inpainting with RBF regression.

Some pixels of an image are missing (covered by text, scratched, ...). Images vary smoothly
over short distances, so each colour channel is treated as a function of pixel position and
fitted, one small patch at a time, by RBF regression on the pixels that are known. The fit
is then evaluated at the missing pixels.
"""
import numpy as np

from cscc11.rbf import RBFRegression


def colour_mask(image, colour, tolerance=0.1):
    """Pixels whose squared distance to `colour` is at most `tolerance`: a boolean (H, W) array.

    Use it to mark as missing everything painted in one known colour.
    """
    difference = np.asarray(image, dtype=np.float64) - np.asarray(colour, dtype=np.float64)
    return np.sum(difference ** 2, axis=2) <= tolerance


def inpaint(image, mask, spacing=3, width=2.0, l2=0.5, patch_size=25):
    """Fill in the masked pixels.

    Args:
    - image (ndarray (Shape: (H, W, C))): values between 0 and 1. It is not modified.
    - mask (ndarray (Shape: (H, W), dtype bool)): True where the pixel is missing.
    - spacing (int): distance in pixels between neighbouring RBF centers.
    - width (float): width of every RBF, in pixels.
    - l2 (float): regularization constant of the regression.
    - patch_size (int): side of the square patches that are filled in one at a time.

    Output:
    - ndarray (Shape: (H, W, C)): the image with the masked pixels replaced. The model is not
      constrained to [0, 1]; clip before displaying.

    Each patch is fitted on the known pixels of the patch grown by `spacing` pixels on every
    side, so that pixels on the edge of the patch also have neighbours to learn from. Only
    pixels that were known in the input are used for fitting.
    """
    image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    height, width_px = mask.shape
    restored = image.copy()

    for top in range(0, height, patch_size):
        for left in range(0, width_px, patch_size):
            bottom, right = min(top + patch_size, height), min(left + patch_size, width_px)
            patch_missing = mask[top:bottom, left:right]
            if not patch_missing.any():
                continue

            # Grow the region until it holds more known pixels than the model has weights.
            margin = spacing
            while True:
                r0, r1 = max(top - margin, 0), min(bottom + margin, height)
                c0, c1 = max(left - margin, 0), min(right + margin, width_px)
                grid_x, grid_y = np.meshgrid(np.arange(c0, c1 + 1, spacing), np.arange(r0, r1 + 1, spacing))
                centers = np.column_stack((grid_x.ravel(), grid_y.ravel()))
                known = ~mask[r0:r1, c0:c1]
                covers_image = (r0, c0, r1, c1) == (0, 0, height, width_px)
                if known.sum() > len(centers) + 1 or covers_image:
                    break
                margin *= 2
            if known.sum() == 0:
                continue  # nothing to learn from anywhere

            cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
            model = RBFRegression(centers, width)
            model.fit(np.column_stack((cols[known], rows[known])), image[r0:r1, c0:c1][known], l2)

            fill_rows, fill_cols = np.nonzero(patch_missing)
            fill_rows, fill_cols = fill_rows + top, fill_cols + left
            restored[fill_rows, fill_cols] = model.predict(np.column_stack((fill_cols, fill_rows)))
    return restored

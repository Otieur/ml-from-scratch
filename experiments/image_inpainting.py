"""Remove text painted over photographs with RBF regression: python -m experiments.image_inpainting

Text is painted over two clean photographs, so the inpainted pixels can be compared with
the originals. PSNR is measured on the painted pixels only.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.common import OUT, save_metrics
from mlscratch.imaging import load_sample, psnr
from mlscratch.inpainting import colour_mask, inpaint

DEFAULTS = {"spacing": 3, "width": 2, "l2": 0.5}
LINES = ["Radial basis functions", "fill in the missing pixels", "one patch at a time,",
         "using the pixels around", "them as training data.", "Each colour channel is", "fitted on its own."]


def text_mask(height, width, seed=0):
    """A boolean (H, W) mask shaped like lines of text, drawn without anti-aliasing."""
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=30)
    rng = np.random.default_rng(seed)
    y = 14
    for line in LINES:
        draw.text((int(rng.integers(6, 40)), y), line, fill=1, font=font)
        y += 62
    return np.array(canvas, dtype=bool)


def unused_colour(image):
    """A pure colour that no pixel of the image is close to, so that only the painted text is masked."""
    for name, colour in (("red", [1.0, 0.0, 0.0]), ("blue", [0.0, 0.0, 1.0]), ("green", [0.0, 1.0, 0.0])):
        if not colour_mask(image, colour).any():
            return name, np.array(colour)
    raise ValueError("every candidate colour already occurs in the image")


def experiment(name):
    clean = load_sample(name).astype(np.float64) / 255
    colour_name, colour = unused_colour(clean)
    painted = text_mask(*clean.shape[:2])
    corrupted = clean.copy()
    corrupted[painted] = colour
    mask = colour_mask(corrupted, colour)      # found from the colour, as it would be in practice
    assert np.array_equal(mask, painted)

    start = time.perf_counter()
    restored = np.clip(inpaint(corrupted, mask, **DEFAULTS), 0, 1)
    seconds = time.perf_counter() - start

    sweep = []
    for spacing in (2, 3, 4, 5, 6):
        for width in (1, 2, 3, 4, 6, 8):
            if width > 2 * spacing:
                continue
            for l2 in (0.001, 0.05, 0.5):
                result = np.clip(inpaint(corrupted, mask, spacing, width, l2), 0, 1)
                sweep.append({"spacing": spacing, "width": width, "l2": l2,
                              "psnr": psnr(clean, result, mask, peak=1.0)})
    best = max(sweep, key=lambda row: row["psnr"])
    return {
        "clean": clean, "corrupted": corrupted, "restored": restored, "mask": mask,
        "numbers": {"image": name, "height": int(clean.shape[0]), "width": int(clean.shape[1]),
                    "text_colour": colour_name, "pixels_filled": int(mask.sum()),
                    "share_filled": float(mask.mean()),
                    "psnr_corrupted": psnr(clean, corrupted, mask, peak=1.0),
                    "psnr_inpainted": psnr(clean, restored, mask, peak=1.0),
                    "seconds": seconds, "settings": DEFAULTS, "best_settings": best, "sweep": sweep},
    }


def main():
    smooth = experiment("flower.jpg")       # large smooth regions
    detailed = experiment("china.jpg")      # fine detail everywhere

    detail = (slice(120, 260), slice(20, 230))
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.3), gridspec_kw={"width_ratios": [1.5, 1.5, 1.5, 1.5]})
    for row, run in enumerate((smooth, detailed)):
        n = run["numbers"]
        panels = ((run["corrupted"], f"Text over {n['share_filled']:.1%} of the pixels"),
                  (run["restored"], f"Inpainted: {n['psnr_inpainted']:.1f} dB on the filled pixels"),
                  (run["corrupted"][detail], "Detail, before"),
                  (run["restored"][detail], "Detail, after"))
        for ax, (picture, title) in zip(axes[row], panels):
            ax.imshow(picture, interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "image_inpainting.png", dpi=100)
    plt.close(fig)

    sweep = smooth["numbers"]["sweep"]
    spacings = sorted({r["spacing"] for r in sweep})
    widths = sorted({r["width"] for r in sweep})
    l2_values = sorted({r["l2"] for r in sweep})
    fig, axes = plt.subplots(1, len(l2_values), figsize=(4.6 * len(l2_values), 4.2), sharey=True)
    lo, hi = min(r["psnr"] for r in sweep), max(r["psnr"] for r in sweep)
    for ax, l2 in zip(axes, l2_values):
        grid = np.full((len(spacings), len(widths)), np.nan)
        for r in sweep:
            if r["l2"] == l2:
                grid[spacings.index(r["spacing"]), widths.index(r["width"])] = r["psnr"]
        image = ax.imshow(grid, vmin=lo, vmax=hi, cmap="viridis", aspect="auto")
        for i in range(len(spacings)):
            for j in range(len(widths)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=9,
                            color="black" if grid[i, j] > lo + 0.7 * (hi - lo) else "white")
        ax.set(xticks=range(len(widths)), xticklabels=widths, yticks=range(len(spacings)), yticklabels=spacings,
               xlabel="RBF width (pixels)", title=f"regularization = {l2}")
    axes[0].set_ylabel("Spacing between RBF centers (pixels)")
    fig.colorbar(image, ax=axes, label="PSNR on the filled pixels (dB), flower.jpg", shrink=0.9)
    fig.savefig(OUT / "image_inpainting_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    save_metrics("image_inpainting", {"flower": smooth["numbers"], "china": detailed["numbers"]})
    for run in (smooth, detailed):
        n, b = run["numbers"], run["numbers"]["best_settings"]
        print(f"inpainting: {n['image']:10s} {n['psnr_corrupted']:.2f} dB with the text -> {n['psnr_inpainted']:.2f} dB "
              f"inpainted ({n['seconds']:.1f} s); best of the sweep {b['psnr']:.2f} dB at spacing {b['spacing']}, "
              f"width {b['width']}, regularization {b['l2']}")


if __name__ == "__main__":
    main()

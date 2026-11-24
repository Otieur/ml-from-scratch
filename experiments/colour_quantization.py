"""Reduce a photograph to K colours with K-means and with a Gaussian mixture:
python -m experiments.colour_quantization
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import OUT, save_metrics
from mlscratch.gmm import gmm_cluster
from mlscratch.imaging import apply_palette, indexed_png_size, load_sample, png_size, psnr
from mlscratch.kmeans import kmeans

IMAGE = "china.jpg"


def main():
    image = load_sample(IMAGE)
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)
    scratch = OUT / "colour_quantization_tmp.png"
    original_size = png_size(image, scratch)

    rows, kept = [], {}
    for k in (2, 4, 8, 16, 32, 64):
        for method in ("kmeans", "gmm"):
            start = time.perf_counter()
            if method == "kmeans":
                palette, labels = kmeans(pixels, k, seed=0, n_init=3)
            else:
                palette, labels = gmm_cluster(pixels, k, seed=0)
            seconds = time.perf_counter() - start
            quantized = apply_palette(labels, palette, height, width)
            rgb_size = png_size(quantized, scratch)
            indexed_size = indexed_png_size(labels, palette, height, width, scratch)
            rows.append({"method": method, "k": k, "psnr": psnr(image, quantized),
                         "png_bytes": rgb_size, "ratio": original_size / rgb_size,
                         "indexed_png_bytes": indexed_size, "indexed_ratio": original_size / indexed_size,
                         "seconds": seconds})
            kept[(method, k)] = quantized
    scratch.unlink()

    ks = sorted({r["k"] for r in rows})
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.4))
    panels = [("Original", image)] + [(f"K-means, K = {k}", kept[("kmeans", k)]) for k in ks] + [("Gaussian mixture, K = 16", kept[("gmm", 16)])]
    for ax, (title, picture) in zip(axes.ravel(), panels):
        ax.imshow(picture)
        if title != "Original":
            method = "gmm" if title.startswith("Gaussian") else "kmeans"
            k = int(title.split("=")[1])
            r = next(r for r in rows if r["method"] == method and r["k"] == k)
            title += f": {r['psnr']:.1f} dB, {r['ratio']:.1f}x smaller"
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "colour_quantization.png", dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for method, label, marker in (("kmeans", "K-means", "o"), ("gmm", "Gaussian mixture", "s")):
        sel = [r for r in rows if r["method"] == method]
        axes[0].plot([r["k"] for r in sel], [r["psnr"] for r in sel], marker=marker, label=label)
        axes[1].plot([r["k"] for r in sel], [r["png_bytes"] / 1024 for r in sel], marker=marker, label=f"{label}, 24-bit PNG")
        axes[1].plot([r["k"] for r in sel], [r["indexed_png_bytes"] / 1024 for r in sel], marker=marker, linestyle="--",
                     label=f"{label}, palette PNG")
    axes[1].axhline(original_size / 1024, color="gray", linewidth=1)
    axes[1].text(2, original_size / 1024 * 0.94, "original PNG", color="gray")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("Number of colours K")
        ax.legend()
    axes[0].set(ylabel="PSNR (dB)", title="Closeness to the original")
    axes[1].set(ylabel="File size (KiB)", title="Size of the quantized image")
    fig.tight_layout()
    fig.savefig(OUT / "colour_quantization_curves.png", dpi=130)
    plt.close(fig)

    save_metrics("colour_quantization", {"image": IMAGE, "height": height, "width": width,
                                         "original_png_bytes": original_size, "runs": rows})
    for r in rows:
        print(f"quantization: {r['method']:6s} K={r['k']:2d}  PSNR {r['psnr']:5.2f} dB  PNG {r['png_bytes']:6d} B "
              f"({r['ratio']:.2f}x)  palette PNG {r['indexed_png_bytes']:6d} B ({r['indexed_ratio']:.2f}x)  {r['seconds']:.1f} s")


if __name__ == "__main__":
    main()

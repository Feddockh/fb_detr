#!/usr/bin/env python3
# demosaic_ximea_folder.py
# Author: based on Hayden Feddock’s 5×5 Ximea NIR demosaic (2025-07-17)
# Simplified CLI: just src-image-dir and dst-image-dir

import argparse
from pathlib import Path
from typing import Dict, Tuple, Iterable, List

import cv2
import numpy as np

# --------------------- constants (same process as before) --------------------- #
UNCROPPED_H = 1088
UNCROPPED_W = 2048

CROP_TOP   = 3
CROP_LEFT  = 0
CROP_H     = 1080           # 1083 - 3
CROP_W     = 2045           # 2045 - 0

DOWNSAMPLE = 5              # 5×5 mosaic → pick every 5th pixel
DEMOSAIC_H = CROP_H // DOWNSAMPLE   # 216
DEMOSAIC_W = CROP_W // DOWNSAMPLE   # 409

# Wavelength map (row-major over 5×5), preserved exactly
_BW_TABLE = [
    [886, 896, 877, 867, 951],
    [793, 806, 782, 769, 675],
    [743, 757, 730, 715, 690],
    [926, 933, 918, 910, 946],
    [846, 857, 836, 824, 941],
]

# Default RGB band triplet (R,G,B) using your prior choice
DEFAULT_RGB_BANDS = (886, 793, 743)

VALID_EXTS = (".pgm", ".png", ".tif", ".tiff")


# --------------------------- core demosaic functions -------------------------- #
def demosaic_ximea_5x5(image_path: Path, *, sort_bands: bool = True) -> Dict[int, np.ndarray]:
    """
    Returns a dict: {wavelength_nm: band_2d_uint8}, each band sized DEMOSAIC_H × DEMOSAIC_W.
    Uses the exact crop window and 5×5 decimation logic from the reference script.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Validate expected raw size (warn but continue if different and divisible)
    h0, w0 = img.shape
    if (h0, w0) != (UNCROPPED_H, UNCROPPED_W):
        # Still try to crop the same window; raise if it fails
        pass

    # Crop: rows [CROP_TOP : CROP_TOP + CROP_H), cols [CROP_LEFT : CROP_LEFT + CROP_W)
    cropped = img[CROP_TOP:CROP_TOP + CROP_H, CROP_LEFT:CROP_LEFT + CROP_W]
    h, w = cropped.shape
    if h % 5 != 0 or w % 5 != 0:
        raise ValueError(
            f"Cropped dims not divisible by 5 (got {h}×{w}); "
            f"expected multiples of 5 using the fixed crop."
        )

    block_rows, block_cols = h // 5, w // 5

    cube: Dict[int, np.ndarray] = {}
    for ro in range(5):
        for co in range(5):
            band = cropped[ro::5, co::5]
            if band.shape != (block_rows, block_cols):
                raise ValueError(f"Band shape mismatch at ({ro},{co}): {band.shape} vs {(block_rows, block_cols)}")
            cube[_BW_TABLE[ro][co]] = band

    if sort_bands:
        cube = dict(sorted(cube.items()))
    return cube


def bands_to_rgb(cube: Dict[int, np.ndarray], bands_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Stack 3 selected bands into H×W×3 (RGB) and normalize each band to [0,255] uint8.
    """
    try:
        r = cube[bands_rgb[0]]
        g = cube[bands_rgb[1]]
        b = cube[bands_rgb[2]]
    except KeyError as e:
        raise KeyError(f"Requested band {e} not found. Available: {sorted(cube.keys())}") from e

    # Per-band min-max normalize for visualization / ML input
    def norm_u8(arr: np.ndarray) -> np.ndarray:
        out = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        return out.astype(np.uint8)

    rgb = np.stack([norm_u8(r), norm_u8(g), norm_u8(b)], axis=-1)
    return rgb


# ------------------------------- batch utilities ----------------------------- #
def iter_images(src_dir: Path, exts: Iterable[str]) -> List[Path]:
    files = []
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def ensure_outdir(dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------ CLI ------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Demosaic a folder of Ximea 5×5 NIR mosaic images into 3-channel PNGs.\n"
                    "Only two arguments needed: src-image-dir and dst-image-dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("src_image_dir", type=Path, help="Folder containing mosaic images")
    parser.add_argument("dst_image_dir", type=Path, help="Folder to write demosaiced PNGs")
    parser.add_argument(
        "--bands", nargs=3, type=int, metavar=("R", "G", "B"),
        default=DEFAULT_RGB_BANDS,
        help="Wavelengths (nm) to map to R,G,B"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing PNGs if they already exist"
    )
    args = parser.parse_args()

    src_dir: Path = args.src_image_dir
    dst_dir: Path = args.dst_image_dir
    bands_rgb = tuple(args.bands)  # type: ignore[assignment]

    if not src_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {src_dir}")

    ensure_outdir(dst_dir)

    images = iter_images(src_dir, VALID_EXTS)
    if not images:
        raise SystemExit(
            f"No images with extensions {VALID_EXTS} found in {src_dir}"
        )

    print(f"[demosaic] Found {len(images)} images in {src_dir}")
    print(f"[demosaic] Writing PNGs to {dst_dir}")
    print(f"[demosaic] Bands (R,G,B): {bands_rgb}")

    n_ok = 0
    n_skip = 0
    n_err = 0
    for img_path in images:
        out_png = dst_dir / f"{img_path.stem}.png"
        if out_png.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            cube = demosaic_ximea_5x5(img_path)
            rgb  = bands_to_rgb(cube, bands_rgb)  # H×W×3 uint8
            # OpenCV expects BGR on write
            ok = cv2.imwrite(str(out_png), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("cv2.imwrite returned False")
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[ERROR] {img_path.name}: {e}")

    print(f"[demosaic] Done. Wrote: {n_ok}, skipped: {n_skip}, errors: {n_err}")
    print(f"[demosaic] Output: {dst_dir.resolve()}")


if __name__ == "__main__":
    main()

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from mhfd_parser import parse_photo_database


def uyvy_to_rgb(image_data: bytes, width: int, height: int) -> Image.Image:
    """
    Converts raw UYVY image data (NTSC interlaced, all even fields then all odd fields)
    to RGB image using BT.601 conversion.

    Args:
        image_data (bytes): The raw UYVY byte stream from the ithmb file.
        width (int): The width of the image (e.g., 720 for 691200 size).
        height (int): The height of the image (e.g., 480 for 691200 size).
    """
    # --- Input Validation ---
    expected_size = width * height * 2
    if len(image_data) != expected_size:
        raise ValueError(f"Image data size mismatch. Expected {expected_size} bytes for {width}x{height} UYVY, "
                         f"but got {len(image_data)} bytes. This indicates a parsing error or corrupted data.")

    if width <= 0 or height <= 0 or width % 2 != 0 or height % 2 != 0:
        raise ValueError(
            f"Invalid dimensions: width ({width}) and height ({height}) must be positive and even for UYVY interlaced data.")

    # Interpret as interlaced: all even rows first, then all odd rows
    field_height = height // 2
    stride = width * 2

    # Split into even and odd fields
    even_field = np.frombuffer(image_data[:field_height * stride], dtype=np.uint8)
    odd_field = np.frombuffer(image_data[field_height * stride:], dtype=np.uint8)

    even_field = even_field.reshape(field_height, width, 2)
    odd_field = odd_field.reshape(field_height, width, 2)

    # Interleave rows back into full frame
    full = np.empty((height, width, 2), dtype=np.uint8)
    full[0::2, :, :] = even_field
    full[1::2, :, :] = odd_field

    # Convert UYVY → BGR → RGB
    bgr = cv2.cvtColor(full, cv2.COLOR_YUV2BGR_UYVY)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb)


def extract_single_image(args):
    """Worker function to extract and save a single image."""
    photos_dir, mhii, raw_dir, png_dir = args

    for mhod_title in mhii.children:
        mhni = mhod_title.children[0]
        mhod_data = mhni.children[0]

        # Use pathlib.Path for clean path manipulation
        thumb_filename_path = Path(mhod_data.body.content.replace(":", "/")[1:])
        thumb_file_path = photos_dir / thumb_filename_path

        output_filename = mhii.body.digitized_date.strftime(f"{mhii.body.id}_%Y-%m-%d_%H-%M-%S_{mhni.body.image_size}")

        png_output_path = (png_dir / output_filename).with_suffix(".png")

        try:
            with thumb_file_path.open("rb") as f:
                f.seek(mhni.body.ithmb_offset)
                image_data = f.read(mhni.body.image_size)

            if raw_dir is not None:
                raw_output_path = (raw_dir / output_filename).with_suffix(".raw")
                with raw_output_path.open("wb") as f2:
                    f2.write(image_data)

            if mhni.body.image_size == 691200:
                rgb_image = uyvy_to_rgb(image_data, 720, 480)

                # The imhex pattern indicates padding fields are present
                # Use these fields to correctly crop the image
                horizontal_padding = mhni.body.horizontal_padding
                vertical_padding = mhni.body.vertical_padding

                # Define the crop box (left, top, right, bottom)
                crop_box = (
                    horizontal_padding,
                    vertical_padding,
                    720 - horizontal_padding,
                    480 - vertical_padding
                )

                cropped_image = rgb_image.crop(crop_box)

                if png_output_path.exists():
                    tqdm.write(f"{png_output_path} already exists")

                cropped_image.save(png_output_path)
        except Exception as e:
            print(f"Failed to process image {output_filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract images from iPod photo database.")
    parser.add_argument("photos_dir", type=Path,
                        help="Path to the directory containing 'Photo Database' file and 'Thumbs' directory.")
    parser.add_argument("output_dir", type=Path, help="Path to output directory. PNGs will be written here.")
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Number of threads to use for parallel extraction.")
    parser.add_argument("-a", "--all", action="store_true", help="Extract all image data and store in raw formats.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = None
    if args.all:
        raw_dir = args.output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

    # Make the photo database path absolute
    photos_dir = args.photos_dir.resolve()
    db_path = photos_dir / "Photo Database"

    print(f"Parsing photo database at: {db_path}")
    root = parse_photo_database(db_path)

    for mhsd in root.children:
        if mhsd.body.index == 1:
            for mhli in mhsd.children:
                print(f"Discovered {mhli.body.num_images} images")

                # Prepare tasks for the thread pool
                tasks = []
                for mhii in mhli.children:
                    tasks.append((photos_dir, mhii, raw_dir, args.output_dir))

                # Execute tasks in parallel using a thread pool
                with ThreadPoolExecutor(max_workers=args.threads) as executor:
                    list(
                        tqdm(
                            executor.map(extract_single_image, tasks),
                            total=len(tasks),
                            desc="Extracting",
                            unit="image",
                        )
                    )


if __name__ == "__main__":
    main()

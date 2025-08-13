import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image

from mhfd_parser import parse_photo_database


def uyvy_to_rgb(image_data: bytes, width: int, height: int):
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

    # --- 1. Parse UYVY Data into Y, U, V Components ---
    raw_bytes = np.frombuffer(image_data, dtype=np.uint8)

    # Reshape into 4-byte UYVY quads:
    num_quads = len(raw_bytes) // 4
    uyvy_quads = raw_bytes.reshape(num_quads, 4)

    # Calculate number of quads per field.
    # The iPod's specific interlacing scheme means all even field data
    # is concatenated, followed by all odd field data.
    quads_per_field = (height // 2) * (width // 2)

    even_field_quads = uyvy_quads[:quads_per_field]
    odd_field_quads = uyvy_quads[quads_per_field:]

    # Extract U, Y0, V, Y1 for even field
    U_even = even_field_quads[:, 0]
    Y0_even = even_field_quads[:, 1]
    V_even = even_field_quads[:, 2]
    Y1_even = even_field_quads[:, 3]

    # Extract U, Y0, V, Y1 for odd field
    U_odd = odd_field_quads[:, 0]
    Y0_odd = odd_field_quads[:, 1]
    V_odd = odd_field_quads[:, 2]
    Y1_odd = odd_field_quads[:, 3]

    # Reshape Y components for each field to their half-height, full-width dimensions
    # Y0 and Y1 are alternating luminance values for adjacent pixels.
    Y_even_field = np.empty((height // 2, width), dtype=np.uint8)
    Y_even_field[:, 0::2] = Y0_even.reshape(height // 2, width // 2)  # Y0 for even columns
    Y_even_field[:, 1::2] = Y1_even.reshape(height // 2, width // 2)  # Y1 for odd columns

    Y_odd_field = np.empty((height // 2, width), dtype=np.uint8)
    Y_odd_field[:, 0::2] = Y0_odd.reshape(height // 2, width // 2)
    Y_odd_field[:, 1::2] = Y1_odd.reshape(height // 2, width // 2)

    # U and V components are already half-width due to 4:2:2 subsampling.
    # They are reshaped to their half-height, half-width dimensions.
    U_even_field = U_even.reshape(height // 2, width // 2)
    V_even_field = V_even.reshape(height // 2, width // 2)
    U_odd_field = U_odd.reshape(height // 2, width // 2)
    V_odd_field = V_odd.reshape(height // 2, width // 2)

    # --- 2. Implement De-interlacing Logic (Weaving) ---
    # Reconstruct Full Y Plane by interleaving even and odd lines.
    # The "all even fields, then all odd fields" structure means we interleave
    # row by row from the two separate field arrays.
    Y_full = np.empty((height, width), dtype=np.float32)  # Use float for calculations
    Y_full[0::2, :] = Y_even_field  # Place even lines (0, 2, 4,...)
    Y_full[1::2, :] = Y_odd_field  # Place odd lines (1, 3, 5,...)

    # Reconstruct Full U, V Planes (Upsampling Chroma).
    # Since U and V are horizontally subsampled (4:2:2), they need to be upsampled
    # to the full width before interleaving. Nearest neighbor repetition is simple and effective.
    U_even_upsampled = np.repeat(U_even_field, 2, axis=1)
    V_even_upsampled = np.repeat(V_even_field, 2, axis=1)
    U_odd_upsampled = np.repeat(U_odd_field, 2, axis=1)
    V_odd_upsampled = np.repeat(V_odd_field, 2, axis=1)

    U_full = np.empty((height, width), dtype=np.float32)
    U_full[0::2, :] = U_even_upsampled
    U_full[1::2, :] = U_odd_upsampled

    V_full = np.empty((height, width), dtype=np.float32)
    V_full[0::2, :] = V_even_upsampled
    V_full[1::2, :] = V_odd_upsampled

    # --- 3. Apply BT.601 YUV-to-RGB Conversion ---
    # Adjust for Limited Range: Subtract offsets from Y, U, V
    # Y values are typically 16-235, U/V values are typically 16-240 for BT.601 video levels.
    Y_norm = Y_full - 16
    U_norm = U_full - 128
    V_norm = V_full - 128

    # Apply BT.601 Conversion Formulas (using standard coefficients)
    R = 1.164 * Y_norm + 1.596 * V_norm
    G = 1.164 * Y_norm - 0.391 * U_norm - 0.813 * V_norm
    B = 1.164 * Y_norm + 2.018 * U_norm

    # Clamping: Clamp the resulting R, G, B values to the 0-255 range and convert to uint8.
    # This prevents pixel overflow/underflow and ensures valid pixel data for image display.
    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    # --- 4. Structure RGB Data and Save as PNG ---
    # Combine the R, G, B channels into a single 3D NumPy array (height, width, 3)
    rgb_image_array = np.stack((R, G, B), axis=-1)

    return Image.fromarray(rgb_image_array)


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

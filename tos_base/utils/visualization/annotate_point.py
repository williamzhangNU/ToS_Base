#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw points or text labels on a top-down image using mask coordinates from meta_data.json.
The mapping is read from the `topdown_map` field inside `meta_data.json` (no separate topdown_map.json required).

Usage:
    python annotate_point.py \
        --image /path/to/top_down.png \
        --map /path/to/meta_data.json \
        --labels '{"0,1": "A", "0,2": "B", "1,1": null}' \
        --out /path/to/top_down_marked.png
"""

import argparse
import json
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont


def load_mapping_from_meta(meta_path: Path) -> dict:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    topdown = data.get("topdown_map")
    if not topdown:
        raise ValueError("meta_data.json is missing the 'topdown_map' field")
    mapping = {}
    for item in topdown.get("mapping", []):
        r, c = item.get("row"), item.get("col")
        pixel = item.get("pixel", {})
        if r is None or c is None:
            continue
        if "x" not in pixel or "y" not in pixel:
            continue
        mapping[(r, c)] = (float(pixel["x"]), float(pixel["y"]))
    return mapping


def draw_point(img_path: Path, out_path: Path, mapping: dict, label_dict: dict, radius: int = 5):
    """
    Draw multiple annotations on the image.

    Args:
        img_path: Path to the input image
        out_path: Path where the output image will be saved
        mapping: Mapping from position to pixel coordinates, format {(row, col): (x, y)}
        label_dict: Mapping from position to label, format {(row, col): "A"} or {(row, col): None}
                    If the value is None, a red dot will be drawn instead of text.
        radius: Radius of the red dot
    """
    img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    for position, pixel in mapping.items():
        # Only annotate positions that are specified in label_dict
        if position not in label_dict:
            continue

        x, y = pixel
        text = label_dict[position]

        if text:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x - text_width / 2
            text_y = y - text_height / 2

            draw.text((text_x, text_y), text, fill=(255, 0, 0, 255), font=font)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0, 255))
    
    img.save(out_path)
    print(f"✅ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate a top-down image with dots or letters using mask coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate a single position with letter A
  python annotate_point.py --image top_down.png --map meta_data.json --labels '{"0,1": "A"}'

  # Annotate multiple positions
  python annotate_point.py --image top_down.png --map meta_data.json --labels '{"0,1": "A", "0,2": "B", "1,1": null}'

  # Use a file to specify labels
  python annotate_point.py --image top_down.png --map meta_data.json --labels-file labels.json
        """
    )
    parser.add_argument("--image", required=True, help="Path to top_down.png")
    parser.add_argument("--map", dest="map_path", required=True, help="Path to meta_data.json (must contain topdown_map)")
    parser.add_argument("--labels", type=str, help='Label dictionary JSON string, e.g. \'{"0,1": "A", "0,2": "B", "1,1": null}\'')
    parser.add_argument("--labels-file", type=str, help="Path to a JSON file containing the label dictionary")
    parser.add_argument("--out", default=None, help="Output path. Defaults to the same directory as the image with suffix _marked.png")
    args = parser.parse_args()

    image_path = Path(args.image)
    map_path = Path(args.map_path)

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"mapping not found: {map_path}")

    # Parse the labels dictionary
    if args.labels_file:
        label_data = json.loads(Path(args.labels_file).read_text(encoding="utf-8"))
    elif args.labels:
        label_data = json.loads(args.labels)
    else:
        raise ValueError("You must specify either --labels or --labels-file")
    
    # Convert string keys to tuple keys
    label_dict = {}
    for key, value in label_data.items():
        parts = key.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid position key: {key}. Expected 'row,col' format")
        row, col = int(parts[0]), int(parts[1])
        label_dict[(row, col)] = value

    mapping = load_mapping_from_meta(map_path)
    
    # Verify all label positions exist in the mapping
    for position in label_dict.keys():
        if position not in mapping:
            raise ValueError(f"coordinate {position} not found in mapping")

    out_path = Path(args.out) if args.out else image_path.with_name(image_path.stem + "_marked.png")
    draw_point(image_path, out_path, mapping, label_dict)


if __name__ == "__main__":
    main()


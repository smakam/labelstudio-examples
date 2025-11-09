import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from requests import Response

""" To run this:
python3 csv_bound.py datasets/sample_images.csv \
  --output-json images-bounding-box-label.json \
  --image-column image_url \
  --from-name bbox \
  --to-name image \
  --label-field image \
  --min-confidence 60 \
  --log-level INFO """

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read image URLs from a CSV, run AWS Rekognition, and write Label Studio prediction "
            "JSON (tasks) that can be imported for pre-annotation."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV file containing an image URL column.",
    )
    parser.add_argument(
        "-o",
        "--output-json",
        type=Path,
        default=Path("tasks.json"),
        help="Destination JSON file for Label Studio tasks (default: tasks.json).",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="image_url",
        help="Column name in the input CSV that contains image URLs (default: image_url).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=50.0,
        help="Minimum confidence threshold passed to Rekognition (default: 50.0).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS region for Rekognition client (default: use AWS SDK resolution).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds for downloading each image (default: 15s).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for script output (default: INFO).",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Create a task entry even if Rekognition found no boxes (default: skip empties).",
    )
    parser.add_argument(
        "--from-name",
        type=str,
        default="bbox",
        help="Label Studio control tag name for rectanglelabels (default: bbox).",
    )
    parser.add_argument(
        "--to-name",
        type=str,
        default="image",
        help="Label Studio object tag name that displays the image (default: image).",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default="image",
        help="Field name stored under data{} that points Label Studio to the image (default: image).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="rekognition_v1",
        help="Model version string saved alongside the predictions (default: rekognition_v1).",
    )
    return parser.parse_args()


def normalize_bbox(bbox: Dict[str, float]) -> Dict[str, float]:
    """Convert Rekognition bounding box dimensions into percentages for Label Studio."""
    return {
        "x": bbox.get("Left", 0.0) * 100.0,
        "y": bbox.get("Top", 0.0) * 100.0,
        "width": bbox.get("Width", 0.0) * 100.0,
        "height": bbox.get("Height", 0.0) * 100.0,
    }


def detect_bounding_boxes(
    client,
    image_bytes: bytes,
    min_confidence: float,
) -> List[Dict]:
    response = client.detect_labels(
        Image={"Bytes": image_bytes},
        MinConfidence=min_confidence,
    )
    detections: List[Dict] = []
    for label in response.get("Labels", []):
        label_name = label.get("Name")
        for instance in label.get("Instances", []):
            bbox = instance.get("BoundingBox")
            if not bbox:
                continue
            normalized = normalize_bbox(bbox)
            detections.append(
                {
                    "label": label_name or "",
                    "score": instance.get("Confidence", label.get("Confidence")),
                    "rotation": 0,
                    **normalized,
                }
            )
    return detections


def fetch_image(url: str, timeout: float) -> bytes:
    try:
        response: Response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    return response.content


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    input_path = args.input_csv.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input CSV {input_path} does not exist.")

    session_kwargs: Dict[str, str] = {}
    if args.region:
        session_kwargs["region_name"] = args.region

    try:
        rekognition = boto3.client("rekognition", **session_kwargs)
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"Failed to create Rekognition client: {exc}") from exc

    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if args.image_column not in reader.fieldnames:
            raise SystemExit(
                f"Column '{args.image_column}' not found in {input_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        tasks: List[Dict] = []

        for idx, row in enumerate(reader, start=1):
            url = (row.get(args.image_column) or "").strip()
            if not url:
                logging.warning("Row %d has empty image URL; skipping.", idx)
                continue

            logging.info("Processing row %d: %s", idx, url)
            try:
                image_bytes = fetch_image(url, args.request_timeout)
            except RuntimeError as exc:
                logging.error("%s", exc)
                continue

            try:
                detections = detect_bounding_boxes(
                    rekognition,
                    image_bytes=image_bytes,
                    min_confidence=args.min_confidence,
                )
            except (BotoCoreError, ClientError) as exc:
                logging.error("Rekognition error for %s: %s", url, exc)
                continue

            if not detections and not args.include_empty:
                logging.debug("No detections for %s; skipping row.", url)
                continue

            task: Dict[str, Dict] = {"data": {args.label_field: url}}

            if detections or args.include_empty:
                results = [
                    {
                        "from_name": args.from_name,
                        "to_name": args.to_name,
                        "type": "rectanglelabels",
                        "value": {
                            "x": det["x"],
                            "y": det["y"],
                            "width": det["width"],
                            "height": det["height"],
                            "rotation": det["rotation"],
                            "rectanglelabels": [det["label"]] if det["label"] else [],
                        },
                        "score": det.get("score"),
                    }
                    for det in detections
                ]
                task["predictions"] = [
                    {
                        "model_version": args.model_version,
                        "result": results,
                    }
                ]
            tasks.append(task)

    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2))
    logging.info("Finished. Wrote %d tasks to %s", len(tasks), output_path)


if __name__ == "__main__":
    main()

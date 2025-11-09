import argparse
import csv
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError

ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe Bengali audio with AssemblyAI, translate segments with AWS Translate, "
            "and export Label Studio-ready tasks."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("datasets/audio/bengali-audio-samples.csv"),
        help="CSV file containing audio URLs (default: datasets/audio/bengali-audio-samples.csv).",
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default="audio",
        help="Name of the CSV column that stores audio URLs (default: audio).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bengali_audio_tasks.json"),
        help="Destination for the Label Studio JSON export (default: results/bengali_audio_tasks.json).",
    )
    parser.add_argument(
        "--language-code",
        type=str,
        default="bn",
        help="Language code passed to AssemblyAI for transcription (default: bn).",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default="en",
        help="Target language for AWS Translate (default: en).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between polling AssemblyAI for completion (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Maximum seconds to wait for each transcription (default: 1800).",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default=None,
        help="AWS region for the Translate client (default: use AWS SDK config).",
    )
    parser.add_argument(
        "--annotation-user-id",
        type=int,
        default=1,
        help="Value stored in completed_by for pre-filled annotations (default: 1).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity of script logging (default: INFO).",
    )
    return parser.parse_args()


def read_audio_urls(csv_path: Path, column: str) -> List[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if column not in (reader.fieldnames or []):
            raise SystemExit(
                f"Column '{column}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )
        return [
            (row.get(column) or "").strip()
            for row in reader
            if (row.get(column) or "").strip()
        ]


def assemblyai_headers(api_key: str) -> Dict[str, str]:
    return {
        "authorization": api_key,
        "content-type": "application/json",
    }


def submit_transcription(
    session: requests.Session,
    headers: Dict[str, str],
    audio_url: str,
    language_code: str,
) -> str:
    payload = {
        "audio_url": audio_url,
        "language_code": language_code,
        "speaker_labels": True,
        "auto_chapters": False,
        "auto_highlights": False,
    }
    response = session.post(f"{ASSEMBLYAI_BASE_URL}/transcript", headers=headers, json=payload)
    response.raise_for_status()
    transcript_id = response.json().get("id")
    if not transcript_id:
        raise RuntimeError("AssemblyAI did not return a transcript id.")
    return transcript_id


def poll_transcription(
    session: requests.Session,
    headers: Dict[str, str],
    transcript_id: str,
    poll_interval: float,
    timeout: float,
) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while True:
        response = session.get(f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}", headers=headers)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "completed":
            return payload
        if status == "error":
            raise RuntimeError(f"Transcription failed: {payload.get('error')}")
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for transcript {transcript_id}")
        time.sleep(poll_interval)


def translate_text(
    translate_client,
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    if not text:
        return ""
    response = translate_client.translate_text(
        Text=text,
        SourceLanguageCode=source_language,
        TargetLanguageCode=target_language,
    )
    return response.get("TranslatedText", "")


def ms_to_seconds(value: Optional[float]) -> float:
    return round(float(value or 0.0) / 1000.0, 6)


def build_annotation_results(
    transcription: Dict[str, Any],
    translate_client,
    source_language: str,
    target_language: str,
) -> List[Dict[str, Any]]:
    utterances = transcription.get("utterances") or []
    results: List[Dict[str, Any]] = []

    if not utterances and transcription.get("text"):
        utterances = [
            {
                "speaker": "A",
                "text": transcription["text"],
                "start": 0,
                "end": transcription.get("audio_duration", 0),
            }
        ]

    for utter in utterances:
        start = ms_to_seconds(utter.get("start"))
        end = ms_to_seconds(utter.get("end"))
        speaker = (utter.get("speaker") or "Other").strip() or "Other"
        bn_text = (utter.get("text") or "").strip()
        try:
            en_text = translate_text(
                translate_client,
                text=bn_text,
                source_language=source_language,
                target_language=target_language,
            )
        except (BotoCoreError, ClientError) as exc:
            logging.error("Translation failed for segment (%s-%s): %s", start, end, exc)
            en_text = ""

        region_id = uuid.uuid4().hex
        results.append(
            {
                "id": region_id,
                "from_name": "speaker",
                "to_name": "audio",
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "labels": [speaker],
                    "channel": 0,
                },
            }
        )

        if bn_text:
            results.append(
                {
                    "id": uuid.uuid4().hex,
                    "from_name": "bn_transcription",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": [bn_text],
                        "region_id": region_id,
                    },
                }
            )

        if en_text:
            results.append(
                {
                    "id": uuid.uuid4().hex,
                    "from_name": "en_translation",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": [en_text],
                        "region_id": region_id,
                    },
                }
            )
    return results

def generate_tasks(
    audio_urls: List[str],
    session: requests.Session,
    headers: Dict[str, str],
    translate_client,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []

    for audio_url in audio_urls:
        logging.info("Submitting AssemblyAI transcription for %s", audio_url)
        try:
            transcript_id = submit_transcription(
                session=session,
                headers=headers,
                audio_url=audio_url,
                language_code=args.language_code,
            )
            transcription = poll_transcription(
                session=session,
                headers=headers,
                transcript_id=transcript_id,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            )
        except (requests.RequestException, RuntimeError, TimeoutError) as exc:
            logging.error("Failed to transcribe %s: %s", audio_url, exc)
            continue

        results = build_annotation_results(
            transcription=transcription,
            translate_client=translate_client,
            source_language=args.language_code,
            target_language=args.target_language,
        )

        predictions = []
        if results:
            predictions.append(
                {
                    "model_version": "assemblyai-aws-translate-v1",
                    "score": transcription.get("confidence", 0.0),
                    "result": results,
                }
            )

        tasks.append(
            {
                "data": {"audio": audio_url},
                "predictions": predictions,
                "meta": {
                    "assemblyai_transcript_id": transcription.get("id"),
                    "audio_duration": transcription.get("audio_duration"),
                },
            }
        )

    return tasks


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise SystemExit("Set the ASSEMBLYAI_API_KEY environment variable.")

    audio_urls = read_audio_urls(args.input_csv, args.audio_column)
    if not audio_urls:
        logging.warning("No audio URLs found in %s", args.input_csv)

    session = requests.Session()
    headers = assemblyai_headers(api_key)

    translate_kwargs: Dict[str, Any] = {}
    if args.aws_region:
        translate_kwargs["region_name"] = args.aws_region
    try:
        translate_client = boto3.client("translate", **translate_kwargs)
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"Failed to create AWS Translate client: {exc}") from exc

    tasks = generate_tasks(
        audio_urls=audio_urls,
        session=session,
        headers=headers,
        translate_client=translate_client,
        args=args,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False))
    logging.info("Wrote %d tasks to %s", len(tasks), output_path)


if __name__ == "__main__":
    main()

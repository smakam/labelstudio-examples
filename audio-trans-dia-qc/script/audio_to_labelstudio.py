import argparse
import csv
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Label Studio tasks from AssemblyAI transcriptions for the audio files "
            "listed in a CSV."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV file containing a column with audio URLs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("audio_tasks.json"),
        help="Path to write the Label Studio tasks JSON (default: audio_tasks.json).",
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default="audio",
        help="Name of the column in the CSV that stores audio URLs (default: audio).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between polling AssemblyAI for transcription completion.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Maximum seconds to wait for a transcription before giving up (default: 15 minutes).",
    )
    parser.add_argument(
        "--transcription-name",
        type=str,
        default="transcription",
        help="Label Studio control tag name for transcript text (default: transcription).",
    )
    parser.add_argument(
        "--speaker-name",
        type=str,
        default="speaker",
        help="Label Studio control tag name for speakers (default: speaker).",
    )
    parser.add_argument(
        "--sentiment-name",
        type=str,
        default="sentiment",
        help="Label Studio control tag name for sentiment choices (default: sentiment).",
    )
    parser.add_argument(
        "--to-name",
        type=str,
        default="audio",
        help="Label Studio object tag name that references the audio (default: audio).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="assemblyai-v2",
        help="Model version string stored in Label Studio predictions (default: assemblyai-v2).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for script output (default: INFO).",
    )
    return parser.parse_args()


def read_audio_rows(csv_path: Path, column: str) -> List[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if column not in reader.fieldnames:
            raise SystemExit(
                f"Column '{column}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )
        return [
            (row.get(column) or "").strip()
            for row in reader
            if (row.get(column) or "").strip()
        ]


def request_headers(api_key: str) -> Dict[str, str]:
    return {
        "authorization": api_key,
        "content-type": "application/json",
    }


def submit_transcription(
    session: requests.Session,
    audio_url: str,
    headers: Dict[str, str],
) -> str:
    payload = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "sentiment_analysis": True,
        "iab_categories": True,
        "entity_detection": True,
        "auto_chapters": True,
        "auto_highlights": True,
    }
    response = session.post(f"{ASSEMBLYAI_BASE_URL}/transcript", headers=headers, json=payload)
    response.raise_for_status()
    transcript_id = response.json().get("id")
    if not transcript_id:
        raise RuntimeError(f"AssemblyAI did not return a transcript id for {audio_url}.")
    return transcript_id


def poll_transcription(
    session: requests.Session,
    transcript_id: str,
    headers: Dict[str, str],
    poll_interval: float,
    timeout: float,
) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while True:
        response = session.get(
            f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}",
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "completed":
            return payload
        if status == "error":
            raise RuntimeError(f"Transcription {transcript_id} failed: {payload.get('error')}")
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for transcription {transcript_id} to complete.")
        time.sleep(poll_interval)


def ms_to_seconds(value: Optional[float]) -> float:
    return round(float(value or 0.0) / 1000.0, 6)


def find_sentiment_for_span(
    sentiments: Sequence[Dict[str, Any]],
    start_ms: Optional[float],
    end_ms: Optional[float],
) -> Optional[Dict[str, Any]]:
    def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        return min(a_end, b_end) - max(a_start, b_start)

    if not sentiments:
        return None
    start = float(start_ms or 0.0)
    end = float(end_ms or start)
    best_entry = None
    best_overlap = -1.0
    for entry in sentiments:
        entry_start = float(entry.get("start") or 0.0)
        entry_end = float(entry.get("end") or entry_start)
        ov = overlap(start, end, entry_start, entry_end)
        if ov >= 0 and ov > best_overlap:
            best_entry = entry
            best_overlap = ov
    return best_entry


def build_prediction(
    audio_url: str,
    transcription: Dict[str, Any],
    transcription_name: str,
    speaker_name: str,
    sentiment_name: str,
    to_name: str,
    model_version: str,
) -> Dict[str, Any]:
    utterances = transcription.get("utterances") or []
    sentiments = transcription.get("sentiment_analysis_results") or []

    metadata = {
        "confidence": transcription.get("confidence"),
        "audio_duration": transcription.get("audio_duration"),
        "chapters": transcription.get("chapters"),
        "highlights": transcription.get("auto_highlights_result"),
        "entities": transcription.get("entities"),
        "sentiment_analysis": sentiments,
        "iab_categories": transcription.get("iab_categories_result"),
        "utterances": utterances,
    }
    filtered_metadata = {k: v for k, v in metadata.items() if v}

    result: List[Dict[str, Any]] = []

    for utter in utterances:
        start_ms = utter.get("start")
        end_ms = utter.get("end")
        start = ms_to_seconds(start_ms)
        end = ms_to_seconds(end_ms)
        speaker_label = (utter.get("speaker") or "Other").strip() or "Other"
        text_value = utter.get("text") or ""
        confidence = utter.get("confidence")
        region_id = uuid.uuid4().hex

        result.append(
            {
                "id": region_id,
                "from_name": speaker_name,
                "to_name": to_name,
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "labels": [speaker_label],
                    "channel": 0,
                },
                "score": confidence,
            }
        )

        if text_value:
            result.append(
                {
                    "from_name": transcription_name,
                    "to_name": to_name,
                    "type": "textarea",
                    "value": {
                        "text": [text_value],
                        "start": start,
                        "end": end,
                        "channel": 0,
                        "region_id": region_id,
                    },
                    "score": confidence,
                }
            )

        sentiment_entry = find_sentiment_for_span(sentiments, start_ms, end_ms)
        if sentiment_entry:
            sentiment_label = sentiment_entry.get("sentiment")
            if sentiment_label:
                result.append(
                    {
                        "from_name": sentiment_name,
                        "to_name": to_name,
                        "type": "choices",
                        "value": {
                            "choices": [sentiment_label],
                            "start": start,
                            "end": end,
                            "channel": 0,
                            "region_id": region_id,
                        },
                        "score": sentiment_entry.get("confidence"),
                    }
                )

    # Fallback to full-text transcript if no utterance-level data was returned.
    if not result:
        text_value = transcription.get("text")
        if text_value:
            result.append(
                {
                    "from_name": transcription_name,
                    "to_name": to_name,
                    "type": "textarea",
                    "value": {"text": [text_value]},
                    "score": transcription.get("confidence"),
                }
            )

    return {
        "model_version": model_version,
        "result": result,
        "score": transcription.get("confidence"),
        "extra_metadata": filtered_metadata,
        "audio_url": audio_url,
    }


def generate_tasks(
    audio_urls: List[str],
    session: requests.Session,
    headers: Dict[str, str],
    poll_interval: float,
    timeout: float,
    transcription_name: str,
    speaker_name: str,
    sentiment_name: str,
    to_name: str,
    model_version: str,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for audio_url in audio_urls:
        logging.info("Submitting AssemblyAI transcription for %s", audio_url)
        try:
            transcript_id = submit_transcription(session, audio_url, headers)
            logging.debug("Transcript id %s for %s", transcript_id, audio_url)
            transcription = poll_transcription(
                session,
                transcript_id,
                headers=headers,
                poll_interval=poll_interval,
                timeout=timeout,
            )
        except (requests.RequestException, RuntimeError, TimeoutError) as exc:
            logging.error("Failed to transcribe %s: %s", audio_url, exc)
            continue

        prediction = build_prediction(
            audio_url=audio_url,
            transcription=transcription,
            transcription_name=transcription_name,
            speaker_name=speaker_name,
            sentiment_name=sentiment_name,
            to_name=to_name,
            model_version=model_version,
        )

        task = {
            "data": {to_name: audio_url},
            "predictions": [prediction] if prediction["result"] else [],
            "assemblyai_raw": transcription,
        }
        tasks.append(task)
    return tasks


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise SystemExit("ASSEMBLYAI_API_KEY environment variable is required.")

    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.is_file():
        raise SystemExit(f"Input CSV {input_csv} does not exist.")

    audio_urls = read_audio_rows(input_csv, args.audio_column)
    if not audio_urls:
        logging.warning("No audio URLs found in %s", input_csv)

    session = requests.Session()
    headers = request_headers(api_key)

    tasks = generate_tasks(
        audio_urls=audio_urls,
        session=session,
        headers=headers,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        transcription_name=args.transcription_name,
        speaker_name=args.speaker_name,
        sentiment_name=args.sentiment_name,
        to_name=args.to_name,
        model_version=args.model_version,
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2))

    logging.info("Generated %d Label Studio tasks at %s", len(tasks), output_path)


if __name__ == "__main__":
    main()

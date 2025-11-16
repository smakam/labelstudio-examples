import os
import logging
from typing import List, Dict, Any

import assemblyai as aai
from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)

# Configure AssemblyAI
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")


class AssemblyAIBengaliASR(LabelStudioMLBase):
    """
    Label Studio ML backend that uses AssemblyAI to pre-annotate Bengali audio
    with a transcription.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Optional: you can pre-configure transcription options here
        self.transcriber = aai.Transcriber()
        self.config = aai.TranscriptionConfig(
            # For strict Bengali, you can set language_code="bn"
            language_code="bn",   # or use language_detection=True
            punctuate=True,
            format_text=True,
        )
        logger.info("AssemblyAIBengaliASR backend initialized")

    def predict(self, tasks: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        For each task, call AssemblyAI and return a prefilled transcript.
        """
        predictions = []

        for task in tasks:
            audio_source = task["data"].get("audio")
            if not audio_source:
                logger.warning("No 'audio' field in task data")
                predictions.append({"result": []})
                continue

            try:
                transcript_text = self._transcribe(audio_source)
            except Exception as e:
                logger.exception(f"AssemblyAI transcription failed for {audio_source}: {e}")
                # Return empty prediction so annotator still sees the task
                predictions.append({"result": []})
                continue

            # IMPORTANT: from_name/to_name/type must match your labeling config
            result = [
                {
                    "from_name": "transcription",  # TextArea name
                    "to_name": "audio",            # Audio name
                    "type": "textarea",
                    "value": {
                        "text": [transcript_text]
                    },
                }
            ]

            predictions.append(
                {
                    "result": result,
                    # Optional: some "confidence" score if you want active learning
                    "score": 0.5,
                }
            )

        return predictions

    def _transcribe(self, audio_source: str) -> str:
        """
        Call AssemblyAI on a URL or local path.
        AssemblyAI SDK can handle both remote and local files.
        """
        logger.info(f"Submitting audio to AssemblyAI: {audio_source}")
        transcript = self.transcriber.transcribe(audio_source, config=self.config)

        if transcript.error:
            raise RuntimeError(transcript.error)

        return transcript.text or ""

"""
Audio transcription functionality using OpenAI Whisper.
"""

import wave
import whisper
import numpy as np
from typing import Dict


class AudioTranscriber:
    """Handle audio transcription using Whisper."""

    def __init__(self, model_name: str = "base"):
        """
        Initialize transcriber with a Whisper model.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            self.model = whisper.load_model(self.model_name)

    def transcribe(self, audio_input, language: str = "en",
                   **kwargs) -> Dict:
        """
        Transcribe audio to text.

        Args:
            audio_input: Can be filepath (str) or numpy array
            language: Language code (default: "en")
            **kwargs: Additional arguments for whisper.transcribe()

        Returns:
            Dictionary with transcription results
        """
        self.load_model()
        
        # Handle different input types
        if isinstance(audio_input, str):
            # Pass file path directly to Whisper - it handles all preprocessing
            result = self.model.transcribe(audio_input, language=language, **kwargs)
        elif isinstance(audio_input, np.ndarray):
            # Ensure audio data is float32
            audio = audio_input.astype(np.float32)
            result = self.model.transcribe(audio, language=language, **kwargs)
        elif isinstance(audio_input, bytes):
            # Convert bytes to numpy array
            audio = np.frombuffer(
                audio_input, dtype=np.int16
            ).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio, language=language, **kwargs)
        else:
            raise ValueError(
                f"Unsupported audio input type: {type(audio_input)}"
            )
        
        return result
    
    def get_segments(self, result: Dict) -> list:
        """
        Extract timestamped segments from transcription result.

        Args:
            result: Whisper transcription result

        Returns:
            List of segments with start, end, and text
        """
        if "segments" not in result:
            return []

        return [
            {
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip()
            }
            for seg in result['segments']
        ]

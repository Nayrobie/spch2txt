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
            # Load from file
            audio = self._load_wav_file(audio_input)
        elif isinstance(audio_input, np.ndarray):
            # Ensure audio data is float32
            audio = audio_input.astype(np.float32)
        elif isinstance(audio_input, bytes):
            # Convert bytes to numpy array
            audio = np.frombuffer(
                audio_input, dtype=np.int16
            ).astype(np.float32) / 32768.0
        else:
            raise ValueError(
                f"Unsupported audio input type: {type(audio_input)}"
            )
        # Transcribe

        audio = audio.astype(np.float32)
        result = self.model.transcribe(audio, language=language, **kwargs)
        return result
    
    def _load_wav_file(self, filepath: str) -> np.ndarray:
        """
        Load a WAV file and convert to format expected by Whisper.

        Args:
            filepath: Path to WAV file

        Returns:
            Audio as numpy array (float32, normalized to [-1, 1])
        """
        with wave.open(filepath, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            audio_data = wf.readframes(n_frames)

            if sampwidth == 2:
                audio = np.frombuffer(audio_data, dtype=np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")
            
            # Convert to float32 and normalize
            audio = audio.astype(np.float32) / 32768.0
            
            # Convert stereo to mono if needed
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            
            # Resample to 16kHz if needed
            if framerate != 16000:
                duration = len(audio) / framerate
                target_length = int(duration * 16000)
                audio = np.interp(
                    np.linspace(0, len(audio), target_length,
                                dtype=np.float32),
                    np.arange(len(audio), dtype=np.float32),
                    audio
                ).astype(np.float32)

            return audio.astype(np.float32)
    
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

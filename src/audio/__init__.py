"""
Audio capture and processing module for speech-to-text transcription.

This package provides a complete solution for recording audio from multiple devices,
transcribing speech to text using OpenAI Whisper, and identifying speakers using
pyannote.audio diarization.

Main components:
- AudioCapture: Multi-device audio recording with WASAPI loopback support
- AudioTranscriber: Speech-to-text using Whisper with hallucination filtering
- PyannoteDiarizer: Speaker identification and diarization
- Utility functions: Audio processing, device categorization, and file operations

Example usage:
    from src.audio import AudioCapture, AudioTranscriber
    
    # Record audio
    capture = AudioCapture()
    devices = capture.list_devices()
    audio_files = capture.record_multi_device(...)
    
    # Transcribe
    transcriber = AudioTranscriber(model_name="base")
    result = transcriber.transcribe_multiple(audio_files, device_names)
"""

from .capture import AudioCapture
from .transcription import AudioTranscriber
from .diarization import PyannoteDiarizer, assign_speakers_to_segments
from .summarizer import MeetingSummarizer, summarize_transcript
from .utils import (
    categorize_devices,
    format_timestamp,
    get_audio_duration,
    get_audio_level,
    mix_wav_files,
    normalize_audio,
    save_audio_array
)

__all__ = [
    'AudioCapture',
    'AudioTranscriber',
    'PyannoteDiarizer',
    'assign_speakers_to_segments',
    'MeetingSummarizer',
    'summarize_transcript',
    'categorize_devices',
    'format_timestamp',
    'get_audio_duration',
    'get_audio_level',
    'mix_wav_files',
    'normalize_audio',
    'save_audio_array'
]

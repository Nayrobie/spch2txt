"""
Audio capture and processing module for speech-to-text transcription.
"""

from .capture import AudioCapture
from .transcription import AudioTranscriber
from .diarization import PyannoteDiarizer, assign_speakers_to_segments
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
    'categorize_devices',
    'format_timestamp',
    'get_audio_duration',
    'get_audio_level',
    'mix_wav_files',
    'normalize_audio',
    'save_audio_array'
]

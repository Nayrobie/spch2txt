"""
Audio capture and processing module for speech-to-text transcription.
"""

from .capture import AudioCapture
from .transcription import AudioTranscriber

__all__ = ['AudioCapture', 'AudioTranscriber']

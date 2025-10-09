"""
Utility functions for audio processing.
"""

import os
import wave
import numpy as np
from typing import Dict, List


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as MM:SS or HH:MM:SS.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def categorize_devices(devices: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize audio devices by type.
    
    Args:
        devices: List of device information dictionaries
        
    Returns:
        Dictionary with 'input', 'output', and 'loopback' device lists
    """
    categorized = {
        'input': [],
        'output': [],
        'loopback': []
    }
    
    for device in devices:
        if device.get('isLoopback', False):
            categorized['loopback'].append(device)
        elif device['maxInputChannels'] > 0:
            categorized['input'].append(device)
        elif device['maxOutputChannels'] > 0:
            categorized['output'].append(device)
    
    return categorized


def get_audio_duration(filepath: str) -> float:
    """
    Get duration of a WAV file in seconds.
    
    Args:
        filepath: Path to WAV file
        
    Returns:
        Duration in seconds
    """
    with wave.open(filepath, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.

    Args:
        audio: Audio array

    Returns:
        Normalized audio array
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def mix_wav_files(filepaths: List[str],
                  target_rate: int = 48000) -> np.ndarray:
    """
    Mix multiple WAV files into a single audio stream.

    Args:
        filepaths: List of WAV file paths
        target_rate: Target sample rate for output

    Returns:
        Mixed audio as numpy array (float32, normalized)
    """
    audio_data = []

    for filepath in filepaths:
        filepath = os.path.normpath(filepath)

        if not os.path.exists(filepath):
            continue

        with wave.open(filepath, 'rb') as wf:
            n_channels = wf.getnchannels()
            rate = wf.getframerate()
            n_frames = wf.getnframes()

            audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)

            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            audio = audio.astype(np.float32) / 32768.0

            if rate != target_rate:
                duration = len(audio) / rate
                target_length = int(duration * target_rate)
                audio = np.interp(
                    np.linspace(0, len(audio), target_length,
                                dtype=np.float32),
                    np.arange(len(audio), dtype=np.float32),
                    audio
                )

            audio_data.append(audio)

    if not audio_data:
        return np.zeros(target_rate, dtype=np.float32)

    max_length = max(len(audio) for audio in audio_data)

    padded_audio = []
    for audio in audio_data:
        if len(audio) < max_length:
            padding = np.zeros(max_length - len(audio), dtype=np.float32)
            padded_audio.append(np.concatenate([audio, padding]))
        else:
            padded_audio.append(audio)

    mixed_audio = np.zeros(max_length, dtype=np.float32)
    for audio in padded_audio:
        mixed_audio += audio
    mixed_audio /= len(audio_data)

    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio = mixed_audio / max_val

    return mixed_audio


def save_audio_array(audio: np.ndarray, filepath: str,
                     rate: int = 48000, channels: int = 1):
    """
    Save audio array to WAV file.

    Args:
        audio: Audio array (float32, normalized to [-1, 1])
        filepath: Output file path
        rate: Sample rate
        channels: Number of channels
    """
    audio_int = (audio * 32767).astype(np.int16)

    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio_int.tobytes())


def get_audio_level(audio: np.ndarray) -> float:
    """
    Calculate RMS audio level.

    Args:
        audio: Audio array

    Returns:
        RMS level as float
    """
    return float(np.sqrt(np.mean(audio ** 2)))

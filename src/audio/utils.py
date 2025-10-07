"""
Utility functions for audio processing.
"""

import numpy as np
import wave
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

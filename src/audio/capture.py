"""
Audio capture functionality using pyaudiowpatch for Windows WASAPI support.
"""

import wave
import pyaudiowpatch as pyaudio
from typing import Optional, Dict, List


class AudioCapture:
    """Handle audio recording from various input devices."""
    
    def __init__(self, rate: int = 16000, channels: int = 1, frames_per_buffer: int = 1024):
        """
        Initialize audio capture.
        
        Args:
            rate: Sample rate in Hz (default 16000 for Whisper)
            channels: Number of audio channels (1=mono, 2=stereo)
            frames_per_buffer: Buffer size for audio chunks
        """
        self.rate = rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.pa = None
        self.stream = None
        
    def list_devices(self) -> List[Dict]:
        """
        List all available audio devices.
        
        Returns:
            List of device information dictionaries
        """
        pa = pyaudio.PyAudio()
        devices = []
        
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': info['name'],
                'hostApi': info['hostApi'],
                'maxInputChannels': info['maxInputChannels'],
                'maxOutputChannels': info['maxOutputChannels'],
                'defaultSampleRate': info['defaultSampleRate'],
                'isLoopback': 'loopback' in info['name'].lower()
            })
        
        pa.terminate()
        return devices
    
    def get_default_input_device(self) -> Dict:
        """Get the default input device information."""
        pa = pyaudio.PyAudio()
        device_info = pa.get_default_input_device_info()
        pa.terminate()
        return device_info
    
    def record(self, duration: int, device_index: Optional[int] = None, 
               output_file: Optional[str] = None) -> bytes:
        """
        Record audio for a specified duration.
        
        Args:
            duration: Recording duration in seconds
            device_index: Audio device index (None for default)
            output_file: Optional WAV file path to save recording
            
        Returns:
            Raw audio data as bytes
        """
        self.pa = pyaudio.PyAudio()
        
        # Get device
        if device_index is None:
            device_info = self.pa.get_default_input_device_info()
            device_index = device_info['index']
        
        # Open stream
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
            input_device_index=device_index
        )
        
        # Record
        num_chunks = int(self.rate / self.frames_per_buffer * duration)
        frames = []
        
        for _ in range(num_chunks):
            data = self.stream.read(self.frames_per_buffer)
            frames.append(data)
        
        # Cleanup
        self.stream.close()
        self.pa.terminate()
        
        audio_data = b"".join(frames)
        
        # Save to file if requested
        if output_file:
            self._save_wav(audio_data, output_file)
        
        return audio_data
    
    def _save_wav(self, audio_data: bytes, filepath: str):
        """Save audio data to WAV file."""
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.rate)
            wf.writeframes(audio_data)
    
    def cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.close()
        if self.pa:
            self.pa.terminate()

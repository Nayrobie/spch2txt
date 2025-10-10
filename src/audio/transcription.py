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
    
    def transcribe_multiple(self, audio_files: list, device_names: list) -> Dict:
        """
        Transcribe multiple audio files separately and combine results.
        Uses timestamps to interleave segments in chronological order.
        
        Args:
            audio_files: List of audio file paths
            device_names: List of device names corresponding to audio files
            
        Returns:
            Dictionary with separate and combined transcripts
        """
        import os
        
        transcripts = []
        all_segments = []
        
        for i, (audio_file, device_name) in enumerate(zip(audio_files, device_names)):
            print(f"\nTranscribing device {i+1}: {device_name}")
            print(f"File: {os.path.basename(audio_file)}")
            
            try:
                result = self.transcribe(audio_file, verbose=False)
                text = result["text"].strip()
                
                # Determine speaker label
                is_loopback = 'loopback' in device_name.lower()
                speaker_label = "System Audio" if is_loopback else "Microphone"
                
                if text:
                    transcripts.append({
                        "device": device_name,
                        "speaker": speaker_label,
                        "text": text,
                        "language": result.get("language", "unknown"),
                        "audio_file": os.path.basename(audio_file)
                    })
                    
                    print(f"✓ Transcribed ({len(text)} chars)")
                    
                    # Extract segments with timestamps
                    if "segments" in result:
                        for segment in result["segments"]:
                            segment_text = segment["text"].strip()
                            if segment_text:
                                all_segments.append({
                                    "start": segment["start"],
                                    "end": segment["end"],
                                    "text": segment_text,
                                    "speaker": speaker_label
                                })
                else:
                    print("⚠ No speech detected")
                    
            except Exception as e:
                print(f"❌ Error transcribing {device_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Sort segments by start time to get chronological order
        all_segments.sort(key=lambda x: x["start"])
        
        # Build combined text with timestamps
        combined_lines = []
        for segment in all_segments:
            # Format timestamp as MM:SS
            minutes = int(segment["start"] // 60)
            seconds = int(segment["start"] % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            
            combined_lines.append(f"[{timestamp}] [{segment['speaker']}]: {segment['text']}")
        
        combined_text = "\n\n".join(combined_lines) if combined_lines else "(No speech detected)"
        
        return {
            "transcripts": transcripts,
            "combined_text": combined_text,
            "segments": all_segments,
            "num_devices": len(audio_files)
        }

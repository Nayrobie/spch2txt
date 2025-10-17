"""
Audio transcription functionality using OpenAI Whisper.
"""

import whisper
import numpy as np
from typing import Dict
import os


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

    def transcribe(self, audio_input, language: str | None = None,
                   **kwargs) -> Dict:
        """
        Transcribe audio to text.

        Args:
            audio_input: Can be filepath (str) or numpy array
            language: Language code (default: detects automatically)
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
    
    def _is_valid_transcription(self, text: str, no_speech_prob: float = 0.0) -> bool:
        """
        Check if a transcription segment is valid (not a hallucination).
        
        Args:
            text: Transcribed text
            no_speech_prob: Probability that segment contains no speech
            
        Returns:
            True if segment appears to be valid speech
        """
        if no_speech_prob > 0.6:
            return False
        
        hallucinations = [
            "1.5%",
            "2.5%",
            "3.5%",
            "subscribe",
            ".",
            "...",
            "♪",
            "[BLANK_AUDIO]",
            "(blank)"
        ]
        
        text_lower = text.lower().strip()
        
        if text_lower in hallucinations:
            return False
        
        if len(text_lower) <= 3 and not any(c.isalpha() for c in text_lower):
            return False
        
        if len(set(text_lower.replace(" ", ""))) <= 2 and len(text_lower) < 10:
            return False
        
        return True

    def transcribe_multiple(
        self,
        audio_files: list,
        device_names: list
    ) -> Dict:
        """
        Transcribe multiple audio files separately and combine results.
        Uses timestamps to interleave segments in chronological order.
        
        Args:
            audio_files: List of audio file paths
            device_names: List of device names corresponding to audio files
            
        Returns:
            Dictionary with separate and combined transcripts
        """
        
        transcripts = []
        all_segments = []
        
        for i, (audio_file, device_name) in enumerate(zip(audio_files, device_names)):
            print(f"\nTranscribing device {i+1}: {device_name}")
            
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
                    
                    # Extract segments with timestamps
                    if "segments" in result:
                        filtered_count = 0
                        kept_count = 0
                        
                        for segment in result["segments"]:
                            segment_text = segment["text"].strip()
                            no_speech_prob = segment.get("no_speech_prob", 0.0)
                            
                            is_valid = self._is_valid_transcription(
                                segment_text, no_speech_prob
                            )
                            
                            if segment_text and is_valid:
                                all_segments.append({
                                    "start": segment["start"],
                                    "end": segment["end"],
                                    "text": segment_text,
                                    "speaker": speaker_label
                                })
                                kept_count += 1
                            else:
                                filtered_count += 1
                        
                        if filtered_count > 0:
                            print(f"  ⚠ Filtered {filtered_count} hallucination(s), kept {kept_count} segment(s)")
                        else:
                            print(f"  ✓ Kept {kept_count} segment(s)")
                else:
                    print("  ⚠ No speech detected")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
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

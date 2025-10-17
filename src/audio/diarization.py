"""
Speaker diarization functionality using pyannote.audio.
"""

import os
from typing import List, Tuple, Optional


class PyannoteDiarizer:
    """Handle speaker diarization using pyannote.audio."""

    def __init__(self, hf_token: str):
        """
        Initialize diarizer with Hugging Face token.

        Args:
            hf_token: Hugging Face authentication token
        """
        self.hf_token = hf_token
        self.pipeline = None
        self._pipeline_loaded = False

    def _load_pipeline(self):
        """Load the pyannote diarization pipeline (lazy loading)."""
        if self._pipeline_loaded:
            return
        
        try:
            # Import pyannote ONLY when loading pipeline (not at module import time)
            from pyannote.audio import Pipeline
            
            print("Loading diarization pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            self._pipeline_loaded = True
            print("✓ Diarization pipeline loaded")
        except Exception as e:
            self.pipeline = None
            self._pipeline_loaded = False
            print(f"⚠ Diarization pipeline failed to load: {e}")

    def diarize(self, audio_path: str) -> Optional[List[Tuple[float, float, str]]]:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of (start_time, end_time, speaker_label) tuples or None if failed
        """
        # Lazy load the pipeline on first use
        if not self._pipeline_loaded:
            self._load_pipeline()
        
        if not self.pipeline:
            print("⚠ Diarization pipeline not available")
            return None

        if not os.path.exists(audio_path):
            print(f"⚠ Audio file not found: {audio_path}")
            return None

        try:
            # Import ProgressHook only when needed
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            
            # Check if audio file is empty or too small
            file_size = os.path.getsize(audio_path)
            if file_size < 1000:
                print(f"⚠ Audio file too small for diarization: {audio_path}")
                return []

            with ProgressHook() as hook:
                diarization = self.pipeline(audio_path, hook=hook)

                segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append((turn.start, turn.end, speaker))

                return segments

        except Exception as e:
            print(f"⚠ Diarization failed for {audio_path}: {e}")
            return None


def assign_speakers_to_segments(
    whisper_segments: List[dict],
    diarization_segments: List[Tuple[float, float, str]]
) -> List[dict]:
    """
    Assign speaker labels to Whisper transcription segments using overlap.

    Args:
        whisper_segments: List of Whisper segments with 'start', 'end', 'text'
        diarization_segments: List of (start, end, speaker) tuples from diarization

    Returns:
        List of segments with added 'speaker' field
    """
    if not diarization_segments:
        return whisper_segments

    result_segments = []

    for segment in whisper_segments:
        seg_start = segment['start']
        seg_end = segment['end']

        # Find overlapping diarization segments
        overlaps = []
        for dia_start, dia_end, speaker in diarization_segments:
            # Calculate overlap
            overlap_start = max(seg_start, dia_start)
            overlap_end = min(seg_end, dia_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                overlaps.append((overlap_duration, speaker))

        # Assign speaker with most overlap
        if overlaps:
            overlaps.sort(reverse=True, key=lambda x: x[0])
            assigned_speaker = overlaps[0][1]
        else:
            assigned_speaker = "UNKNOWN"

        result_segment = segment.copy()
        result_segment['speaker'] = assigned_speaker
        result_segments.append(result_segment)

    return result_segments

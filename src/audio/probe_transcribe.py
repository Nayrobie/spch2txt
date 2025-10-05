import whisper

"""
Transcribe probe audio from 10 sec probe recording.

How to run:
    poetry run python src/audio/probe_transcribe.py
"""

import whisper
model = whisper.load_model("small")
result = model.transcribe("out.wav")
print(result["text"])